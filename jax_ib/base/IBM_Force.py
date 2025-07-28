import jax.numpy as jnp
import jax
from jax_ib.base import grids
from jax import debug as jax_debug

def calculate_tension_force(xp, yp, sigma):
    # (This function is correct and remains unchanged)
    dxL = jnp.roll(xp, -1) - xp
    dyL = jnp.roll(yp, -1) - yp
    dS = jnp.sqrt(dxL**2 + dyL**2) + 1e-9
    l_hat_x, l_hat_y = dxL / dS, dyL / dS
    l_hat_x_prev, l_hat_y_prev = jnp.roll(l_hat_x, 1), jnp.roll(l_hat_y, 1)
    force_x = sigma * (l_hat_x - l_hat_x_prev)
    force_y = sigma * (l_hat_y - l_hat_y_prev)
    return force_x, force_y

def calculate_penalty_force(xp, yp, Ym_x, Ym_y, Kp):
    """Calculates the penalty spring force F = Kp(Y - X) from Eq. (4)."""
    force_x = Kp * (Ym_x - xp)
    force_y = Kp * (Ym_y - yp)
    return force_x, force_y

def integrate_trapz(integrand,dx,dy):
    return jnp.trapz(jnp.trapz(integrand,dx=dx),dx=dy)

def Integrate_Field_Fluid_Domain(field):
    grid = field.grid
    dxEUL, dyEUL = grid.step[0], grid.step[1]
    return integrate_trapz(field.data,dxEUL,dyEUL)

# --- REWRITTEN FUNCTION for Deformable Body ---
def IBM_force_GENERAL(field, Xi, particle, discrete_fn):
    
    grid = field.grid
    offset = field.offset
    X, Y = grid.mesh(offset)
    dxEUL = grid.step[0]
    
    # Get current particle state
    xp, yp = particle.xp, particle.yp
    Ym_x, Ym_y = particle.Ym_x, particle.Ym_y
    Kp = particle.stiffness
    sigma = particle.sigma

    # --- Calculate total physical force ON the fluid ---
    # The force exerted BY the particle ON the fluid is the sum of the
    # penalty force and the tension force acting on the fluid markers.
    
    # 1. Penalty spring force (Eq. 4) on the fluid marker
    penalty_force_x, penalty_force_y = calculate_penalty_force(xp, yp, Ym_x, Ym_y, Kp)
    
    # 2. Surface tension force (Eq. 7) on the fluid marker
    tension_force_x, tension_force_y = jnp.zeros_like(xp), jnp.zeros_like(yp)
    if sigma is not None and sigma > 0.0:
        tension_force_x, tension_force_y = calculate_tension_force(xp, yp, sigma)

    # Total force exerted on the fluid
    force_on_fluid_x = penalty_force_x + tension_force_x
    force_on_fluid_y = penalty_force_y + tension_force_y
    
    force_to_spread = force_on_fluid_x if Xi == 0 else force_on_fluid_y

    # --- Spread this physical force to the fluid grid ---
    x_i, y_i = jnp.roll(xp, -1), jnp.roll(yp, -1)
    dxL, dyL = x_i - xp, y_i - yp
    dS = jnp.sqrt(dxL**2 + dyL**2) + 1e-9

    # Convert point force to force density (F/L) for spreading
    force_density_to_spread = force_to_spread / dS
    
    def calc_force(F_density, xp_pt, yp_pt, dss_pt):
        return F_density * discrete_fn(jnp.sqrt((xp_pt - X)**2 + (yp_pt - Y)**2), 0, dxEUL) * dss_pt

    def foo(tree_arg):
        F_density, xp_pt, yp_pt, dss_pt = tree_arg
        return calc_force(F_density, xp_pt, yp_pt, dss_pt)
    
    def foo_pmap(tree_arg):
        return jnp.sum(jax.vmap(foo, in_axes=1)(tree_arg), axis=0)
        
    divider = jax.device_count()
    n = len(xp) // divider
    mapped = []
    for i in range(divider):
       mapped.append([force_density_to_spread[i*n:(i+1)*n], xp[i*n:(i+1)*n], yp[i*n:(i+1)*n], dS[i*n:(i+1)*n]])

    return jnp.sum(jax.pmap(foo_pmap)(jnp.array(mapped)), axis=0)

# --- REWRITTEN HIGHER-LEVEL FUNCTIONS ---
def IBM_Multiple_NEW(field, Xi, particles, discrete_fn):
    force = jnp.zeros_like(field.data)
    # This loop is now over a list of particle objects if you have multiple
    for particle in particles.particles: # Assuming particles is now a container
        force += IBM_force_GENERAL(field, Xi, particle, discrete_fn)
    return grids.GridArray(force, field.offset, field.grid)

def calc_IBM_force_NEW_MULTIPLE(all_variables, discrete_fn, dt):
    velocity = all_variables.velocity
    particles = all_variables.particles
    axis = [0, 1]
    # Note: dt is unused here now, but kept for API consistency with the solver.
    ibm_forcing = lambda field, Xi: IBM_Multiple_NEW(field, Xi, particles, discrete_fn)
    
    return tuple(grids.GridVariable(ibm_forcing(field, Xi), field.bc) for field, Xi in zip(velocity, axis))
