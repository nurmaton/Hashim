import jax.numpy as jnp
import jax
from jax_ib.base import grids
from jax import debug as jax_debug

def calculate_tension_force(xp, yp, sigma):
    dxL = jnp.roll(xp, -1) - xp
    dyL = jnp.roll(yp, -1) - yp
    dS = jnp.sqrt(dxL**2 + dyL**2) + 1e-9
    l_hat_x, l_hat_y = dxL / dS, dyL / dS
    l_hat_x_prev, l_hat_y_prev = jnp.roll(l_hat_x, 1), jnp.roll(l_hat_y, 1)
    force_x = sigma * (l_hat_x - l_hat_x_prev)
    force_y = sigma * (l_hat_y - l_hat_y_prev)
    return force_x, force_y

def calculate_penalty_force(xp, yp, Ym_x, Ym_y, Kp):
    force_x = Kp * (Ym_x - xp)
    force_y = Kp * (Ym_y - yp)
    return force_x, force_y

def integrate_trapz(integrand,dx,dy):
    return jnp.trapz(jnp.trapz(integrand,dx=dx),dx=dy)

def Integrate_Field_Fluid_Domain(field):
    grid = field.grid
    dxEUL, dyEUL = grid.step[0], grid.step[1]
    return integrate_trapz(field.data,dxEUL,dyEUL)

def IBM_force_GENERAL(field, Xi, particle, discrete_fn):
    grid = field.grid
    offset = field.offset
    X, Y = grid.mesh(offset)
    dxEUL = grid.step[0]
    
    xp, yp = particle.xp, particle.yp
    Ym_x, Ym_y = particle.Ym_x, particle.Ym_y
    Kp = particle.stiffness
    sigma = particle.sigma

    penalty_force_x, penalty_force_y = calculate_penalty_force(xp, yp, Ym_x, Ym_y, Kp)
    
    # --- MODIFIED SECTION: Replaced Python 'if' with 'jax.lax.cond' ---
    # Define functions for the true and false branches of the condition.
    # They must have the same input/output structure.
    def compute_tension(operands):
        x, y, s = operands
        return calculate_tension_force(x, y, s)

    def no_tension(operands):
        x, y, s = operands
        return jnp.zeros_like(x), jnp.zeros_like(y)

    # Use jax.lax.cond for JAX-compatible conditional logic.
    tension_force_x, tension_force_y = jax.lax.cond(
        sigma > 0.0,
        compute_tension,
        no_tension,
        (xp, yp, sigma) # Operands passed to the selected function
    )
    # --- END MODIFIED SECTION ---

    force_on_fluid_x = penalty_force_x + tension_force_x
    force_on_fluid_y = penalty_force_y + tension_force_y
    
    force_to_spread = force_on_fluid_x if Xi == 0 else force_on_fluid_y

    x_i, y_i = jnp.roll(xp, -1), jnp.roll(yp, -1)
    dxL, dyL = x_i - xp, y_i - yp
    dS = jnp.sqrt(dxL**2 + dyL**2) + 1e-9

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


def IBM_Multiple_NEW(field, Xi, particles_container, discrete_fn):
    particle = particles_container.particles[0]
    force = IBM_force_GENERAL(field, Xi, particle, discrete_fn)
    return grids.GridArray(force, field.offset, field.grid)


def calc_IBM_force_NEW_MULTIPLE(all_variables, discrete_fn, dt):
    velocity = all_variables.velocity
    particles = all_variables.particles
    axis = [0, 1]
    ibm_forcing = lambda field, Xi: IBM_Multiple_NEW(field, Xi, particles, discrete_fn)
    
    return tuple(grids.GridVariable(ibm_forcing(field, Xi), field.bc) for field, Xi in zip(velocity, axis))
