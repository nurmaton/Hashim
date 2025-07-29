import jax.numpy as jnp
import jax
from jax_ib.base import grids
from jax import debug as jax_debug

# --- NEW HELPER FUNCTION ---
# This function implements the surface tension force model from the Sustiel & Grier paper (Eq. 7).
# It's a physical force based on the curvature of the boundary.
def calculate_tension_force(xp, yp, sigma):
    """Calculates the surface tension force F = -sigma * (d(l_hat)/ds) at each marker."""
    dxL = jnp.roll(xp, -1) - xp
    dyL = jnp.roll(yp, -1) - yp
    dS = jnp.sqrt(dxL**2 + dyL**2) + 1e-9 # Segment lengths
    l_hat_x, l_hat_y = dxL / dS, dyL / dS # Unit tangent vectors l_hat_i
    
    # Get the previous segment's unit tangent vector l_hat_{i-1}
    l_hat_x_prev, l_hat_y_prev = jnp.roll(l_hat_x, 1), jnp.roll(l_hat_y, 1)
    
    # The force is the difference in the tangent vectors, which measures curvature.
    force_x = sigma * (l_hat_x - l_hat_x_prev)
    force_y = sigma * (l_hat_y - l_hat_y_prev)
    return force_x, force_y

# --- NEW HELPER FUNCTION ---
# This function implements the penalty spring force from the Sustiel & Grier paper (Eq. 4).
# It models the internal elasticity of the deformable body, tethering the fluid markers (X)
# to the mass-carrying markers (Y).
def calculate_penalty_force(xp, yp, Ym_x, Ym_y, Kp):
    """Calculates the penalty spring force F = Kp(Y - X)."""
    force_x = Kp * (Ym_x - xp)
    force_y = Kp * (Ym_y - yp)
    return force_x, force_y

# --- UNCHANGED UTILITY FUNCTIONS ---
def integrate_trapz(integrand,dx,dy):
    return jnp.trapz(jnp.trapz(integrand,dx=dx),dx=dy)

def Integrate_Field_Fluid_Domain(field):
    grid = field.grid
    dxEUL, dyEUL = grid.step[0], grid.step[1]
    return integrate_trapz(field.data,dxEUL,dyEUL)

# --- HEAVILY REWRITTEN CORE FUNCTION ---
# OLD vs NEW:
# - The function signature is completely different. It no longer takes kinematic functions
#   (dx_dt, rotation, etc.). Instead, it takes a single stateful 'particle' object.
# - The core logic has changed from calculating a velocity-mismatch force to calculating
#   real physical forces (penalty and tension).
def IBM_force_GENERAL(field, Xi, particle, discrete_fn):
    
    grid = field.grid
    offset = field.offset
    X, Y = grid.mesh(offset)
    dxEUL = grid.step[0]
    
    # Get the particle's CURRENT state (positions and properties) directly from the object.
    # This is a major change from the OLD code, which calculated positions from kinematic functions.
    xp, yp = particle.xp, particle.yp
    Ym_x, Ym_y = particle.Ym_x, particle.Ym_y
    Kp = particle.stiffness
    sigma = particle.sigma

    # --- NEW: Calculate Physical Forces on the Fluid ---
    # The core logic is now based on the Penalty IBM, not Direct Forcing.
    
    # 1. Calculate the penalty spring force (Eq. 4) that the mass markers exert on the fluid markers.
    penalty_force_x, penalty_force_y = calculate_penalty_force(xp, yp, Ym_x, Ym_y, Kp)
    
    # 2. Calculate the surface tension force (Eq. 7) on the fluid markers.
    # This section uses jax.lax.cond for efficient conditional execution in JAX.
    # It avoids using a standard Python 'if' statement, which is not JAX-jittable.
    def compute_tension(operands):
        x, y, s = operands
        return calculate_tension_force(x, y, s)

    def no_tension(operands):
        x, y, s = operands
        return jnp.zeros_like(x), jnp.zeros_like(y)

    tension_force_x, tension_force_y = jax.lax.cond(
        sigma > 0.0,
        compute_tension,
        no_tension,
        (xp, yp, sigma)
    )

    # 3. The total force ON THE FLUID is the sum of these physical forces.
    # OLD code: force was a fictitious term (UP - U_fluid)/dt.
    # NEW code: force is the physical F_penalty + F_tension.
    force_on_fluid_x = penalty_force_x + tension_force_x
    force_on_fluid_y = penalty_force_y + tension_force_y
    
    # Select the correct force component (X or Y) for the current field.
    force_to_spread = force_on_fluid_x if Xi == 0 else force_on_fluid_y

    # --- Spreading Logic (Structurally similar, but now spreading a physical force) ---
    x_i, y_i = jnp.roll(xp, -1), jnp.roll(yp, -1)
    dxL, dyL = x_i - xp, y_i - yp
    dS = jnp.sqrt(dxL**2 + dyL**2) + 1e-9

    # Convert the point force (F) into a force density (F/L) for correct spreading.
    force_density_to_spread = force_to_spread / dS
    
    # The spreading kernel and mapping logic remains the same.
    def calc_force(F_density, xp_pt, yp_pt, dss_pt):
        return -F_density * discrete_fn(jnp.sqrt((xp_pt - X)**2 + (yp_pt - Y)**2), 0, dxEUL) * dss_pt

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


# --- REWRITTEN HIGH-LEVEL FUNCTION ---
# OLD vs NEW:
# - The loop over particles and reconstruction of kinematic functions is gone.
# - It's now a simpler wrapper that gets the particle from the container and calls the new IBM_force_GENERAL.
# - This assumes a single particle simulation, which is what the demo is currently running.
def IBM_Multiple_NEW(field, Xi, particles_container, discrete_fn):
    # Get the first (and only) particle object from the container.
    particle = particles_container.particles[0]
    # Calculate and spread its forces.
    force = IBM_force_GENERAL(field, Xi, particle, discrete_fn)
    return grids.GridArray(force, field.offset, field.grid)


# --- REWRITTEN HIGH-LEVEL FUNCTION ---
# OLD vs NEW:
# - The signature is simplified. It no longer needs `surface_fn` because the force
#   is not based on fluid velocity at the surface.
# - The lambda function now calls the new, simpler `IBM_Multiple_NEW`.
def calc_IBM_force_NEW_MULTIPLE(all_variables, discrete_fn, dt):
    velocity = all_variables.velocity
    particles = all_variables.particles
    axis = [0, 1]
    # The lambda function is now simpler. `dt` is passed for API consistency but not used in the new force calculation.
    ibm_forcing = lambda field, Xi: IBM_Multiple_NEW(field, Xi, particles, discrete_fn)
    
    return tuple(grids.GridVariable(ibm_forcing(field, Xi), field.bc) for field, Xi in zip(velocity, axis))
