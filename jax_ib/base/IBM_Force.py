# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implements the core physical force calculations for the deformable Immersed
Boundary Method (IBM).

This module is central to simulating dynamic, deformable bodies. It uses a
penalty-based approach, inspired by the model from Sustiel & Grier, to compute
the real physical forces a body exerts on the surrounding fluid.

The physical model is based on two sets of Lagrangian markers:
1.  **Mass-carrying markers (`Y`)**: These hold the object's inertia and are
    evolved by the Molecular Dynamics (`MD`) integrator.
2.  **Fluid-interacting boundary markers (`X`)**: These are massless points
    that define the object's boundary and directly experience forces.

The interaction forces calculated here are:
-   **Penalty Force**: A spring-like force `F = Kp(Y - X)` that models the
    body's internal elasticity, tethering the boundary markers to the mass
    markers.
-   **Surface Tension Force**: An inward-pulling force proportional to the local
    boundary curvature, which acts to minimize surface area.

The key steps implemented in this module are:
1.  Calculate the total physical force (penalty + tension) on each Lagrangian
    boundary marker.
2.  "Spread" this discrete Lagrangian force onto the continuous Eulerian fluid
    grid using a regularized (smooth) delta function. This results in a force
    field that is added to the Navier-Stokes equations in the main solver.
"""

import jax.numpy as jnp
import jax
from jax_ib.base import grids

# --- NEW HELPER FUNCTION ---
def calculate_tension_force(xp, yp, sigma):
    """
    Calculates the surface tension force based on boundary curvature.

    This function implements the surface tension model from the Sustiel & Grier
    paper (Eq. 7). The force is proportional to the local curvature of the
    particle's boundary, defined as the rate of change of the tangent vector
    with respect to the arc length (F = -sigma * d(l_hat)/ds). It acts to
    minimize the surface area and pulls the boundary inward.

    Args:
      xp: JAX array of x-coordinates of the Lagrangian markers.
      yp: JAX array of y-coordinates of the Lagrangian markers.
      sigma: The surface tension coefficient.

    Returns:
      A tuple (force_x, force_y) containing the x and y components of the
      surface tension force at each marker.
    """
    # Calculate the vector for each boundary segment (from point i to i+1).
    # jnp.roll(-1) efficiently gets the next point in the array.
    dxL = jnp.roll(xp, -1) - xp
    dyL = jnp.roll(yp, -1) - yp
    
    # Calculate the length of each segment (dS). Add a small epsilon for numerical stability to avoid division by zero.
    dS = jnp.sqrt(dxL**2 + dyL**2) + 1e-9
    
    # Calculate the unit tangent vector for each segment (l_hat = dX/dS).
    l_hat_x, l_hat_y = dxL / dS, dyL / dS
    
    # Get the unit tangent vector of the *previous* segment by rolling the array.
    l_hat_x_prev, l_hat_y_prev = jnp.roll(l_hat_x, 1), jnp.roll(l_hat_y, 1)
    
    # Approximate d(l_hat)/ds as the difference between the current and previous tangent vectors.
    # This difference vector points inward and its magnitude is proportional to the local curvature.
    force_x = sigma * (l_hat_x - l_hat_x_prev)
    force_y = sigma * (l_hat_y - l_hat_y_prev)
    return force_x, force_y

# --- NEW HELPER FUNCTION ---
def calculate_penalty_force(xp, yp, Ym_x, Ym_y, Kp):
    """
    Calculates the penalty spring force between two sets of markers.

    This implements the internal elasticity of the deformable body as described
    in the Sustiel & Grier paper (Eq. 4). It models the body as a set of
    mass-carrying markers (Y) connected by springs to a set of fluid-interacting
    markers (X). The force is a simple Hooke's Law spring force, F = Kp(Y - X),
    that pulls the fluid markers (X) back towards their corresponding mass
    markers (Y).

    Args:
      xp: JAX array of x-coordinates of the fluid-interacting markers (X).
      yp: JAX array of y-coordinates of the fluid-interacting markers (X).
      Ym_x: JAX array of x-coordinates of the mass-carrying markers (Y).
      Ym_y: JAX array of y-coordinates of the mass-carrying markers (Y).
      Kp: The penalty stiffness coefficient of the springs.

    Returns:
      A tuple (force_x, force_y) containing the x and y components of the
      penalty force at each marker.
    """
    # Calculate the vector difference between the mass markers and fluid markers.
    force_x = Kp * (Ym_x - xp)
    force_y = Kp * (Ym_y - yp)
    return force_x, force_y

# --- UNCHANGED UTILITY FUNCTIONS ---
def integrate_trapz(integrand, dx, dy):
    """Performs a 2D integration using the trapezoidal rule."""
    # This integrates first along one axis, then integrates the result along the other.
    return jnp.trapz(jnp.trapz(integrand, dx=dx), dx=dy)

def Integrate_Field_Fluid_Domain(field):
    """Integrates a GridArray field over its entire fluid domain."""
    grid = field.grid
    # Get the grid spacing in each direction.
    dxEUL, dyEUL = grid.step[0], grid.step[1]
    # Perform the 2D trapezoidal integration.
    return integrate_trapz(field.data, dxEUL, dyEUL)

# --- HEAVILY REWRITTEN CORE FUNCTION ---
def IBM_force_GENERAL(field, Xi, particle, discrete_fn):
    """
    Calculates and spreads the total physical IBM force for a single particle.

    WHY THE CHANGE WAS MADE:
    The OLD `IBM_force_GENERAL` was for a "Direct Forcing" method suitable for
    kinematically-prescribed (i.e., non-deformable) motion. The force was a
    fictitious correction term: `force = (U_particle - U_fluid) / dt`.
    The NEW version implements the "Penalty Method", which is essential for
    simulating truly dynamic, DEFORMABLE objects. The force is a REAL PHYSICAL
    force (elasticity + tension) exerted by the particle on the fluid.

    KEY DIFFERENCES:
    1. SIGNATURE: The new function takes a single stateful `particle` object,
       which is a cleaner, object-oriented approach.
    2. FORCE LOGIC: The force is the sum of the physical penalty and tension
       forces, not a kinematic correction term.

    Args:
      field: The GridArray of the fluid velocity component being updated.
      Xi: The axis index (0 for x-velocity, 1 for y-velocity).
      particle: A particle object containing the current state (positions, stiffness, etc.).
      discrete_fn: The discrete delta function kernel used to spread the
                   Lagrangian force to the Eulerian grid.

    Returns:
      A JAX array representing the IBM force field spread onto the Eulerian grid.
    """
    # --- Setup and Particle State Unpacking ---
    grid = field.grid
    offset = field.offset
    # Get the Eulerian grid coordinates.
    X, Y = grid.mesh(offset)
    dxEUL = grid.step[0]

    # Get the particle's CURRENT state directly from the object.
    # This is a major change from the OLD code, which calculated positions
    # from separate kinematic functions.
    xp, yp = particle.xp, particle.yp          # Fluid-interacting marker positions
    Ym_x, Ym_y = particle.Ym_x, particle.Ym_y  # Mass-carrying marker positions
    Kp = particle.stiffness                    # Penalty spring stiffness
    sigma = particle.sigma                     # Surface tension coefficient

    # --- Calculate Physical Forces on the Fluid ---
    
    # 1. Calculate the penalty spring force (Eq. 4) that tethers the
    #    fluid markers to the mass markers.
    penalty_force_x, penalty_force_y = calculate_penalty_force(xp, yp, Ym_x, Ym_y, Kp)

    # 2. Calculate the surface tension force (Eq. 7).
    #    We use jax.lax.cond for conditional execution. This is crucial for JIT
    #    compilation, as a standard Python `if sigma > 0:` would not be traceable by JAX.
    def compute_tension(operands):
        # Unpack operands and compute tension.
        x, y, s = operands
        return calculate_tension_force(x, y, s)

    def no_tension(operands):
        # If sigma is zero, return zero forces to avoid unnecessary computation.
        x, y, s = operands
        return jnp.zeros_like(x), jnp.zeros_like(y)

    tension_force_x, tension_force_y = jax.lax.cond(
        sigma > 0.0,      # The condition to check.
        compute_tension,  # The function to run if True.
        no_tension,       # The function to run if False.
        (xp, yp, sigma)   # The operands to pass to the chosen function.
    )

    # 3. The total force ON THE FLUID is the sum of these physical forces.
    force_on_fluid_x = penalty_force_x + tension_force_x
    force_on_fluid_y = penalty_force_y + tension_force_y

    # Select the correct force component (X or Y) for the current velocity field being updated.
    force_to_spread = force_on_fluid_x if Xi == 0 else force_on_fluid_y

    # --- Spreading Logic ---
    
    # Calculate segment lengths `dS` along the particle boundary.
    x_i, y_i = jnp.roll(xp, -1), jnp.roll(yp, -1)
    dxL, dyL = x_i - xp, y_i - yp
    dS = jnp.sqrt(dxL**2 + dyL**2) + 1e-9

    # IMPORTANT: Convert the point force (F) into a force density (f = F/dS).
    # When spreading, we are integrating F * delta(x) ds over the boundary.
    # By pre-dividing by dS, the `dss_pt` term in `calc_force` correctly
    # represents the `ds` differential element.
    force_density_to_spread = force_to_spread / dS
    
    # This inner function calculates the contribution of a single Lagrangian
    # point force to the entire Eulerian grid.
    def calc_force(F_density, xp_pt, yp_pt, dss_pt):
        # Force contribution = (Force_density) * delta_function * (segment_length_ds)
        return F_density * discrete_fn(jnp.sqrt((xp_pt - X)**2 + (yp_pt - Y)**2), 0, dxEUL) * dss_pt

    # Define a function to be vectorized. It just unpacks the arguments.
    def foo(tree_arg):
        F_density, xp_pt, yp_pt, dss_pt = tree_arg
        return calc_force(F_density, xp_pt, yp_pt, dss_pt)

    # Define a function that sums the contributions from all points on a single device.
    # jax.vmap vectorizes the 'foo' function, applying it to all markers efficiently.
    def foo_pmap(tree_arg):
        return jnp.sum(jax.vmap(foo, in_axes=1)(tree_arg), axis=0)

    # Manually split the data across available JAX devices for parallel processing.
    divider = jax.device_count()
    n = len(xp) // divider
    mapped = []
    for i in range(divider):
       mapped.append([force_density_to_spread[i*n:(i+1)*n], xp[i*n:(i+1)*n], yp[i*n:(i+1)*n], dS[i*n:(i+1)*n]])

    # jax.pmap runs 'foo_pmap' on each device in parallel.
    # The final jnp.sum aggregates the results from all devices into the final force field.
    return jnp.sum(jax.pmap(foo_pmap)(jnp.array(mapped)), axis=0)

# --- REWRITTEN HIGH-LEVEL FUNCTION ---
def IBM_Multiple_NEW(field, Xi, particles_container, discrete_fn):
    """
    High-level wrapper to calculate IBM force for particles in a container.

    OLD vs NEW:
    - The OLD function iterated through a list of particles, reconstructing complex
      kinematic functions for each one.
    - The NEW function is much simpler. For now, it assumes a single particle (as in
      the demonstration) and directly calls the new `IBM_force_GENERAL`. This can be
      extended to a loop for multiple particles if needed.

    Args:
      field: The velocity GridArray.
      Xi: The axis index.
      particles_container: A container object holding the particle(s).
      discrete_fn: The discrete delta function.

    Returns:
      A GridArray of the calculated force field.
    """
    # Get the first (and currently, only) particle object from the container.
    particle = particles_container.particles[0]
    
    # Calculate and spread the forces from this particle.
    force = IBM_force_GENERAL(field, Xi, particle, discrete_fn)
    
    # Wrap the resulting array in a GridArray, preserving grid and offset info.
    return grids.GridArray(force, field.offset, field.grid)

# --- REWRITTEN HIGH-LEVEL FUNCTION ---
def calc_IBM_force_NEW_MULTIPLE(all_variables, discrete_fn, dt):
    """
    Top-level entry point to calculate the IBM forcing term for the solver.

    OLD vs NEW:
    - The signature is simplified. It no longer needs `surface_fn` because the
      force is no longer based on the fluid velocity at the surface.
    - The `dt` argument is now unused in the penalty force calculation, as the force
      is physical (F=kx) not a kinematic correction (F=dv/dt). It is kept for
      API consistency with the main solver steps.

    Args:
      all_variables: A container with all current simulation state (velocity, particles).
      discrete_fn: The discrete delta function kernel.
      dt: The time step (kept for API consistency, but unused).

    Returns:
      A tuple of GridVariables, one for each velocity component (e.g., (Fx, Fy)),
      representing the IBM forcing term.
    """
    # Unpack variables from the main state object.
    velocity = all_variables.velocity
    particles = all_variables.particles
    axis = [0, 1]  # The axes to compute forces for.

    # Create a lambda function for the forcing calculation.
    ibm_forcing = lambda field, Xi: IBM_Multiple_NEW(field, Xi, particles, discrete_fn)

    # Apply the forcing function to each velocity component (u, v) and return
    # a tuple of new GridVariables containing the force fields.
    return tuple(grids.GridVariable(ibm_forcing(field, Xi), field.bc) for field, Xi in zip(velocity, axis))
