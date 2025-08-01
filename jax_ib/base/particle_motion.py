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
Implements the dynamic equations of motion for deformable particles.

This module is responsible for updating the state (positions and velocities) of
the Lagrangian particles based on the physical forces acting on them. This
represents the "solid-side" of the Fluid-Structure Interaction (FSI) problem,
implementing the physics described by Sustiel & Grier.

The core function, `update_massive_deformable_particle`, advances the particle
state for one time step by implementing two key physical principles:

1.  **Advection of the Boundary (Eq. 3)**: The massless, fluid-interacting
    boundary markers (`X`) are advected with the local fluid velocity. This is
    achieved by first interpolating the Eulerian fluid velocity field onto the
    Lagrangian marker locations using a discrete convolution.

2.  **Newtonian Dynamics of the Mass (Eq. 5)**: The mass-carrying markers (`Y`)
    are accelerated according to Newton's Second Law (`F=ma`). The net force
    includes the reaction to the internal penalty-spring force and any body
    forces like gravity. A semi-implicit Euler scheme is used to update the
    positions and velocities of these markers.

This module is a critical departure from the previous kinematic model, as the
particle's motion is now a direct, dynamic consequence of the simulated physics.
"""

from jax_ib.base import particle_class as pc
from jax_ib.base import interpolation
from jax_ib.base import IBM_Force
from jax_ib.base import convolution_functions
import jax
import jax.numpy as jnp

# --- HELPER FUNCTION ---
def interpolate_velocity_to_surface(velocity_field, xp, yp, discrete_fn):
    """
    Interpolates the Eulerian velocity field to the Lagrangian particle markers.

    This function implements the core fluid-to-solid coupling of the Immersed
    Boundary Method, representing Equation (3) from the Sustiel & Grier paper.
    It calculates the velocity of the fluid *at the precise location* of each
    particle marker by performing a discrete convolution (an integral) of the
    fluid velocity field with the discrete delta function.

    Args:
      velocity_field: A GridVariableVector containing the (u, v) fluid velocity fields.
      xp: JAX array of x-coordinates of the Lagrangian markers.
      yp: JAX array of y-coordinates of the Lagrangian markers.
      discrete_fn: The discrete delta function kernel for interpolation.

    Returns:
      A tuple (u_at_markers, v_at_markers) of the interpolated velocity
      components at each particle marker.
    """
    # Define a helper lambda that calls the underlying convolution function.
    _surface_fn_component = lambda field, xp_pts, yp_pts: convolution_functions.new_surf_fn(field, xp_pts, yp_pts, discrete_fn)
    
    # Interpolate the u-component of velocity to the marker positions.
    u_at_markers = _surface_fn_component(velocity_field[0], xp, yp)
    # Interpolate the v-component of velocity to the marker positions.
    v_at_markers = _surface_fn_component(velocity_field[1], xp, yp)
    
    return u_at_markers, v_at_markers

# --- HEAVILY REWRITTEN CORE FUNCTION ---
def update_massive_deformable_particle(all_variables, dt, gravity_g=0.0):
    """
    Updates the state of a massive, deformable particle for one time step.

    OLD vs NEW COMPARISON & EXPLANATION:
    - WHY THE CHANGE WAS MADE: The OLD code was for a KINEMATIC model where motion
      was pre-defined. The NEW function implements the DYNAMIC equations of motion
      for the penalty IBM, where motion is CALCULATED from physical forces.
    - KEY DIFFERENCES:
      1. LOGIC: Old code used a math function (`Displacement_EQ`) to find the next
         position. New code implements the full physics: it advects the fluid markers (X)
         with the local fluid velocity (Eq. 3) and accelerates the mass markers (Y)
         using Newton's Second Law with penalty and gravity forces (Eq. 5).
      2. INTERPOLATION: Old code didn't need to know the fluid velocity at the particle.
         New code CRITICALLY relies on `interpolate_velocity_to_surface` to advect the
         fluid markers.
      3. STATE MANAGEMENT: Old code just updated the `particle_center`. New code returns a
         completely new `all_variables` object with the fully updated state of ALL
         marker positions and velocities. This is a robust, functional approach
         required for JAX.
    - WHY THE NEW METHOD IS BETTER: It simulates the actual physics of a deformable
      object, allowing complex behaviors like deformation and sedimentation to emerge naturally.

    Args:
      all_variables: The complete state of the simulation.
      dt: The time step duration.
      gravity_g: The acceleration due to gravity.

    Returns:
      A new `All_Variables` object with the particle state advanced by `dt`.
    """
    # Unpack the necessary data structures from the main state container.
    particles_container = all_variables.particles
    velocity_field = all_variables.velocity
    
    # The delta function is the kernel used for all interpolation/spreading operations.
    discrete_fn = lambda dist, center, width: convolution_functions.delta_approx_logistjax(dist, center, width)

    # This function operates on a single particle (the first in the list).
    # This could be extended to a loop or `jax.vmap` for multiple particles.
    particle = particles_container.particles[0] 
    
    # --- 1. Advect Fluid Markers (X) using the IBM Integral (Eq. 3) ---
    # First, find the velocity of the fluid at the location of the fluid markers.
    U_fluid_x_pts, U_fluid_y_pts = interpolate_velocity_to_surface(
        velocity_field, particle.xp, particle.yp, discrete_fn
    )
    
    # Update the fluid marker positions with a simple forward Euler step.
    # The fluid markers are massless and are simply carried along by the fluid flow.
    new_xp = particle.xp + dt * U_fluid_x_pts
    new_yp = particle.yp + dt * U_fluid_y_pts

    # --- 2. Update Mass Markers (Y) using Newton's Second Law (Eq. 5) ---
    # Calculate the internal spring force F = Kp(Y - X). This is the force
    # exerted by the mass markers (Y) on the fluid markers (X).
    penalty_force_x, penalty_force_y = IBM_Force.calculate_penalty_force(
        particle.xp, particle.yp, particle.Ym_x, particle.Ym_y, particle.stiffness
    )

    # By Newton's third law, the force on the mass markers is equal and opposite.
    # We also include the force of gravity.
    # F_net_on_Y = -F_penalty - M*g
    net_force_on_Ym_x = -penalty_force_x
    net_force_on_Ym_y = -penalty_force_y - (particle.mass_per_marker * gravity_g)

    # Calculate acceleration from F=ma -> a = F/m.
    accel_x = net_force_on_Ym_x / particle.mass_per_marker
    accel_y = net_force_on_Ym_y / particle.mass_per_marker

    # Update the mass marker velocities with a forward Euler step: v_new = v_old + a*dt.
    new_Vm_x = particle.Vm_x + dt * accel_x
    new_Vm_y = particle.Vm_y + dt * accel_y
    
    # Update the mass marker positions: y_new = y_old + v_new*dt.
    # Using the *new* velocity here makes this a semi-implicit Euler step,
    # which is slightly more stable than a standard forward Euler step.
    new_Ym_x = particle.Ym_x + dt * new_Vm_x
    new_Ym_y = particle.Ym_y + dt * new_Vm_y
    
    # Update the overall particle center based on the mean of mass markers (for tracking/diagnostics).
    new_center = jnp.array([[jnp.mean(new_Ym_x), jnp.mean(new_Ym_y)]])

    # --- 3. Create the new particle object with the updated state ---
    # We create a brand new `particle` object. This "out-of-place" update is
    # a core concept in functional programming and is required by JAX.
    updated_particle = pc.particle(
        xp=new_xp, yp=new_yp, Ym_x=new_Ym_x, Ym_y=new_Ym_y,
        Vm_x=new_Vm_x, Vm_y=new_Vm_y,
        mass_per_marker=particle.mass_per_marker, stiffness=particle.stiffness, sigma=particle.sigma,
        particle_center=new_center, geometry_param=particle.geometry_param,
        Grid=particle.Grid, shape=particle.shape
    )
    
    # --- 4. Rebuild the container holding the list of particles ---
    new_particles_container = pc.particle_lista(particles=[updated_particle])
    
    # --- 5. Return a new All_Variables instance with the updated fields ---
    # The entire state of the simulation is replaced with this new object.
    return pc.All_Variables(
        particles=new_particles_container,
        velocity=all_variables.velocity,
        pressure=all_variables.pressure,
        Drag=all_variables.Drag,
        Step_count=all_variables.Step_count + 1,
        MD_var=all_variables.MD_var
    )
