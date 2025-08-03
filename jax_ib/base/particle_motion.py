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
    Updates the state of ALL massive, deformable particles for one time step.
    This corrected version iterates through every particle in the container.
    """
    particles = all_variables.particles.particles
    velocity_field = all_variables.velocity
    discrete_fn = lambda dist, center, width: convolution_functions.delta_approx_logistjax(dist, center, width)

    updated_particle_list = []

    # Loop over all particles (THIS IS THE FIX)
    for particle in particles:
        U_fluid_x_pts, U_fluid_y_pts = interpolate_velocity_to_surface(
            velocity_field, particle.xp, particle.yp, discrete_fn
        )
        new_xp = particle.xp + dt * U_fluid_x_pts
        new_yp = particle.yp + dt * U_fluid_y_pts

        penalty_force_x, penalty_force_y = IBM_Force.calculate_penalty_force(
            particle.xp, particle.yp, particle.Ym_x, particle.Ym_y, particle.stiffness
        )
        net_force_on_Ym_x = -penalty_force_x
        net_force_on_Ym_y = -penalty_force_y - (particle.mass_per_marker * gravity_g)
        accel_x = net_force_on_Ym_x / particle.mass_per_marker
        accel_y = net_force_on_Ym_y / particle.mass_per_marker
        new_Vm_x = particle.Vm_x + dt * accel_x
        new_Vm_y = particle.Vm_y + dt * accel_y
        new_Ym_x = particle.Ym_x + dt * new_Vm_x
        new_Ym_y = particle.Ym_y + dt * new_Vm_y
        new_center = jnp.array([[jnp.mean(new_Ym_x), jnp.mean(new_Ym_y)]])

        updated_particle = pc.particle(
            xp=new_xp, yp=new_yp, Ym_x=new_Ym_x, Ym_y=new_Ym_y,
            Vm_x=new_Vm_x, Vm_y=new_Vm_y,
            mass_per_marker=particle.mass_per_marker, stiffness=particle.stiffness, sigma=particle.sigma,
            particle_center=new_center, geometry_param=particle.geometry_param,
            Grid=particle.Grid, shape=particle.shape
        )
        updated_particle_list.append(updated_particle)

    new_particles_container = pc.particle_lista(particles=updated_particle_list)

    return pc.All_Variables(
        particles=new_particles_container,
        velocity=all_variables.velocity,
        pressure=all_variables.pressure,
        Drag=all_variables.Drag,
        Step_count=all_variables.Step_count + 1,
        MD_var=all_variables.MD_var
    )
