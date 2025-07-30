from jax_ib.base import particle_class as pc
from jax_ib.base import interpolation
from jax_ib.base import IBM_Force
from jax_ib.base import convolution_functions # <-- NEW IMPORT
import jax
import jax.numpy as jnp

# --- NEW HELPER FUNCTION ---
# This is a general, reusable version of the surface_fn from your notebook.
# It interpolates the velocity field to the particle markers using the IBM integral.
def interpolate_velocity_to_surface(velocity_field, xp, yp, discrete_fn):
    """
    Interpolates the Eulerian velocity field to the Lagrangian particle markers.
    This function implements Eq. (3) from the Sustiel & Grier paper.

    Args:
        velocity_field: A tuple of (u, v) GridVariable velocity components.
        xp, yp: The x and y coordinates of the particle markers.
        discrete_fn: The discrete delta function kernel.

    Returns:
        A tuple of (u_at_markers, v_at_markers).
    """
    # Create a simplified surface_fn lambda for a single velocity component
    _surface_fn_component = lambda field, xp_pts, yp_pts: convolution_functions.new_surf_fn(field, xp_pts, yp_pts, discrete_fn)

    # Apply the function to both the u and v components of the velocity field
    u_at_markers = _surface_fn_component(velocity_field[0], xp, yp)
    v_at_markers = _surface_fn_component(velocity_field[1], xp, yp)
    
    return u_at_markers, v_at_markers

# --- REWRITTEN AND CORRECTED UPDATE FUNCTION ---
def update_massive_deformable_particle(all_variables, dt, gravity_g=0.0):
    """
    Updates the state of a massive, deformable particle, now using the
    correct IBM velocity interpolation (surface_fn).
    """
    particles_container = all_variables.particles
    velocity_field = all_variables.velocity
    
    # The delta function is the kernel for interpolation.
    # This matches the function you define in your notebook.
    discrete_fn = lambda dist, center, width: convolution_functions.delta_approx_logistjax(dist, center, width)

    # This function will operate on the first particle in the list.
    # This can be easily extended to a loop or vmap for multi-particle simulations.
    particle = particles_container.particles[0] 
    
    # --- 1. Advect Fluid Markers (X) using the IBM Integral (surface_fn) ---
    # This is the corrected, physically consistent way to get the fluid velocity.
    U_fluid_x_pts, U_fluid_y_pts = interpolate_velocity_to_surface(
        velocity_field, particle.xp, particle.yp, discrete_fn
    )
    
    # Update marker positions with a forward Euler step
    new_xp = particle.xp + dt * U_fluid_x_pts
    new_yp = particle.yp + dt * U_fluid_y_pts

    # --- 2. Update Mass Markers (Y) using Newton's Second Law (Eq. 5) ---
    # This section remains the same, as it's based on the spring forces.
    penalty_force_x, penalty_force_y = IBM_Force.calculate_penalty_force(
        particle.xp, particle.yp, particle.Ym_x, particle.Ym_y, particle.stiffness
    )

    # Net force on mass markers is the reaction to the penalty force + gravity.
    net_force_on_Ym_x = -penalty_force_x
    net_force_on_Ym_y = -penalty_force_y - (particle.mass_per_marker * gravity_g)

    # a = F/m
    accel_x = net_force_on_Ym_x / particle.mass_per_marker
    accel_y = net_force_on_Ym_y / particle.mass_per_marker

    # v_new = v_old + a*dt
    new_Vm_x = particle.Vm_x + dt * accel_x
    new_Vm_y = particle.Vm_y + dt * accel_y
    
    # y_new = y_old + v_new*dt
    new_Ym_x = particle.Ym_x + dt * new_Vm_x
    new_Ym_y = particle.Ym_y + dt * new_Vm_y
    
    # Update the overall particle center for tracking purposes
    new_center = jnp.array([[jnp.mean(new_Ym_x), jnp.mean(new_Ym_y)]])

    # --- 3. Create the new particle object with the updated state ---
    # We use the full constructor to create the new immutable particle object.
    updated_particle = pc.particle(
        xp=new_xp, yp=new_yp, Ym_x=new_Ym_x, Ym_y=new_Ym_y,
        Vm_x=new_Vm_x, Vm_y=new_Vm_y,
        mass_per_marker=particle.mass_per_marker, stiffness=particle.stiffness, sigma=particle.sigma,
        particle_center=new_center, geometry_param=particle.geometry_param,
        Grid=particle.Grid, shape=particle.shape,
        Displacement_EQ=None, Rotation_EQ=None, displacement_param=None, rotation_param=None
    )
    
    # --- 4. Rebuild the container holding the list of particles ---
    new_particles_container = pc.particle_lista(particles=[updated_particle])
    
    # --- 5. Return a new All_Variables instance with the updated fields ---
    return pc.All_Variables(
        particles=new_particles_container,
        velocity=all_variables.velocity, # velocity is updated by the main solver
        pressure=all_variables.pressure, # pressure is updated by the main solver
        Drag=all_variables.Drag,
        Step_count=all_variables.Step_count + 1,
        MD_var=all_variables.MD_var
    )
