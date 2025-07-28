from jax_ib.base import particle_class as pc
from jax_ib.base import interpolation
from jax_ib.base import IBM_Force # Import the whole module
import jax
import jax.numpy as jnp

# --- NEW FUNCTION to update the massive, deformable particle ---
def update_massive_deformable_particle(all_variables, dt, gravity_g=0.0):
    particles = all_variables.particles
    velocity_field = all_variables.velocity
    
    # We will update the first particle in the list. This can be extended to a loop.
    particle = particles.particles[0] 
    
    # --- 1. Advect Fluid Markers (X) with the local fluid velocity ---
    # Interpolate fluid velocity to the fluid markers X.
    # This is still a simplification, a fully vectorized interpolation would be ideal.
    U_fluid_x_pts = jax.vmap(interpolation.point_interpolation, in_axes=(0, None))(jnp.stack([particle.xp, particle.yp], axis=1), velocity_field[0].array)
    U_fluid_y_pts = jax.vmap(interpolation.point_interpolation, in_axes=(0, None))(jnp.stack([particle.xp, particle.yp], axis=1), velocity_field[1].array)
    
    new_xp = particle.xp + dt * U_fluid_x_pts
    new_yp = particle.yp + dt * U_fluid_y_pts

    # --- 2. Update Mass Markers (Y) using Newton's Second Law (Eq. 5) ---
    # F_i^m = Kp(Y-X)
    penalty_force_x, penalty_force_y = IBM_Force.calculate_penalty_force(
        particle.xp, particle.yp, particle.Ym_x, particle.Ym_y, particle.stiffness
    )

    # Net force on mass markers: F_net_Y = -F_i^m - Mg
    # Assuming gravity acts in the negative y-direction (ẑ in the paper)
    net_force_on_Ym_x = -penalty_force_x
    net_force_on_Ym_y = -penalty_force_y - (particle.mass_per_marker * gravity_g)

    # Acceleration a = F/m
    accel_x = net_force_on_Ym_x / particle.mass_per_marker
    accel_y = net_force_on_Ym_y / particle.mass_per_marker

    # Update velocity of mass markers (Velocity Verlet)
    new_Vm_x = particle.Vm_x + dt * accel_x
    new_Vm_y = particle.Vm_y + dt * accel_y
    
    # Update position of mass markers
    new_Ym_x = particle.Ym_x + dt * new_Vm_x
    new_Ym_y = particle.Ym_y + dt * new_Vm_y
    
    # Update the overall particle center for tracking
    new_center = jnp.array([[jnp.mean(new_Ym_x), jnp.mean(new_Ym_y)]])

    # --- 3. Create the new particle object with the updated state ---
    updated_particle = particle_class.particle(
        xp=new_xp, yp=new_yp, Ym_x=new_Ym_x, Ym_y=new_Ym_y,
        Vm_x=new_Vm_x, Vm_y=new_Vm_y,
        mass_per_marker=particle.mass_per_marker, stiffness=particle.stiffness, sigma=particle.sigma,
        particle_center=new_center, geometry_param=particle.geometry_param,
        Grid=particle.Grid, shape=particle.shape
    )
    
    # Update the list of particles in the main state object
    # In a multi-particle simulation, this would be a loop or a vmap.
    updated_particles_list = particles.particles.copy()
    updated_particles_list[0] = updated_particle
    new_particles_container = particles.tree_replace(particles=updated_particles_list)
    
    return all_variables.tree_replace(particles=new_particles_container, Step_count=all_variables.Step_count + 1)
