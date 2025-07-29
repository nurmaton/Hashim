from jax_ib.base import particle_class as pc
from jax_ib.base import interpolation
from jax_ib.base import IBM_Force
import jax
import jax.numpy as jnp

def update_massive_deformable_particle(all_variables, dt, gravity_g=0.0):
    """
    Updates the state of a massive, deformable particle for one time step.
    This version is now fully consistent and JAX-compatible.
    """
    particles_container = all_variables.particles
    velocity_field = all_variables.velocity
    
    particle = particles_container.particles[0] 
    
    # --- 1. Advect Fluid Markers (X) ---
    U_fluid_x_pts = jax.vmap(interpolation.point_interpolation, in_axes=(0, None))(jnp.stack([particle.xp, particle.yp], axis=1), velocity_field[0].array)
    U_fluid_y_pts = jax.vmap(interpolation.point_interpolation, in_axes=(0, None))(jnp.stack([particle.xp, particle.yp], axis=1), velocity_field[1].array)
    
    new_xp = particle.xp + dt * U_fluid_x_pts
    new_yp = particle.yp + dt * U_fluid_y_pts

    # --- 2. Update Mass Markers (Y) ---
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

    # --- 3. Create the new particle object ---
    updated_particle = pc.particle(
        xp=new_xp, yp=new_yp, Ym_x=new_Ym_x, Ym_y=new_Ym_y,
        Vm_x=new_Vm_x, Vm_y=new_Vm_y,
        mass_per_marker=particle.mass_per_marker, stiffness=particle.stiffness, sigma=particle.sigma,
        particle_center=new_center, geometry_param=particle.geometry_param,
        Grid=particle.Grid, shape=particle.shape
    )
    
    # --- 4. Rebuild the container ---
    new_particles_container = pc.particle_lista(particles=[updated_particle])
    
    # --- 5. Return a new All_Variables instance with the updated fields ---
    # --- MODIFIED: Replaced .tree_replace() with the standard constructor ---
    return pc.All_Variables(
        particles=new_particles_container,
        velocity=all_variables.velocity, # velocity is updated by the main solver
        pressure=all_variables.pressure, # pressure is updated by the main solver
        Drag=all_variables.Drag,
        Step_count=all_variables.Step_count + 1,
        MD_var=all_variables.MD_var
    )
