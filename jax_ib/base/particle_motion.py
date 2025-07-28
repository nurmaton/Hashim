from jax_ib.base import particle_class as pc
import jax
import jax.numpy as jnp

def Update_particle_position_Multiple_and_MD_Step(step_fn, all_variables, dt):
    particles = all_variables.particles
    Drag = all_variables.Drag
    velocity = all_variables.velocity
    current_t = velocity[0].bc.time_stamp
    particle_centers = particles.particle_center
    Displacement_EQ = particles.Displacement_EQ
    displacement_param = particles.displacement_param
    New_eq = lambda t: Displacement_EQ(displacement_param, t)
    dx_dt = jax.jacrev(New_eq)

    U0 = dx_dt(current_t)
    Newparticle_center = jnp.array([particle_centers[:, 0] + dt * U0[0], particle_centers[:, 1] + dt * U0[1]]).T
    mygrids = particles.Grid
    param_geometry = particles.geometry_param
    shape_fn = particles.shape
    pressure = all_variables.pressure
    Step_count = all_variables.Step_count + 1
    rotation_param = particles.rotation_param

    MD_var = step_fn(all_variables)

    New_particles = pc.particle(Newparticle_center, param_geometry, displacement_param, rotation_param, mygrids, shape_fn, Displacement_EQ, particles.Rotation_EQ)
    return pc.All_Variables(New_particles, velocity, pressure, Drag, Step_count, MD_var)

def Update_particle_position_Multiple(all_variables, dt):
    particles = all_variables.particles
    Drag = all_variables.Drag
    velocity = all_variables.velocity
    current_t = velocity[0].bc.time_stamp
    particle_centers = particles.particle_center
    Displacement_EQ = particles.Displacement_EQ
    displacement_param = particles.displacement_param
    New_eq = lambda t: Displacement_EQ(displacement_param, t)
    dx_dt = jax.jacrev(New_eq)

    U0 = dx_dt(current_t)
    Newparticle_center = jnp.array([particle_centers[:, 0] + dt * U0[0], particle_centers[:, 1] + dt * U0[1]]).T
    mygrids = particles.Grid
    param_geometry = particles.geometry_param
    shape_fn = particles.shape
    pressure = all_variables.pressure
    Step_count = all_variables.Step_count + 1
    rotation_param = particles.rotation_param

    MD_var = all_variables.MD_var

    New_particles = pc.particle(Newparticle_center, param_geometry, displacement_param, rotation_param, mygrids, shape_fn, Displacement_EQ, particles.Rotation_EQ)
    return pc.All_Variables(New_particles, velocity, pressure, Drag, Step_count, MD_var)

# === NEW FREE PARTICLE UPDATE FUNCTION BELOW ===

def Update_particle_position_Free(all_variables, dt):
    """
    Updates particle positions using the velocity of the fluid/IBM (not a prescribed equation).
    This allows particles to move and deform under the IBM force, e.g., tension.
    """
    particles = all_variables.particles
    Drag = all_variables.Drag
    velocity = all_variables.velocity
    pressure = all_variables.pressure
    Step_count = all_variables.Step_count + 1
    MD_var = all_variables.MD_var

    # For a flexible interface, particle_centers could be marker positions (not just the center of mass)
    particle_centers = particles.particle_center

    # You need to interpolate the velocity field at the marker positions.
    # For this demo, we'll just use the velocity at grid index [0,0] for both x and y (replace with your own interpolation)
    # For a real setup, use your existing interpolation function.

    # Dummy code (replace with interpolation!):
    # v_x = float(velocity[0].data[0, 0])
    # v_y = float(velocity[1].data[0, 0])
    v_x = velocity[0].data[0, 0]
    v_y = velocity[1].data[0, 0]
    # Update all centers (broadcast for all particles)
    Newparticle_center = particle_centers + dt * jnp.array([v_x, v_y])

    mygrids = particles.Grid
    param_geometry = particles.geometry_param
    shape_fn = particles.shape
    rotation_param = particles.rotation_param

    # Set Displacement_EQ and Rotation_EQ to None to disable scripted motion
    New_particles = pc.particle(Newparticle_center, param_geometry, particles.displacement_param, rotation_param,
                                mygrids, shape_fn, None, None)
    return pc.All_Variables(New_particles, velocity, pressure, Drag, Step_count, MD_var)
