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

# === NEW FREE PARTICLE UPDATE FUNCTION WITH INTERPOLATION ===

def interpolate_velocity_nearest(velocity, xp, yp, grid):
    """
    Nearest neighbor interpolation of velocity field at given marker positions.
    velocity: tuple of (vx, vy) as GridVariables
    xp, yp: marker positions
    grid: grid object with .step and .domain
    Returns: vx_marker, vy_marker (arrays, same length as xp/yp)
    """
    dx, dy = grid.step
    x0, y0 = grid.domain[0][0], grid.domain[1][0]
    # Convert positions to grid indices
    ix = jnp.clip(jnp.round((xp - x0) / dx).astype(int), 0, velocity[0].data.shape[0] - 1)
    iy = jnp.clip(jnp.round((yp - y0) / dy).astype(int), 0, velocity[1].data.shape[1] - 1)
    vx_marker = velocity[0].data[ix, iy]
    vy_marker = velocity[1].data[ix, iy]
    return vx_marker, vy_marker

def Update_particle_position_Free(all_variables, dt):
    """
    Updates particle positions using the velocity of the fluid/IBM (not a prescribed equation).
    Each marker moves according to its interpolated local velocity.
    """
    particles = all_variables.particles
    Drag = all_variables.Drag
    velocity = all_variables.velocity
    pressure = all_variables.pressure
    Step_count = all_variables.Step_count + 1
    MD_var = all_variables.MD_var

    particle_centers = particles.particle_center

    # Interpolate local velocity for each marker
    xp = particle_centers[:, 0]
    yp = particle_centers[:, 1]
    vx_marker, vy_marker = interpolate_velocity_nearest(velocity, xp, yp, particles.Grid)
    new_xp = xp + dt * vx_marker
    new_yp = yp + dt * vy_marker
    Newparticle_center = jnp.stack([new_xp, new_yp], axis=1)

    mygrids = particles.Grid
    param_geometry = particles.geometry_param
    shape_fn = particles.shape
    rotation_param = particles.rotation_param

    New_particles = pc.particle(Newparticle_center, param_geometry, particles.displacement_param, rotation_param,
                                mygrids, shape_fn, particles.Displacement_EQ, particles.Rotation_EQ)
    return pc.All_Variables(New_particles, velocity, pressure, Drag, Step_count, MD_var)
