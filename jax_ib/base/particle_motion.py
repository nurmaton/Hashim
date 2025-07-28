from jax_ib.base import particle_class as pc
from jax_ib.base import interpolation  # <-- Already imported
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

# === UPDATED FREE PARTICLE UPDATE FUNCTION WITH BILINEAR INTERPOLATION ===
def Update_particle_position_Free(all_variables, dt):
    """
    Updates particle positions using bilinearly interpolated local IBM/fluid velocity.
    Each marker moves according to its local velocity.
    """
    particles = all_variables.particles
    Drag = all_variables.Drag
    velocity = all_variables.velocity  # tuple of GridVariable
    pressure = all_variables.pressure
    Step_count = all_variables.Step_count + 1
    MD_var = all_variables.MD_var

    particle_centers = particles.particle_center  # shape [N,2]
    xp = particle_centers[:, 0]
    yp = particle_centers[:, 1]

    # Interpolate local velocity for each marker using your built-in function (order=1: bilinear)
    vx_marker = jax.vmap(lambda x, y: interpolation.point_interpolation(jnp.array([x, y]), velocity[0].array, order=1))(xp, yp)
    vy_marker = jax.vmap(lambda x, y: interpolation.point_interpolation(jnp.array([x, y]), velocity[1].array, order=1))(xp, yp)

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
