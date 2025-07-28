import jax.numpy as jnp
import jax
from jax_ib.base import grids

def integrate_trapz(integrand, dx, dy):
    return jnp.trapz(jnp.trapz(integrand, dx=dx), dx=dy)

def Integrate_Field_Fluid_Domain(field):
    grid = field.grid
    dxEUL = grid.step[0]
    dyEUL = grid.step[1]
    return integrate_trapz(field.data, dxEUL, dyEUL)

def IBM_force_GENERAL(
    field, Xi, particle, dx_dt, domega_dt, rotation, dt, sigma=1e9
):
    """
    Compute IBM force (e.g., for surface tension) using current marker positions.
    """
    grid = field.grid
    offset = field.offset
    X, Y = grid.mesh(offset)
    dxEUL = grid.step[0]
    dyEUL = grid.step[1]
    current_t = field.bc.time_stamp

    # --- Get marker positions from particle object
    marker_xy = particle.marker_positions(current_t)  # shape (Nmarkers, 2)
    xp = marker_xy[:, 0]
    yp = marker_xy[:, 1]

    # velocity at marker
    surface_coord = [(xp)/dxEUL - offset[0], (yp)/dyEUL - offset[1]]
    velocity_at_surface = field.data[tuple(map(lambda arr: arr.astype(int), surface_coord))]  # Or use surface_fn if needed

    # Penalty force (if using rigid motion, typically off for tension-only)
    if Xi == 0:
        position_r = -(yp - particle.particle_center[0][1])
    elif Xi == 1:
        position_r = (xp - particle.particle_center[0][0])

    U0 = dx_dt(current_t)
    Omega = domega_dt(current_t)
    UP = U0[Xi] + Omega * position_r
    force_penalty = (UP - velocity_at_surface) / dt

    # --- Surface tension force addition ---
    N = xp.shape[0]
    i_next = jnp.roll(jnp.arange(N), -1)
    i_prev = jnp.roll(jnp.arange(N), 1)
    l_i = jnp.stack([xp[i_next] - xp, yp[i_next] - yp], axis=1)
    l_im1 = jnp.stack([xp - xp[i_prev], yp - yp[i_prev]], axis=1)
    l_i_norm = l_i / (jnp.linalg.norm(l_i, axis=1, keepdims=True) + 1e-12)
    l_im1_norm = l_im1 / (jnp.linalg.norm(l_im1, axis=1, keepdims=True) + 1e-12)
    force_sigma = -sigma * (l_i_norm - l_im1_norm)
    force_tension = force_sigma[:, Xi]

    # --- Total force ---
    force = force_tension  # Use tension only

    def debug_print_forces(_):
        jax.debug.print(
            "Step {t}: Penalty mean={pmean:.3g} max={pmax:.3g}; Tension mean={tmean:.3g} max={tmax:.3g}; Total mean={fmean:.3g} max={fmax:.3g}",
            t=current_t,
            pmean=jnp.mean(force_penalty),
            pmax=jnp.max(jnp.abs(force_penalty)),
            tmean=jnp.mean(force_tension),
            tmax=jnp.max(jnp.abs(force_tension)),
            fmean=jnp.mean(force),
            fmax=jnp.max(jnp.abs(force))
        )
    jax.lax.cond(
        jnp.abs(jnp.remainder(current_t, 10.0)) < 1e-6,
        debug_print_forces,
        lambda _: None,
        operand=None
    )

    x_i = jnp.roll(xp, -1)
    y_i = jnp.roll(yp, -1)
    dxL = x_i - xp
    dyL = y_i - yp
    dS = jnp.sqrt(dxL ** 2 + dyL ** 2)

    def calc_force(F, xp, yp, dxi, dyi, dss):
        return F * jnp.exp(-((xp - X)**2 + (yp - Y)**2) / (2*dxEUL**2)) * dss  # Use your discrete_fn here if needed

    def foo(tree_arg):
        F, xp, yp, dxi, dyi, dss = tree_arg
        return calc_force(F, xp, yp, dxi, dyi, dss)

    def foo_pmap(tree_arg):
        return jnp.sum(jax.vmap(foo, in_axes=1)(tree_arg), axis=0)

    divider = jax.device_count()
    n = len(xp) // divider
    mapped = []
    for i in range(divider):
        mapped.append([
            force[i * n:(i + 1) * n],
            xp[i * n:(i + 1) * n],
            yp[i * n:(i + 1) * n],
            dxL[i * n:(i + 1) * n],
            dyL[i * n:(i + 1) * n],
            dS[i * n:(i + 1) * n]
        ])
    return jnp.sum(jax.pmap(foo_pmap)(jnp.array(mapped)), axis=0)

def IBM_Multiple_NEW(field, Xi, particles, discrete_fn, surface_fn, dt, sigma=1.0):
    Nparticles = len(particles)
    force = jnp.zeros_like(field.data)
    for i in range(Nparticles):
        Xc = lambda t: particles[i].Displacement_EQ([particles[i].displacement_param], t)
        rotation = lambda t: particles[i].Rotation_EQ([particles[i].rotation_param], t)
        dx_dt = jax.jacrev(Xc)
        domega_dt = jax.jacrev(rotation)
        force += IBM_force_GENERAL(
            field, Xi, particles[i], dx_dt, domega_dt, rotation, dt, sigma
        )
    return grids.GridArray(force, field.offset, field.grid)

def calc_IBM_force_NEW_MULTIPLE(all_variables, discrete_fn, surface_fn, dt, sigma=1.0):
    velocity = all_variables.velocity
    particles = all_variables.particles
    axis = [0, 1]
    ibm_forcing = lambda field, Xi: IBM_Multiple_NEW(field, Xi, particles, discrete_fn, surface_fn, dt, sigma)
    return tuple(grids.GridVariable(ibm_forcing(field, Xi), field.bc) for field, Xi in zip(velocity, axis))
