import jax.numpy as jnp
import jax
from jax_ib.base import grids
from jax import debug as jax_debug # <-- IMPORT JAX DEBUG

# --- This function remains the same ---
def calculate_tension_force(xp, yp, sigma):
    """
    Calculates the surface tension force on each Lagrangian marker.
    """
    dxL = jnp.roll(xp, -1) - xp
    dyL = jnp.roll(yp, -1) - yp
    dS = jnp.sqrt(dxL**2 + dyL**2) + 1e-9
    l_hat_x = dxL / dS
    l_hat_y = dyL / dS
    l_hat_x_prev = jnp.roll(l_hat_x, 1)
    l_hat_y_prev = jnp.roll(l_hat_y, 1)
    force_x = sigma * (l_hat_x - l_hat_x_prev)
    force_y = sigma * (l_hat_y - l_hat_y_prev)
    return force_x, force_y


def integrate_trapz(integrand,dx,dy):
    return jnp.trapz(jnp.trapz(integrand,dx=dx),dx=dy)


def Integrate_Field_Fluid_Domain(field):
    grid = field.grid
    dxEUL = grid.step[0]
    dyEUL = grid.step[1]
    return integrate_trapz(field.data,dxEUL,dyEUL)

# --- MODIFIED FUNCTION ---
# Now isolates the tension force and adds printing.
def IBM_force_GENERAL(field,Xi,particle_center,geom_param,Grid_p,shape_fn,discrete_fn,surface_fn,dx_dt,domega_dt,rotation,dt, sigma=0.0):
    
    grid = field.grid
    offset = field.offset
    X,Y = grid.mesh(offset)
    dxEUL = grid.step[0]
    dyEUL = grid.step[1]
    current_t = field.bc.time_stamp

    xp0,yp0 = shape_fn(geom_param,Grid_p)

    xp = (xp0)*jnp.cos(rotation(current_t))-(yp0)*jnp.sin(rotation(current_t))+particle_center[0]
    yp = (xp0)*jnp.sin(rotation(current_t))+(yp0 )*jnp.cos(rotation(current_t))+particle_center[1]
    
    velocity_at_surface = surface_fn(field,xp,yp)
    
    if Xi==0:
        position_r = -(yp-particle_center[1])
    elif Xi==1:
        position_r = (xp-particle_center[0])
    
    U0 = dx_dt(current_t)
    Omega=domega_dt(current_t)    
    UP= U0[Xi] + Omega*position_r 
    
    # 1. Direct forcing term (a force density)
    direct_force_density = (UP - velocity_at_surface)/dt
    
    # Calculate segment lengths dS for density conversion
    x_i = jnp.roll(xp,-1)
    y_i = jnp.roll(yp,-1)
    dxL = x_i-xp
    dyL = y_i-yp
    dS = jnp.sqrt(dxL**2 + dyL**2) + 1e-9

    # --- MODIFICATION: Isolate Tension Force and Print ---
    total_force_density = jnp.zeros_like(direct_force_density) # Initialize as zero
    if sigma is not None and sigma > 0.0:
        # 2. Calculate the physical tension force (a point force) on the particle
        tension_force_x, tension_force_y = calculate_tension_force(xp, yp, sigma)

        # To see the effect of ONLY the tension force:
        # The force on the fluid is the reaction force (-F_tension). Convert to density by dividing by dS.
        if Xi == 0:
            total_force_density = - (tension_force_x / dS)
        else: # Xi == 1
            total_force_density = - (tension_force_y / dS)
        
        # To see the COMBINED effect (direct forcing + tension), you would use this line instead:
        # if Xi == 0:
        #     total_force_density = direct_force_density - (tension_force_x / dS)
        # else:
        #     total_force_density = direct_force_density - (tension_force_y / dS)

        # --- DEBUG PRINTING ---
        # This will only print for the x-component (Xi==0) to avoid duplicate messages.
        if Xi == 0:
            jax_debug.print("--- Force Analysis (t={t}) ---", t=current_t)
            jax_debug.print("Max Abs Direct Force Density: {x}", x=jnp.max(jnp.abs(direct_force_density)))
            jax_debug.print("Max Abs Tension Force (Point Force): {x}", x=jnp.max(jnp.abs(tension_force_x)))
            jax_debug.print("Max Abs Total Force Density (Tension Only): {x}", x=jnp.max(jnp.abs(total_force_density)))
            jax_debug.print("------------------------------------")
    # --- END MODIFICATION ---

    def calc_force(F,xp,yp,dxi,dyi,dss):
        return F*discrete_fn(jnp.sqrt((xp-X)**2 + (yp-Y)**2),0,dxEUL)*dss

    def foo(tree_arg):
        F,xp,yp,dxi,dyi,dss = tree_arg
        return calc_force(F,xp,yp,dxi,dyi,dss)
    
    def foo_pmap(tree_arg):
        return jnp.sum(jax.vmap(foo,in_axes=1)(tree_arg),axis=0)
        
    divider=jax.device_count()
    n = len(xp)//divider
    mapped = []
    for i in range(divider):
       mapped.append([total_force_density[i*n:(i+1)*n],xp[i*n:(i+1)*n],yp[i*n:(i+1)*n],dxL[i*n:(i+1)*n],dyL[i*n:(i+1)*n],dS[i*n:(i+1)*n]])

    return jnp.sum(jax.pmap(foo_pmap)(jnp.array(mapped)),axis=0)


def IBM_Multiple_NEW(field, Xi, particles,discrete_fn,surface_fn,dt, sigma=0.0):
    Grid_p = particles.generate_grid()
    shape_fn = particles.shape
    Displacement_EQ = particles.Displacement_EQ
    Rotation_EQ = particles.Rotation_EQ
    Nparticles = len(particles.particle_center)
    particle_center = particles.particle_center
    geom_param = particles.geometry_param
    displacement_param = particles.displacement_param
    rotation_param = particles.rotation_param
    force = jnp.zeros_like(field.data)
    for i in range(Nparticles):
        Xc = lambda t:Displacement_EQ([displacement_param[i]],t)
        rotation = lambda t:Rotation_EQ([rotation_param[i]],t)
        dx_dt = jax.jacrev(Xc)
        domega_dt = jax.jacrev(rotation)
        force+= IBM_force_GENERAL(field,Xi,particle_center[i],geom_param[i],Grid_p,shape_fn,discrete_fn,surface_fn,dx_dt,domega_dt,rotation,dt, sigma)
    return grids.GridArray(force,field.offset,field.grid)


def calc_IBM_force_NEW_MULTIPLE(all_variables,discrete_fn,surface_fn,dt, sigma=0.0):
    velocity = all_variables.velocity
    particles = all_variables.particles
    axis = [0,1]
    ibm_forcing = lambda field,Xi:IBM_Multiple_NEW(field, Xi, particles,discrete_fn,surface_fn,dt, sigma)
    
    return tuple(grids.GridVariable(ibm_forcing(field,Xi),field.bc) for field,Xi in zip(velocity,axis))
