import jax.numpy as jnp
import jax
from jax_ib.base import grids

# --- NEW FUNCTION ---
# This function implements the surface tension force as described in
# "Complex Dynamics of an Acoustically Levitated Fluid Droplet..." (Sustiel & Grier), Eq. (7).
def calculate_tension_force(xp, yp, sigma):
    """
    Calculates the surface tension force on each Lagrangian marker.

    Args:
        xp (jax.numpy.ndarray): Array of x-coordinates of the markers.
        yp (jax.numpy.ndarray): Array of y-coordinates of the markers.
        sigma (float): The surface tension coefficient.

    Returns:
        tuple[jax.numpy.ndarray, jax.numpy.ndarray]: A tuple containing the
        x and y components of the tension force at each marker.
    """
    # Calculate the tangent vectors l_i = X_{i+1} - X_i
    # jnp.roll with shift=-1 gets the (i+1)th element for each i
    dxL = jnp.roll(xp, -1) - xp
    dyL = jnp.roll(yp, -1) - yp

    # Calculate the magnitude of the tangent vectors, dS = ||l_i||
    # A small epsilon is added for numerical stability to avoid division by zero.
    dS = jnp.sqrt(dxL**2 + dyL**2) + 1e-9

    # Calculate the normalized (unit) tangent vectors, l_hat_i
    l_hat_x = dxL / dS
    l_hat_y = dyL / dS

    # Get the normalized tangent vector from the *previous* segment, l_hat_{i-1}
    # jnp.roll with shift=1 gets the (i-1)th element for each i
    l_hat_x_prev = jnp.roll(l_hat_x, 1)
    l_hat_y_prev = jnp.roll(l_hat_y, 1)

    # Implement Eq. (7): F_T = sigma * (l_hat_i - l_hat_{i-1})
    force_x = sigma * (l_hat_x - l_hat_x_prev)
    force_y = sigma * (l_hat_y - l_hat_y_prev)

    return force_x, force_y
# --- END NEW FUNCTION ---


def integrate_trapz(integrand,dx,dy):
    return jnp.trapz(jnp.trapz(integrand,dx=dx),dx=dy)


def Integrate_Field_Fluid_Domain(field):
    grid = field.grid
   # offset = field.offset
    dxEUL = grid.step[0]
    dyEUL = grid.step[1]
   # X,Y =grid.mesh(offset)
    
    return integrate_trapz(field.data,dxEUL,dyEUL)

# --- MODIFIED FUNCTION ---
def IBM_force_GENERAL(field,Xi,particle_center,geom_param,Grid_p,shape_fn,discrete_fn,surface_fn,dx_dt,domega_dt,rotation,dt, sigma=0.0):
    
    grid = field.grid
    offset = field.offset
    X,Y = grid.mesh(offset)
    dxEUL = grid.step[0]
    dyEUL = grid.step[1]
    current_t = field.bc.time_stamp
    #current_t = 0.0
    xp0,yp0 = shape_fn(geom_param,Grid_p)
    #print('yp',yp0,'xp',xp0)
    #print('angle',current_t,rotation(current_t),particle_center)
    #print(yp0)
    xp = (xp0)*jnp.cos(rotation(current_t))-(yp0)*jnp.sin(rotation(current_t))+particle_center[0]
    yp = (xp0)*jnp.sin(rotation(current_t))+(yp0 )*jnp.cos(rotation(current_t))+particle_center[1]
    surface_coord =[(xp)/dxEUL-offset[0],(yp)/dyEUL-offset[1]]
    #print(rotation(current_t))
    velocity_at_surface = surface_fn(field,xp,yp)
    
    if Xi==0:
        position_r = -(yp-particle_center[1])
    elif Xi==1:
        position_r = (xp-particle_center[0])
    
    U0 = dx_dt(current_t)
    #print('U0',U0)
    Omega=domega_dt(current_t)    
    UP= U0[Xi] + Omega*position_r 
    #print(xp)
    #print('XI',Xi,UP,len(UP))
    
    # Direct forcing term based on velocity mismatch
    force = (UP -velocity_at_surface)/dt
    
    # --- NEW: Calculate and consider Surface Tension Force ---
    # This section calculates the tension force based on the new paper.
    # It is a physical force, distinct from the direct forcing term calculated above.
    tension_force_x, tension_force_y = jnp.zeros_like(xp), jnp.zeros_like(yp)
    if sigma is not None and sigma > 0.0:
        if Xi == 0:
            tension_force_x, _ = calculate_tension_force(xp, yp, sigma)
        else: # Xi == 1
            _, tension_force_y = calculate_tension_force(xp, yp, sigma)

    # NOTE on integration: The `force` variable here is a force *density* (F/ds) for the direct forcing method.
    # The `tension_force` is a point force (F). To combine them, the tension force would need to be
    # converted to a density and added to the total force on the fluid.
    # For example, before spreading, one might define a total force density:
    # total_force_density = force - (tension_force_x / dS)  # Note the negative sign for force on fluid
    # For now, the tension force is calculated but not combined, awaiting your specific integration strategy.
    # --- END NEW SECTION ---

   # if Xi==0:
        #plt.plot(xp,force)
        #maxforce =  delta_approx_logistjax(xp[0],X,0.003,1)
   #     maxforce = discrete_fn(xp[3],X)
   #     plt.imshow(maxforce)
   #     print('Maxforce',jnp.max(maxforce))
    #    print(xp)
    x_i = jnp.roll(xp,-1)
    y_i = jnp.roll(yp,-1)
    dxL = x_i-xp
    dyL = y_i-yp
    dS = jnp.sqrt(dxL**2 + dyL**2)
    
    
    def calc_force(F,xp,yp,dxi,dyi,dss):
        # Here we are spreading the direct forcing term 'F'.
        # If combining with tension, the new total force density would be passed here.
        return F*discrete_fn(jnp.sqrt((xp-X)**2 + (yp-Y)**2),0,dxEUL)*dss
        #return F*discrete_fn(xp-X,0,dxEUL)*discrete_fn(yp-Y,0,dyEUL)*dss
        #return F*discrete_fn(xp,X,dxEUL)*discrete_fn(yp,Y,dyEUL)*dss**2
    def foo(tree_arg):
        F,xp,yp,dxi,dyi,dss = tree_arg
        return calc_force(F,xp,yp,dxi,dyi,dss)
    
    def foo_pmap(tree_arg):
        #print(tree_arg)
        return jnp.sum(jax.vmap(foo,in_axes=1)(tree_arg),axis=0)
    divider=jax.device_count()
    n = len(xp)//divider
    mapped = []
    for i in range(divider):
       # print(i)
        mapped.append([force[i*n:(i+1)*n],xp[i*n:(i+1)*n],yp[i*n:(i+1)*n],dxL[i*n:(i+1)*n],dyL[i*n:(i+1)*n],dS[i*n:(i+1)*n]])
    #mapped = jnp.array([force,xp,yp])
    #remapped = mapped.reshape(())#jnp.array([[force[:n],xp[:n],yp[:n]],[force[n:],xp[n:],yp[n:]]])
    
    #return cfd.grids.GridArray(jnp.sum(jax.pmap(foo_pmap)(jnp.array(mapped)),axis=0),offset,grid)
    return jnp.sum(jax.pmap(foo_pmap)(jnp.array(mapped)),axis=0)
# --- END MODIFIED FUNCTION ---

# --- MODIFIED FUNCTION ---
# Added sigma to the call to IBM_force_GENERAL
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
# --- END MODIFIED FUNCTION ---

# --- MODIFIED FUNCTION ---
# Added sigma to be passed down. This would typically be part of your simulation parameters.
def calc_IBM_force_NEW_MULTIPLE(all_variables,discrete_fn,surface_fn,dt, sigma=0.0):
    velocity = all_variables.velocity
    particles = all_variables.particles
    axis = [0,1]
    ibm_forcing = lambda field,Xi:IBM_Multiple_NEW(field, Xi, particles,discrete_fn,surface_fn,dt, sigma)
    
    return tuple(grids.GridVariable(ibm_forcing(field,Xi),field.bc) for field,Xi in zip(velocity,axis))
# --- END MODIFIED FUNCTION ---
