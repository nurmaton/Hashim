from jax_ib.base import particle_class as pc
from jax_ib.base import interpolation
from jax_ib.base import IBM_Force
from jax_ib.base import convolution_functions
import jax
import jax.numpy as jnp

# --- HELPER FUNCTION ---
# This correctly interpolates the velocity field to the particle markers using the IBM integral.
# It is the implementation of Equation (3) from the Sustiel & Grier paper.
def interpolate_velocity_to_surface(velocity_field, xp, yp, discrete_fn):
    """
    Interpolates the Eulerian velocity field to the Lagrangian particle markers
    using the IBM discrete delta function integral.
    """
    _surface_fn_component = lambda field, xp_pts, yp_pts: convolution_functions.new_surf_fn(field, xp_pts, yp_pts, discrete_fn)
    u_at_markers = _surface_fn_component(velocity_field[0], xp, yp)
    v_at_markers = _surface_fn_component(velocity_field[1], xp, yp)
    return u_at_markers, v_at_markers

# --- HEAVILY REWRITTEN CORE FUNCTION ---
#
# OLD vs NEW COMPARISON & EXPLANATION:
#
# WHY THE CHANGE WAS MADE:
# The OLD file contained functions like `Update_particle_position_Multiple` which were designed
# for the KINEMATIC model. They calculated the particle's next position based on pre-defined
# mathematical functions (`Displacement_EQ`, etc.). The NEW function implements the DYNAMIC
# equations of motion for the penalty IBM. It calculates the next state of the particle
# based on physical forces.
#
# KEY DIFFERENCES:
# 1. LOGIC:
#    - OLD: Calculated the particle's velocity from a kinematic function (`dx_dt`) and updated
#           the center position. The shape was rigid.
#    - NEW: Implements the full physics for the two sets of markers:
#        a) Advects the fluid markers (X) with the local fluid velocity (Eq. 3).
#        b) Accelerates the mass markers (Y) using Newton's Second Law with penalty
#           and gravity forces (Eq. 5).
#
# 2. INTERPOLATION:
#    - OLD: Did not need to interpolate fluid velocity to the markers because the particle's
#           motion was prescribed.
#    - NEW: Critically relies on interpolating the fluid velocity to the fluid markers (`X`)
#           to determine how they are advected. `interpolate_velocity_to_surface` is the
#           correct, physically-consistent way to do this, matching the IBM formulation.
#
# 3. STATE MANAGEMENT:
#    - OLD: Took in a particle object and returned a new one with an updated `particle_center`.
#    - NEW: Takes the entire simulation state (`all_variables`) and returns a completely new
#           `all_variables` object with the fully updated particle state (all marker positions
#           and velocities). This is a robust, functional approach suitable for JAX.
#
# WHY THE NEW METHOD IS BETTER:
#    - It simulates the actual physics of a deformable object, rather than just imposing a motion.
#    - It correctly couples the fluid's motion to the particle's motion and vice-versa, allowing
#      for complex behaviors like deformation and sedimentation to emerge naturally.
#
def update_massive_deformable_particle(all_variables, dt, gravity_g=0.0):
    """
    Updates the state of a massive, deformable particle for one time step.
    """
    particles_container = all_variables.particles
    velocity_field = all_variables.velocity
    
    # The delta function is the kernel for interpolation.
    discrete_fn = lambda dist, center, width: convolution_functions.delta_approx_logistjax(dist, center, width)

    # This function operates on the first particle in the list.
    particle = particles_container.particles[0] 
    
    # --- 1. Advect Fluid Markers (X) using the IBM Integral (Eq. 3) ---
    U_fluid_x_pts, U_fluid_y_pts = interpolate_velocity_to_surface(
        velocity_field, particle.xp, particle.yp, discrete_fn
    )
    
    # Update marker positions with a forward Euler step
    new_xp = particle.xp + dt * U_fluid_x_pts
    new_yp = particle.yp + dt * U_fluid_y_pts

    # --- 2. Update Mass Markers (Y) using Newton's Second Law (Eq. 5) ---
    # F_i^m = Kp(Y-X)
    penalty_force_x, penalty_force_y = IBM_Force.calculate_penalty_force(
        particle.xp, particle.yp, particle.Ym_x, particle.Ym_y, particle.stiffness
    )

    # Net force on mass markers: F_net_Y = -F_i^m - Mg
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
    updated_particle = pc.particle(
        xp=new_xp, yp=new_yp, Ym_x=new_Ym_x, Ym_y=new_Ym_y,
        Vm_x=new_Vm_x, Vm_y=new_Vm_y,
        mass_per_marker=particle.mass_per_marker, stiffness=particle.stiffness, sigma=particle.sigma,
        particle_center=new_center, geometry_param=particle.geometry_param,
        Grid=particle.Grid, shape=particle.shape
    )
    
    # --- 4. Rebuild the container holding the list of particles ---
    new_particles_container = pc.particle_lista(particles=[updated_particle])
    
    # --- 5. Return a new All_Variables instance with the updated fields ---
    return pc.All_Variables(
        particles=new_particles_container,
        velocity=all_variables.velocity,
        pressure=all_variables.pressure,
        Drag=all_variables.Drag,
        Step_count=all_variables.Step_count + 1,
        MD_var=all_variables.MD_var
    )
