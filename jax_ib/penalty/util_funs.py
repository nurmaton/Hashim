# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module provides the core utility functions for implementing the Brinkman
penalty-based Immersed Boundary Method (IBM). Its central purpose is to create
a differentiable, grid-based representation of immersed solid objects, which
is then used to apply forces that simulate the solid's presence in the fluid.

The module's functionality can be broken down into several layers:

1.  **Geometric Projection (`project_particle`)**: This is the lowest-level
    geometric function. It "rasterizes" an abstract, Lagrangian shape
    (defined by radii at various angles) onto the Eulerian fluid grid. It
    calculates crucial geometric quantities like the interpolated boundary
    position and normal vectors at every grid point. It also applies a
    regularized delta function to ensure these quantities are concentrated
    in a thin, smooth band around the boundary.

2.  **Solid Mask Generation (`calc_perm`)**: This function builds upon the
    geometric projection to create a smooth, grid-based scalar field that
    acts as a "solid mask." This field, termed "inverse permeability," has a
    high value inside the object and zero outside. It effectively defines the
    region where a penalty force will be applied to resist fluid flow.

3.  **Multi-Particle Orchestration (`perm_vmap_multiple_particles`)**: This is
    the main, high-level entry point. It efficiently handles simulations with
    many objects by using `jax.vmap` to vectorize the `calc_perm` operation
    over all particles. It produces a single, combined field representing the
    union of all solid regions in the simulation domain.

4.  **Forcing Function (`arbitrary_obstacle`)**: This provides a simple, direct
    way to use the generated solid mask as a Brinkman-style penalty force
    within the main fluid solver.

Together, these functions form the machinery for defining the solid-fluid
interface and enforcing boundary conditions in a smooth, differentiable manner,
making them essential for the project's optimization capabilities.
"""
import jax_ib.base as ib
import jax
import jax.numpy as jnp
import numpy as np


def arbitrary_obstacle(pressure_gradient, permeability):
    """Creates a forcing function for a stationary, porous obstacle.

    This function implements a Brinkman-style penalty method. It returns a
    `forcing` function that can be passed to the main CFD solver. This
    forcing term has two components: a background pressure gradient that drives
    the flow, and a penalty force that opposes the fluid velocity, simulating
    the presence of an obstacle.

    Args:
        pressure_gradient: A tuple `(px, py)` representing the pressure
                           gradient in the x and y directions.
        permeability: A scalar value representing the drag or penalty
                      coefficient. A large value creates a strong force that
                      drives the local fluid velocity to zero, simulating a
                      solid object. A small value simulates a porous medium.

    Returns:
        A forcing function `forcing(v)` that takes a velocity field `v` and
        returns the corresponding force field as a tuple of `GridArray`s.
    """

    def forcing(v):
        """The actual forcing calculation."""
        # The permeability is treated as the penalty coefficient for both directions.
        force_vector = (permeability, permeability)
        px = pressure_gradient
        # For each velocity component (u, v):
        # Force = (Pressure Gradient) - (Penalty Coefficient * Local Velocity)
        # This force is then wrapped in the GridArray data structure.
        return tuple(ib.grids.GridArray(pxn * jnp.ones_like(u.data) - f * u.data, u.offset, u.grid)
                     for pxn, f, u in zip(px, force_vector, v))

    return forcing


def delta_approx_tanh(rf2, r2):
    """A regularized (smooth) delta function based on the derivative of tanh.

    In the IB method, we need to apply forces at the boundary, which is
    infinitely thin. A true Dirac delta function is numerically unstable.
    This function provides a smooth, bell-shaped approximation of a delta
    function that is concentrated in a narrow band. It will have a large value
    when `r2` (the squared distance of a grid point) is close to `rf2` (the
    squared distance of the object boundary) and will decay rapidly away from it.

    Args:
        rf2: The squared radial distance of the object's boundary.
        r2: The squared radial distance of a grid point.

    Returns:
        A scalar value for the smoothed delta function, normalized to have a max value of 1.
    """
    inv_perm = 200000  # A large scaling factor for the penalty force.
    width = 0.002      # The characteristic thickness of the boundary region.
    r = jnp.sqrt(r2)
    # This functional form is proportional to d/dx(tanh(x)) = 1/cosh^2(x),
    # creating a smooth, localized peak.
    approx = inv_perm / width * r / jnp.cosh((rf2 - r2) / width)**2

    # Normalize to prevent the maximum value from being excessively large.
    return approx / np.max(approx)


def project_particle(grid, circle_center, Rtheta, delta_approx_fn):
    """Projects a Lagrangian particle shape onto the Eulerian fluid grid.

    This is a core IBM routine. It takes a shape defined in polar coordinates
    by a set of radii at different angles (`Rtheta`) and generates two crucial
    grid-based fields:
    1. A field of outward-pointing normal vectors, defined only at the boundary.
    2. A scalar field representing the interpolated radius of the object at every
       point on the grid.

    Args:
        grid: The Eulerian `Grid` object from the solver.
        circle_center: The `(x, y)` coordinates of the object's center.
        Rtheta: A 1D JAX array of radii defining the object's shape. This comes
                from one of the `parametric_fns`.
        delta_approx_fn: A function (like `delta_approx_tanh`) to compute the
                         strength of the boundary force.

    Returns:
        A tuple `(normal_v, Rfinal)` where:
        - `normal_v` is a tuple of `GridArray`s for the x and y components of
          the normal vector field, multiplied by the delta function.
        - `Rfinal` is a `GridArray` containing the interpolated object radius
          at each grid cell.
    """
    xc, yc = circle_center
    # Get the Cartesian coordinates (X, Y) for every cell center on the fluid grid.
    X, Y = grid.mesh(grid.cell_center)

    # Reconstruct the angles corresponding to the input radii `Rtheta`.
    ntheta = Rtheta.size
    theta = jnp.linspace(0, 2 * jnp.pi, ntheta)
    
    #--- Calculate the polar angle (theta_grid) for every Cartesian grid point (X, Y) ---
    # This is an implementation of atan2(Y-yc, X-xc) using arctan and heaviside functions
    # to correctly handle all four quadrants.
    theta_grid = jnp.arctan((Y - yc) / (X - xc)) * jnp.heaviside(X - xc, 1)  # Quadrant 1 & 4
    theta_grid += (jnp.arctan((Y - yc) / (X - xc)) + jnp.pi) * jnp.heaviside(xc - X, 1)  # Quadrant 2 & 3
    # The original implementation had some redundancy in quadrant handling.
    # The above two lines are generally sufficient for a valid atan2.
    
    dtheta = 2 * jnp.pi / (ntheta - 1)

    #--- Interpolate the Lagrangian shape onto the Eulerian grid ---
    # Find the index of the Lagrangian point just before each Eulerian grid point's angle.
    flattened_indx = (theta_grid // dtheta).astype(int)

    # Get the bracketing Lagrangian points (R0, theta_0) and (R1, theta_1).
    R0 = Rtheta[flattened_indx]
    theta_0 = theta[flattened_indx]
    
    # Use jnp.roll to get the "next" point for robust interpolation at the 2pi boundary.
    Rtheta_i = jnp.roll(Rtheta, -1)
    theta_i = jnp.roll(theta, -1)
    R1 = Rtheta_i[flattened_indx]

    # Linearly interpolate to find the object's radius `Rfinal` at the exact
    # angle of each grid point. This creates a smooth boundary representation.
    drdtheta = (R1 - R0) / (theta_i[flattened_indx] - theta_0)
    Rfinal = (R0 + drdtheta * (theta_grid - theta_0)).reshape(X.shape)
    
    #--- Calculate the normal vectors and apply the delta function ---
    # Calculate the squared distance of each grid point from the object's center.
    r2 = (Y - circle_center[1])**2 + (X - circle_center[0])**2

    # Call the provided delta function. This will create a force field that is
    # non-zero only in a thin, fuzzy band around the interpolated boundary.
    # The commented out lines show alternative delta function definitions.
    #delta_approx = jnp.exp(-prefac*distance_sq)
    #delta_approx = delta_approx_fn(distance_sq)
    delta_approx = delta_approx_tanh(Rfinal**2, r2)
    
    # Calculate the components of the outward normal vector to the boundary
    # using the formula for a normal to a polar curve. `drdtheta` is the derivative.
    nx = (drdtheta * jnp.sin(theta_grid) + Rfinal * jnp.cos(theta_grid))
    ny = (-drdtheta * jnp.cos(theta_grid) + Rfinal * jnp.sin(theta_grid))
    length = jnp.sqrt(nx**2 + ny**2) + 1e-9 # Add epsilon for stability
    
    # Package the final normal vector field (multiplied by the delta function)
    # into the GridArray structure.
    normal_v = (ib.grids.GridArray(nx / length * delta_approx, grid.cell_center, grid),
                ib.grids.GridArray(ny / length * delta_approx, grid.cell_center, grid))
                
    return normal_v, Rfinal

def calc_perm(grid, circle_center, Rtheta, smoothening_fn, Know):
    """Calculates a smooth "permeability" field for a single object.

    In the context of the Brinkman penalty method, this function doesn't compute
    physical permeability. Instead, it generates a scalar field representing the
    inverse permeability (or drag strength) that defines the object's solid
    region. The field has a high value inside the object (high drag, simulating
    a solid) and a value of zero outside (no drag), with a smooth transition
    at the boundary to ensure differentiability.

    Args:
        grid: The Eulerian `Grid` object from the solver.
        circle_center: The (x, y) coordinates of the object's center.
        Rtheta: A 1D JAX array of radii defining the object's shape.
        smoothening_fn: A function that creates the smooth transition from
                        solid to fluid (e.g., a smoothed Heaviside function).
        Know: A parameter (e.g., stiffness or width) for the `smoothening_fn`.

    Returns:
        A `GridArray` containing the scalar penalty field.
    """
    X, Y = grid.mesh(grid.cell_center)
    # This lambda function is not used in the final version but shows an
    # example of how a delta function could be partially applied.
    delta_approx = lambda r: delta_approx_fn(r, grid)

    # Project the particle's Lagrangian description onto the Eulerian grid.
    # We only need Rfinal from this; the normal vectors are discarded.
    normal_v, Rfinal = project_particle(grid, circle_center, Rtheta, delta_approx)
    del normal_v  # Free up memory.

    # Calculate a signed distance function `G`.
    # `r_squared` is the squared distance of each grid point from the center.
    r_squared = (Y - circle_center[1])**2 + (X - circle_center[0])**2
    # `Rfinal` is the interpolated radius of the object at each grid point.
    # Therefore, `G` will be > 0 for points inside the object and < 0 for points outside.
    G = (Rfinal**2 - r_squared)

    # The commented-out lines below show alternative ways to create the field:
    # A sharp "where" condition (not differentiable).
    # return jnp.where(G>0,inv_perm*jnp.ones_like(G),jnp.zeros_like(G))
    # A sharp Heaviside step function (not easily differentiable).
    # return inv_perm*jnp.heaviside(G,1.0)
    # A classic smooth tanh transition.
    # return inv_perm/2.0*(1.0 + jnp.tanh(G))

    # Apply the user-provided smoothing function to the signed distance `G`.
    # This is the key to creating a smooth, differentiable solid representation.
    return smoothening_fn(G, Know)


def perm_vmap_multiple_particles(grid, particles, smoothening_fn, Know):
    """Generates a combined permeability field for multiple particles using JAX's vmap.

    This is the high-level entry point for defining all solid obstacles in the
    simulation. It takes a `particles` data structure (a JAX pytree) containing
    the state of all immersed objects, computes the permeability field for each
    one, and sums them to create a single field representing the union of all
    solids.

    The use of `jax.vmap` is critical for performance. It vectorizes the
    computation over all particles of a given type, allowing for efficient
    execution on accelerators like GPUs without writing explicit loops in Python.

    Args:
        grid: The Eulerian `Grid` object from the solver.
        particles: A custom pytree (or a tuple of pytrees) containing the data
                   for all particles (e.g., centers, shape parameters).
        smoothening_fn: The smoothing function passed down to `calc_perm`.
        Know: The parameter for the smoothing function.

    Returns:
        A single `GridArray` representing the combined solid regions of all
        particles.
    """

    def Vmap_calc_perm(grid, Grid_p, tree_arg):
        """A wrapper function to set up the vmap over particle data."""
        # A function that generates the radial description of a particle.
        calc_r = tree_arg.shape

        def foo(tree_arg):
            """The function to be vmapped. Processes a SINGLE particle."""
            # Unpack the data for one particle.
            (particle_center, geometry_param, _, _) = tree_arg
            # Generate its shape description (radii vs. angle).
            R_theta = calc_r(geometry_param, Grid_p)
            # Calculate the permeability field for this single particle.
            return calc_perm(grid, particle_center, R_theta, smoothening_fn, Know)

        # Flatten the particle data pytree to pass it to vmap.
        xs_flat, xs_tree = jax.tree_flatten(tree_arg)

        # `jax.vmap` creates a new function that applies `foo` to each element
        # in the flattened list of particle data.
        return jax.vmap(foo)(xs_flat)

    X, Y = grid.mesh()
    # Initialize a permeability field of all zeros.
    perm = jnp.zeros_like(X)

    # Handle both a single particle type and a tuple of different particle types.
    if isinstance(particles, tuple):
        # If there are multiple types of particles (e.g., ellipses and circles).
        for particle in particles:
            # Generate the angular grid for this particle type.
            Grid_p = particle.generate_grid()
            # Call the vmapped function to get a batch of permeability fields,
            # then sum them along the batch axis to get one field for this type.
            perm += jnp.sum(Vmap_calc_perm(grid, Grid_p, particle), axis=0)
        return perm
    else:
        # If there is only one type of particle.
        Grid_p = particles.generate_grid()
        perm += jnp.sum(Vmap_calc_perm(grid, Grid_p, particles), axis=0)
        return perm
