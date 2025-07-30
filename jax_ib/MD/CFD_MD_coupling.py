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
Functions for coupling CFD (fluid) and MD (particle) simulations.

This module provides the necessary tools to transfer information from the
Eulerian fluid grid to the Lagrangian MD particles. The primary mechanism for
this is interpolation, which allows us to find the fluid velocity at the precise
location of each particle. This fluid velocity can then be used to apply a
drag force or other hydrodynamic interaction to the particles in the MD simulation.
"""

import jax.numpy as jnp
import jax


def surface_fn_jax(field: jnp.ndarray, surface_coord: list) -> jnp.ndarray:
    """
    Interpolates a field at a set of fractional coordinates using JAX's `map_coordinates`.

    This is a low-level wrapper around JAX's N-dimensional spline interpolation
    function. It takes a field (as a raw array) and a list of coordinate arrays
    and returns the interpolated values.

    Args:
      field: The N-dimensional JAX array of data to interpolate from.
      surface_coord: A list of coordinate arrays. For a 2D field, this would be
        `[x_coords, y_coords]`, where `x_coords` and `y_coords` are arrays of
        the target locations in fractional grid index units.

    Returns:
      A JAX array of the interpolated values at the specified coordinates.
    """
    # `jax.scipy.ndimage.map_coordinates` is a powerful, JAX-jittable function
    # that performs multi-dimensional spline interpolation. `order=1` specifies
    # that it should use linear interpolation.
    return jax.scipy.ndimage.map_coordinates(field, surface_coord, order=1)


def interpolate_pbc(field: grids.GridVariable, list_p: jnp.ndarray) -> jnp.ndarray:
    """
    Interpolates a `GridVariable` field to a list of physical particle positions.

    This function serves as the main bridge from the Eulerian grid to the
    Lagrangian particles. It takes a `GridVariable` (which contains the fluid data
    and all the necessary grid metadata) and a list of physical particle positions,
    and returns the value of the field at each particle's location.

    It assumes periodic boundary conditions, which are handled by the underlying
    `map_coordinates` function (implicitly, via its default `mode`).

    Args:
      field: The `GridVariable` representing the fluid field (e.g., a velocity component).
      list_p: A JAX array of shape `(N, D)` where `N` is the number of particles
        and `D` is the number of dimensions, containing the physical coordinates
        of each particle.

    Returns:
      A 1D JAX array of shape `(N,)` containing the interpolated field value
      at each particle's location.
    """
    # Transpose the particle position array from shape (N, D) to (D, N) for easier indexing.
    list_p = jnp.moveaxis(list_p, 0, -1)

    # Extract grid metadata from the field object.
    grid = field.grid
    offset = field.offset
    dxEUL = grid.step[0]
    dyEUL = grid.step[1]

    # Unpack the physical x and y coordinates of all particles.
    xp = list_p[0]
    yp = list_p[1]
  
    # --- Coordinate Transformation ---
    # Convert the physical coordinates (`xp`, `yp`) into the fractional grid
    # index coordinates required by `surface_fn_jax`. This is a linear mapping.
    surface_coord = (((xp) / dxEUL - offset[0]), ((yp) / dyEUL - offset[1]))
    
    # Call the low-level interpolation function with the field's raw data.
    return surface_fn_jax(field.data, surface_coord)


def custom_force_fn_pbc(all_variables) -> jnp.ndarray:
    """
    Calculates the hydrodynamic force on MD particles from the fluid velocity.

    This function represents the coupling from the CFD solver to the MD solver.
    It interpolates the fluid velocity to each particle's position and returns
    this as a "force" (or more accurately, a velocity that can be used to
    calculate a Stokes drag force, F = γ * (u_fluid - v_particle)).

    The name is slightly misleading; it doesn't calculate the final force but
    provides the fluid velocity needed to do so in the MD code.

    Args:
      all_variables: The complete state object of the simulation, containing
        both the CFD (`.velocity`) and MD (`.MD_var.position`) states.

    Returns:
      A JAX array of shape `(N, D)` containing the interpolated fluid velocity
      vector at the location of each of the N particles.
    """
    # Extract the CFD velocity field and MD particle positions from the state.
    trajectory_cfd = all_variables.velocity
    R = all_variables.MD_var.position
    
    # Interpolate the u-component of the fluid velocity to all particle positions.
    u = interpolate_pbc(trajectory_cfd[0], R)
    # Interpolate the v-component of the fluid velocity to all particle positions.
    v = interpolate_pbc(trajectory_cfd[1], R)
    
    # Combine the interpolated u and v components into a single array of velocity vectors.
    # `jnp.array([u,v])` creates a shape (2, N) array.
    # `jnp.moveaxis` transposes it to the standard (N, 2) shape for particle data.
    # The multiplication by 1.0 is likely just to ensure the result is a float.
    return jnp.moveaxis(jnp.array([u,v]), 0, -1) * 1.0
