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
Provides discrete delta functions and convolution operations for the IBM.

This module contains the core mathematical machinery for the Immersed Boundary
Method (IBM). Its primary purpose is to facilitate communication between the
fixed Eulerian grid (where fluid properties are defined) and the moving
Lagrangian markers (which define the particle boundary).

This communication is achieved through two related operations, both implemented
as discrete convolutions (numerical integrals) using a discrete delta function:

1.  **Interpolation**: Calculating the value of a fluid field (e.g., velocity)
    at the location of the Lagrangian markers.
2.  **Spreading**: Distributing a force from the Lagrangian markers onto the
    surrounding fluid grid points.

The functions here are designed to be highly efficient and scalable using JAX's
vectorization (`vmap`) and parallelization (`pmap`) capabilities.
"""

from typing import Any, Callable, Tuple
import jax
import jax.numpy as jnp
from jax_ib.base import grids

def delta_approx_logistjax(x: jnp.ndarray, x0: float, w: float) -> jnp.ndarray:
    """
    A smoothed, differentiable approximation of the 1D Dirac delta function.

    NOTE: Despite its name, this function implements a Gaussian (normal distribution)
    function, not one based on a logistic function. The Gaussian is a common
    choice for a smooth delta function in numerical methods.

    The true Dirac delta function is infinite at a single point and zero
    elsewhere, which is unsuitable for numerical computation. This function
    provides a smooth "bump" that approximates this behavior while being
    differentiable, which is essential for JAX.

    Args:
      x: A JAX array of positions on the Eulerian grid.
      x0: The center of the delta function (the Lagrangian marker's position).
      w: A parameter controlling the "spread" or standard deviation of the
         Gaussian. This is typically related to the grid spacing `dx` to ensure
         the function has the correct properties (e.g., support over a few cells).

    Returns:
      A JAX array of values representing the strength of the delta function at
      the given positions.
    """
    # This is the standard formula for a Gaussian function, normalized to integrate to 1.
    return 1.0 / (w * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * ((x - x0) / w)**2)

def new_surf_fn(
    field: grids.GridVariable,
    xp: jnp.ndarray,
    yp: jnp.ndarray,
    discrete_fn: Callable[..., jnp.ndarray]
) -> jnp.ndarray:
    """
    Performs the core IBM convolution integral using a separable 2D delta function.

    This function numerically evaluates the integral `∫ f(x) δ(x - X) dx`, where
    `f` is a fluid field, `X` is a point on the Lagrangian boundary, and `δ` is
    the discrete delta function. The 2D delta function is formed by multiplying
    two 1D delta functions: `δ(x-X, y-Y) = δ_1D(x-X) * δ_1D(y-Y)`.

    The implementation is heavily optimized for performance on parallel hardware
    (like GPUs/TPUs) using `jax.vmap` for vectorization and `jax.pmap` for
    multi-device execution.

    Args:
      field: The `GridVariable` representing the fluid field to be convolved.
      xp: A 1D JAX array of the Lagrangian markers' x-coordinates.
      yp: A 1D JAX array of the Lagrangian markers' y-coordinates.
      discrete_fn: The 1D discrete delta function to use as the kernel
        (e.g., `delta_approx_logistjax`).

    Returns:
      A 1D JAX array where each element is the result of the convolution for the
      corresponding Lagrangian marker.
    """
    # Get the Eulerian grid coordinates and grid spacing.
    grid = field.grid
    offset = field.offset
    X, Y = grid.mesh(offset)
    dx = grid.step[0]
    dy = grid.step[1]

    # This inner function computes the convolution for a SINGLE Lagrangian point.
    def calc_convolution(xp_pt: float, yp_pt: float) -> float:
        """Calculates the weighted sum for a single particle point."""
        # Evaluate the 1D delta function for the x and y dimensions separately.
        delta_x = discrete_fn(xp_pt, X, dx)
        delta_y = discrete_fn(yp_pt, Y, dy)
        # Combine them to form the 2D separable delta function weights.
        delta_2d = delta_x * delta_y
        # The integral is approximated as a weighted sum over the entire grid.
        # The `*dx*dy` term represents the area element `dA` for the 2D integral.
        return jnp.sum(field.data * delta_2d * dx * dy)

    # This is a wrapper to make the function compatible with vmap's signature.
    def foo(tree_arg: Tuple[float, float]) -> float:
        """Unpacks arguments and calls the convolution function."""
        xp_pt, yp_pt = tree_arg
        return calc_convolution(xp_pt, yp_pt)

    # This function defines the computation to be done on a single device.
    # `jax.vmap` creates a new function that efficiently runs `foo` for a
    # whole batch of particle points at once (vectorization).
    def foo_pmap(tree_arg: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """Vectorizes the convolution over a batch of points."""
        # `in_axes=1` is not strictly necessary here but good practice if the
        # input arrays were shaped differently.
        return jax.vmap(foo)(tree_arg)

    # --- Manual Data Parallelism Setup ---
    # To use multiple devices (e.g., multiple GPUs), we manually split the
    # particle data into chunks, one for each available device.
    divider = jax.device_count()
    n = len(xp) // divider
    mapped = []
    for i in range(divider):
       # Create a list of chunks, where each chunk is `[xp_chunk, yp_chunk]`.
       mapped.append([xp[i*n:(i+1)*n], yp[i*n:(i+1)*n]])

    # `jax.pmap` takes the single-device function `foo_pmap` and runs it in
    # parallel on all available devices, each with its own chunk of data.
    # The results from all devices are then automatically gathered.
    U_deltas = jax.pmap(foo_pmap)(jnp.array(mapped))

    # The result `U_deltas` is a 2D array of shape (num_devices, num_points_per_device).
    # We flatten it back into a single 1D array containing the results for all points.
    return U_deltas.flatten()
