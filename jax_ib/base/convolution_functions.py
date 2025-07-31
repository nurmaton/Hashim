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
Method (IBM). Its primary purpose is to facilitate the two-way communication
between the fixed Eulerian grid (where fluid properties are defined) and the
moving Lagrangian markers (which define the particle boundary).

This communication is achieved through two related operations, both implemented
as discrete convolutions (i.e., numerical integrals) using a smooth,
differentiable approximation of the Dirac delta function:

1.  **Interpolation**: Calculating the value of a fluid field (e.g., velocity)
    at the location of the Lagrangian markers. This is a "gathering" operation,
    mathematically represented as:
    `U(X_k) = ∫ u(x) δ(x - X_k) dx`

2.  **Spreading**: Distributing a force from the Lagrangian markers onto the
    surrounding fluid grid points. This is a "spreading" operation,
    mathematically represented as:
    `f(x) = ∑ F_k δ(x - X_k)`
    (Note: The spreading operation is typically implemented in the `IBM_Force`
    module, but it uses the same delta function defined here).

The functions in this module are designed to be highly efficient and scalable,
making heavy use of JAX's features for vectorization (`vmap`) and multi-device
parallelization (`pmap`) to handle large numbers of Lagrangian markers.
"""

import jax
import jax.numpy as jnp


def delta_approx_logistjax(x,x0,w):
    """
    A smoothed, differentiable approximation of the 1D Dirac delta function.

    NOTE: Despite its name, this function implements a Gaussian (normal distribution)
    function, not one based on a logistic function. The Gaussian is a common
    choice for a smooth delta function in numerical methods because it has a
    compact "bump" shape and is infinitely differentiable.

    The true Dirac delta function is not suitable for numerical computation. This
    function provides a smooth approximation that is well-behaved and JAX-jittable.

    Args:
      x: A JAX array of positions on the Eulerian grid.
      x0: The center of the delta function (e.g., a Lagrangian marker's position).
      w: A parameter controlling the "width" or standard deviation of the
         Gaussian. This is typically related to the grid spacing `dx` to ensure
         the function has the correct properties (e.g., support over a few cells).

    Returns:
      A JAX array of values representing the strength of the delta function at
      the given positions.
    """
    # This is the standard formula for a Gaussian (or normal distribution) PDF.
    # The prefactor `1/(w*sqrt(2*pi))` ensures that the function integrates to 1.
    return 1/(w*jnp.sqrt(2*jnp.pi))*jnp.exp(-0.5*((x-x0)/w)**2)



def new_surf_fn(field,xp,yp,discrete_fn):
    """
    Performs the core IBM convolution integral using a separable 2D delta function.

    This function numerically evaluates the integral `∫ f(x,y) δ(x-xp, y-yp) dx dy`,
    where `f` is a fluid field and `(xp, yp)` is a point on the Lagrangian boundary.
    The 2D delta function is formed by multiplying two 1D delta functions:
    `δ_2D(x-xp, y-yp) = δ_1D(x-xp) * δ_1D(y-yp)`.

    The implementation is heavily optimized for performance on parallel hardware
    (like GPUs/TPUs) by vectorizing the calculation over all Lagrangian points
    and distributing these vectorized chunks across all available devices.

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
    # Get the Eulerian grid coordinates and grid spacing from the field object.
    grid = field.grid
    offset = field.offset
    X,Y = grid.mesh(offset)
    dx = grid.step[0]
    dy = grid.step[1]
    
    def calc_force(xp_pt, yp_pt):
        """
        This inner function computes the convolution for a SINGLE Lagrangian point
        against the entire Eulerian grid.
        """
        # Evaluate the 1D delta function for the x and y dimensions separately.
        delta_x = discrete_fn(xp_pt, X, dx)
        delta_y = discrete_fn(yp_pt, Y, dy)
        
        # The integral is approximated as a weighted sum over the entire grid.
        # `field.data * delta_x * delta_y` is the value of the integrand at each grid point.
        # `* dx * dy` represents the area element `dA` for the 2D integral.
        return jnp.sum(field.data * delta_x * delta_y * dx * dy)
        
    def foo(tree_arg):
        """A simple wrapper to unpack arguments for use with `jax.vmap`."""
        xp_pt, yp_pt = tree_arg
        return calc_force(xp_pt, yp_pt)
    
    def foo_pmap(tree_arg):
        """
        This function defines the computation to be done on a single device.
        `jax.vmap(foo)` creates a new function that efficiently runs `foo` for a
        whole batch of particle points at once (vectorization).
        """
        # The commented out print statement is a common debugging technique.
        #print(tree_arg)
        # `in_axes=1` is not strictly necessary here since the default is 0, but
        # it indicates that vmap should iterate over the second axis if the
        # input arrays were shaped differently.
        return jax.vmap(foo,in_axes=1)(tree_arg)
    
    # --- Manual Data Parallelism Setup ---
    # To use multiple devices (e.g., multiple GPUs), we manually split the
    # particle data into chunks, one for each available JAX device.
    divider = jax.device_count()
    n = len(xp)//divider # Number of points per device.
    mapped = []
    for i in range(divider):
       # Create a list of chunks, where each chunk is `[xp_chunk, yp_chunk]`.
        mapped.append([xp[i*n:(i+1)*n],yp[i*n:(i+1)*n]])
        
    # `jax.pmap` takes the single-device function `foo_pmap` and runs it in
    # parallel on all available devices, each with its own chunk of data.
    # The results from all devices are then automatically gathered by JAX.
    U_deltas = jax.pmap(foo_pmap)(jnp.array(mapped))
    
    # The result `U_deltas` is a 2D array of shape (num_devices, num_points_per_device).
    # We flatten it back into a single 1D array containing the results for all points.
    return U_deltas.flatten()
