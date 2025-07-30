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
Functions for approximating derivatives on a staggered grid using finite differences.

This module provides implementations of common differential operators like the
gradient, divergence, curl, and Laplacian. These are the fundamental building
blocks used to construct the terms of the Navier-Stokes equations (e.g.,
advection, diffusion, pressure gradient).

**Design Philosophy:**
The functions here operate on `GridVariable` objects and return `GridArray`
objects. This is a deliberate design choice:
- **Input (`GridVariable`)**: Evaluating derivatives requires knowledge of the
  values in neighboring "ghost" cells, which are defined by the boundary
  conditions. The `GridVariable` class encapsulates both the data array and its
  associated boundary conditions.
- **Output (`GridArray`)**: The result of a derivative operation is a new field
  whose own boundary conditions are not well-defined. For example, if a variable
  `c` has a Dirichlet boundary condition `c=0`, it is unclear what the boundary
  condition on its derivative `dc/dx` should be. Therefore, the functions
  return a `GridArray` (data + grid info, but no BCs), and it is the user's
  responsibility to associate new boundary conditions with the result if needed.
"""

import typing
from typing import Optional, Sequence, Tuple
from jax_ib.base import grids
from jax_ib.base import interpolation
import numpy as np
import jax.numpy as jnp

# Type aliases for clarity.
GridArray = grids.GridArray
GridVariable = grids.GridVariable
GridArrayTensor = grids.GridArrayTensor


def stencil_sum(*arrays: GridArray) -> GridArray:
  """
  Sums a collection of GridArrays, averaging their offsets.

  This is a helper function for building finite difference stencils. For example,
  a central difference `(u(i+1) - u(i-1))` involves two `GridArray`s with
  different offsets. This function correctly sums their data and computes the
  new, averaged offset of the resulting `GridArray`.

  Args:
    *arrays: A sequence of `GridArray`s to be summed.

  Returns:
    A new `GridArray` representing the sum.
  """
  # The new offset is the average of the input offsets.
  offset = grids.averaged_offset(*arrays)
  # The new data is the sum of the input data arrays.
  result_data = sum(array.data for array in arrays)
  # Ensure all input arrays are on the same grid.
  grid = grids.consistent_grid(*arrays)
  return grids.GridArray(result_data, offset, grid)

# The following `# pylint: disable` comments are used to suppress linter
# warnings related to function overloading, which is a standard pattern here.

def central_difference(u: GridVariable, axis: int) -> GridArray:
  """
  Approximates the first derivative using a second-order central difference.

  This computes the derivative `du/dx` using the formula:
  `(u(i+1) - u(i-1)) / (2 * dx)`

  This is a second-order accurate scheme. The `u.shift` method is used to
  access the values at neighboring grid points, correctly handling boundary
  conditions.

  Args:
    u: The `GridVariable` to differentiate.
    axis: The integer axis along which to compute the derivative.

  Returns:
    A `GridArray` containing the approximated derivative.
  """
  # Create a stencil representing `u(i+1) - u(i-1)`.
  diff = stencil_sum(u.shift(+1, axis), -u.shift(-1, axis))
  # Divide by the stencil width, `2 * dx`.
  return diff / (2 * u.grid.step[axis])


def backward_difference(u: GridVariable, axis: int) -> GridArray:
  """
  Approximates the first derivative using a first-order backward difference.

  This computes the derivative `du/dx` using the formula:
  `(u(i) - u(i-1)) / dx`

  This is a first-order accurate scheme.

  Args:
    u: The `GridVariable` to differentiate.
    axis: The integer axis along which to compute the derivative.

  Returns:
    A `GridArray` containing the approximated derivative.
  """
  # Create a stencil representing `u(i) - u(i-1)`.
  diff = stencil_sum(u.array, -u.shift(-1, axis))
  # Divide by the grid spacing, `dx`.
  return diff / u.grid.step[axis]


def forward_difference(u: GridVariable, axis: int) -> GridArray:
  """
  Approximates the first derivative using a first-order forward difference.

  This computes the derivative `du/dx` using the formula:
  `(u(i+1) - u(i)) / dx`

  This is a first-order accurate scheme.

  Args:
    u: The `GridVariable` to differentiate.
    axis: The integer axis along which to compute the derivative.

  Returns:
    A `GridArray` containing the approximated derivative.
  """
  # Create a stencil representing `u(i+1) - u(i)`.
  diff = stencil_sum(u.shift(+1, axis), -u.array)
  # Divide by the grid spacing, `dx`.
  return diff / u.grid.step[axis]


def laplacian(u: GridVariable) -> GridArray:
  """
  Approximates the Laplacian operator `∇²u` using a central difference scheme.

  The Laplacian is the divergence of the gradient (`∇ ⋅ ∇u`). This function
  implements the standard second-order stencil for the Laplacian. For 2D, this is:
  `((u(i+1,j) + u(i-1,j) - 2*u(i,j))/dx² + (u(i,j+1) + u(i,j-1) - 2*u(i,j))/dy²)`

  Args:
    u: The `GridVariable` on which to compute the Laplacian.

  Returns:
    A `GridArray` containing the approximated Laplacian.
  """
  # Pre-calculate the `1/dx²` scaling factors for each axis.
  scales = np.square(1 / np.array(u.grid.step, dtype=u.dtype))
  # Start with the central term, `-2 * u * Σ(1/dx_i²)`.
  result = -2 * u.array * np.sum(scales)
  # Add the contributions from neighboring points along each axis.
  for axis in range(u.grid.ndim):
    # This computes `(u(i-1) + u(i+1)) / dx²`.
    result += stencil_sum(u.shift(-1, axis), u.shift(+1, axis)).data * scales[axis]
  # Wrap the raw result array in a GridArray, preserving offset and grid info.
  return grids.GridArray(result, u.offset, u.grid)


def divergence(v: Sequence[GridVariable]) -> GridArray:
  """
  Approximates the divergence of a vector field `v` using backward differences.

  The divergence `∇ ⋅ v` measures the magnitude of a source or sink at a given
  point. For a 2D vector `v = (u, w)`, it is `du/dx + dw/dy`. This function
  approximates the partial derivatives using `backward_difference`.

  This is typically used in the pressure projection step to compute the
  divergence of the intermediate velocity field.

  Args:
    v: A sequence of `GridVariable`s representing the vector field.

  Returns:
    A `GridArray` containing the approximated divergence.
  """
  grid = grids.consistent_grid(*v)
  if len(v) != grid.ndim:
    raise ValueError('The length of `v` must be equal to `grid.ndim`.'
                     f'Expected length {grid.ndim}; got {len(v)}.')
  # Compute the partial derivative for each component along its corresponding axis.
  differences = [backward_difference(u, axis) for axis, u in enumerate(v)]
  # The divergence is the sum of the partial derivatives.
  return sum(differences)


def centered_divergence(v: Sequence[GridVariable]) -> GridArray:
  """
  Approximates the divergence of `v` using centered differences.
  
  This provides a second-order accurate approximation of the divergence, in
  contrast to the first-order `divergence` function above.
  
  Args:
    v: A sequence of `GridVariable`s representing the vector field.

  Returns:
    A `GridArray` containing the approximated divergence.
  """
  grid = grids.consistent_grid(*v)
  if len(v) != grid.ndim:
    raise ValueError('The length of `v` must be equal to `grid.ndim`.'
                     f'Expected length {grid.ndim}; got {len(v)}.')
  # Compute partial derivatives using the more accurate central difference scheme.
  differences = [central_difference(u, axis) for axis, u in enumerate(v)]
  return sum(differences)


def curl_2d(v: Sequence[GridVariable]) -> GridArray:
  """
  Approximates the curl (vorticity) of a 2D vector field `v`.

  In 2D, the curl is a scalar quantity `ω = dv/dx - du/dy` that measures the
  local rotation of the fluid.

  Args:
    v: A sequence of two `GridVariable`s representing the 2D vector field (u, v).

  Returns:
    A `GridArray` containing the scalar vorticity field.
  """
  if len(v) != 2:
    raise ValueError(f'Length of `v` is not 2: {len(v)}')
  grid = grids.consistent_grid(*v)
  if grid.ndim != 2:
    raise ValueError(f'Grid dimensionality is not 2: {grid.ndim}')
  # Compute `dv/dx` using a forward difference.
  dv_dx = forward_difference(v[1], axis=0)
  # Compute `du/dy` using a forward difference.
  du_dy = forward_difference(v[0], axis=1)
  # The 2D curl is the difference between these two.
  return dv_dx - du_dy


def curl_3d(
    v: Sequence[GridVariable]
) -> Tuple[GridArray, GridArray, GridArray]:
  """
  Approximates the curl of a 3D vector field `v`.

  In 3D, the curl is a vector given by:
  `∇ × v = (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy)`

  Args:
    v: A sequence of three `GridVariable`s representing the 3D vector field.

  Returns:
    A tuple of three `GridArray`s for the x, y, and z components of the curl.
  """
  if len(v) != 3:
    raise ValueError(f'Length of `v` is not 3: {len(v)}')
  grid = grids.consistent_grid(*v)
  if grid.ndim != 3:
    raise ValueError(f'Grid dimensionality is not 3: {grid.ndim}')
  # Compute each component of the curl vector.
  curl_x = (forward_difference(v[2], axis=1) - forward_difference(v[1], axis=2))
  curl_y = (forward_difference(v[0], axis=2) - forward_difference(v[2], axis=0))
  curl_z = (forward_difference(v[1], axis=0) - forward_difference(v[0], axis=1))
  return (curl_x, curl_y, curl_z)
