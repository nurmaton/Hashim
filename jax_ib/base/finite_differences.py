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

**Design Philosophy on a Staggered Grid:**
The functions in this module are carefully designed to handle the complexities
of a staggered grid, where different physical quantities (e.g., velocity
components, pressure) are located at different positions within a grid cell
(faces vs. centers). The core design principle is:

- **Input (`GridVariable`)**: To evaluate a derivative, a function needs access
  to values in neighboring "ghost" cells, which are defined by the variable's
  boundary conditions. All functions here therefore take `GridVariable` objects
  as input, because this class encapsulates both the data array and its
  associated boundary conditions. The internal `.shift()` method automatically
  handles the BCs when accessing neighbor data.

- **Output (`GridArray`)**: The result of a derivative operation is a new field
  whose physical location (offset) on the grid is different from the input,
  and whose own boundary conditions are not well-defined. For example, if a
  variable `c` has a Dirichlet BC `c=0`, it is unclear what the BC on its
  derivative `dc/dx` should be. To avoid making unsafe assumptions, all
  functions return a `GridArray` (data + grid info + new offset, but no BCs).
  It is the responsibility of the calling function to associate new boundary
  conditions with this result if needed.

This design ensures that all derivative calculations are explicit and
numerically correct with respect to the staggered grid layout.
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
  different offsets and a coefficient of -1 on the second. This function
  correctly sums their data and computes the new, averaged offset of the
  resulting `GridArray`.

  Args:
    *arrays: A sequence of `GridArray`s to be summed.

  Returns:
    A new `GridArray` representing the sum.
  """
  # The new offset is the average of the input offsets. For `u(i+1)` and `u(i-1)`,
  # the result is centered at `i`, which is the correct location for the derivative.
  offset = grids.averaged_offset(*arrays)
  # The new data is the sum of the input data arrays.
  # The type ignore comment is to suppress a potential false positive from a linter.
  result = sum(array.data for array in arrays)  # type: ignore
  # Ensure all input arrays are on the same grid.
  grid = grids.consistent_grid(*arrays)
  # Return a new GridArray with the summed data and averaged offset.
  return grids.GridArray(result, offset, grid)


# The following `# pylint: disable` comments are used to suppress linter
# warnings related to function overloading, which is a standard pattern here
# to provide clear type hints for different use cases.

# This overload signature tells type checkers that if `axis` is an integer,
# the function returns a single GridArray.
@typing.overload
def central_difference(u: GridVariable, axis: int) -> GridArray:
  ...

# This overload signature tells type checkers that if `axis` is a tuple of
# integers, the function returns a tuple of GridArrays.
@typing.overload
def central_difference(
    u: GridVariable, axis: Optional[Tuple[int, ...]]) -> Tuple[GridArray, ...]:
  ...


def central_difference(u, axis=None):
  """
  Approximates the first derivative using a second-order central difference.

  This computes the derivative `du/dx` using the formula:
  `(u(i+1) - u(i-1)) / (2 * dx)`

  This is a second-order accurate scheme. The `u.shift` method is used to
  access the values at neighboring grid points, correctly handling boundary
  conditions through ghost cells.

  Args:
    u: The `GridVariable` to differentiate.
    axis: The integer axis along which to compute the derivative. If None,
      computes the gradient (derivatives along all axes).

  Returns:
    A `GridArray` containing the approximated derivative if `axis` is an int,
    or a tuple of `GridArray`s if `axis` is a tuple or None.
  """
  # If no axis is specified, compute the gradient (derivatives along all axes).
  if axis is None:
    axis = range(u.grid.ndim)
  # If `axis` is a sequence, recursively call this function for each axis.
  if not isinstance(axis, int):
    return tuple(central_difference(u, a) for a in axis)
    
  # Create a stencil representing `u(i+1) - u(i-1)`. The `stencil_sum`
  # handles the subtraction and calculates the correct centered offset.
  diff = stencil_sum(u.shift(+1, axis), -u.shift(-1, axis))
  
  # Divide by the stencil width, `2 * dx`, to get the final derivative.
  return diff / (2 * u.grid.step[axis])


# This overload signature tells type checkers that if `axis` is an integer,
# the function returns a single GridArray.
@typing.overload
def backward_difference(u: GridVariable, axis: int) -> GridArray:
  ...

# This overload signature tells type checkers that if `axis` is a tuple of
# integers, the function returns a tuple of GridArrays.
@typing.overload
def backward_difference(
    u: GridVariable, axis: Optional[Tuple[int, ...]]) -> Tuple[GridArray, ...]:
  ...


def backward_difference(u, axis=None):
  """
  Approximates the first derivative using a first-order backward difference.

  This computes the derivative `du/dx` using the formula:
  `(u(i) - u(i-1)) / dx`

  This is a first-order accurate scheme. It is often used to compute the
  divergence on a staggered grid because it naturally results in a quantity
  defined at the cell center.

  Args:
    u: The `GridVariable` to differentiate.
    axis: The integer axis along which to compute the derivative. If None,
      computes the derivatives along all axes.

  Returns:
    A `GridArray` containing the approximated derivative if `axis` is an int,
    or a tuple of `GridArray`s if `axis` is a tuple or None.
  """
  # If no axis is specified, compute the derivatives along all axes.
  if axis is None:
    axis = range(u.grid.ndim)
  # If `axis` is a sequence, recursively call this function for each axis.
  if not isinstance(axis, int):
    return tuple(backward_difference(u, a) for a in axis)
    
  # Create a stencil representing `u(i) - u(i-1)`.
  # The `stencil_sum` correctly averages the offsets of `u` at `i` and `i-1`,
  # placing the result at `i - 0.5`.
  diff = stencil_sum(u.array, -u.shift(-1, axis))
  
  # Divide by the grid spacing, `dx`, to get the final derivative.
  return diff / u.grid.step[axis]


# This overload signature tells type checkers that if `axis` is an integer,
# the function returns a single GridArray.
@typing.overload
def forward_difference(u: GridVariable, axis: int) -> GridArray:
  ...

# This overload signature tells type checkers that if `axis` is a tuple of
# integers, the function returns a tuple of GridArrays.
@typing.overload
def forward_difference(
    u: GridVariable,
    axis: Optional[Tuple[int, ...]] = ...) -> Tuple[GridArray, ...]:
  ...


def forward_difference(u, axis=None):
  """
  Approximates the first derivative using a first-order forward difference.

  This computes the derivative `du/dx` using the formula:
  `(u(i+1) - u(i)) / dx`

  This is a first-order accurate scheme. It is often used to compute the
  gradient on a staggered grid.

  Args:
    u: The `GridVariable` to differentiate.
    axis: The integer axis along which to compute the derivative. If None,
      computes the derivatives along all axes.

  Returns:
    A `GridArray` containing the approximated derivative if `axis` is an int,
    or a tuple of `GridArray`s if `axis` is a tuple or None.
  """
  # If no axis is specified, compute the derivatives along all axes.
  if axis is None:
    axis = range(u.grid.ndim)
  # If `axis` is a sequence, recursively call this function for each axis.
  if not isinstance(axis, int):
    return tuple(forward_difference(u, a) for a in axis)
    
  # Create a stencil representing `u(i+1) - u(i)`.
  # The `stencil_sum` correctly averages the offsets of `u` at `i+1` and `i`,
  # placing the result at `i + 0.5`.
  diff = stencil_sum(u.shift(+1, axis), -u.array)
  
  # Divide by the grid spacing, `dx`, to get the final derivative.
  return diff / u.grid.step[axis]


def laplacian(u: GridVariable) -> GridArray:
  """
  Approximates the Laplacian operator `∇²u` using a central difference scheme.

  The Laplacian is the divergence of the gradient (`∇ ⋅ ∇u`). This function
  implements the standard second-order, nearest-neighbor stencil for the
  Laplacian. For 2D, this is:
  `((u(i+1,j) + u(i-1,j) - 2*u(i,j))/dx² + (u(i,j+1) + u(i,j-1) - 2*u(i,j))/dy²)`

  Args:
    u: The `GridVariable` on which to compute the Laplacian.

  Returns:
    A `GridArray` containing the approximated Laplacian. The result has the
    same offset as the input `u`.
  """
  # Pre-calculate the `1/dx²` scaling factors for each axis. Using `np.square`
  # is fine here as grid steps are static, not JAX-traced arrays.
  scales = np.square(1 / np.array(u.grid.step, dtype=u.dtype)) 
  
  # The commented out line shows an alternative using jnp, which is also valid.
  #scales = jnp.square(1 / jnp.array(u.grid.step, dtype=u.dtype))
  
  # Start with the central term of the stencil, `-2 * u * Σ(1/dx_i²)`.
  result = -2 * u.array.data * jnp.sum(scales)
  
  # Add the contributions from neighboring points along each axis.
  for axis in range(u.grid.ndim):
    # This computes `(u(i-1) + u(i+1)) / dx²` for the current axis.
    # `stencil_sum` is used to combine the shifted arrays.
    result += stencil_sum(u.shift(-1, axis), u.shift(+1, axis)).data * scales[axis]
    
  # Wrap the raw result array in a GridArray, preserving the original offset and grid info.
  return grids.GridArray(result, u.offset, u.grid)


def divergence(v: Sequence[GridVariable]) -> GridArray:
  """
  Approximates the divergence of a vector field `v` using backward differences.

  The divergence `∇ ⋅ v` measures the magnitude of a source or sink at a given
  point. For a 2D vector `v = (u, w)`, it is `∂u/∂x + ∂w/∂y`. This function
  approximates the partial derivatives using `backward_difference`.

  This is the standard choice for the pressure Poisson equation on a staggered
  grid, as applying backward differences to face-centered velocities `u` and `w`
  naturally produces a divergence field located at the cell center, where
  pressure is defined.

  Args:
    v: A sequence of `GridVariable`s representing the vector field components.

  Returns:
    A `GridArray` containing the approximated divergence, located at the cell center.
  """
  grid = grids.consistent_grid(*v)
  if len(v) != grid.ndim:
    raise ValueError('The length of `v` must be equal to `grid.ndim`.'
                     f'Expected length {grid.ndim}; got {len(v)}.')
                     
  # Compute the partial derivative for each component along its corresponding axis.
  # e.g., ∂(v_x)/∂x, ∂(v_y)/∂y, ...
  differences = [backward_difference(u, axis) for axis, u in enumerate(v)]
  
  # The divergence is the sum of the partial derivatives. `sum()` here works on
  # the list of GridArray objects, correctly summing their data and averaging their offsets.
  return sum(differences)


def centered_divergence(v: Sequence[GridVariable]) -> GridArray:
  """
  Approximates the divergence of `v` using centered differences.
  
  This provides a second-order accurate approximation of the divergence, in
  contrast to the first-order `divergence` function above. While more accurate,
  it may not be suitable for the pressure Poisson equation on a staggered grid
  as the result will not be located at the cell center. It can be useful for
  other physical calculations or diagnostics.
  
  Args:
    v: A sequence of `GridVariable`s representing the vector field components.

  Returns:
    A `GridArray` containing the approximated divergence.
  """
  # Ensure all velocity components are on the same grid.
  grid = grids.consistent_grid(*v)
  if len(v) != grid.ndim:
    raise ValueError('The length of `v` must be equal to `grid.ndim`.'
                     f'Expected length {grid.ndim}; got {len(v)}.')
                     
  # Compute partial derivatives using the more accurate central difference scheme.
  differences = [central_difference(u, axis) for axis, u in enumerate(v)]
  
  # The divergence is the sum of the partial derivatives. `sum()` works on the
  # list of GridArray objects, summing their data and averaging their offsets.
  return sum(differences)


# This overload signature tells type checkers that if the input `v` is a single
# GridVariable, the function returns a GridArrayTensor.
@typing.overload
def gradient_tensor(v: GridVariable) -> GridArrayTensor:
  ...

# This overload signature tells type checkers that if the input `v` is a sequence
# of GridVariables, the function also returns a GridArrayTensor.
@typing.overload
def gradient_tensor(v: Sequence[GridVariable]) -> GridArrayTensor:
  ...


def gradient_tensor(v):
  """
  Approximates the cell-centered gradient of a scalar or vector field `v`.

  The gradient of a scalar `s` is a vector `(∂s/∂x, ∂s/∂y, ...)`.
  The gradient of a vector `v` is a tensor `(∇v)_ij = ∂v_i/∂x_j`.

  This function computes the gradient and returns it as a `GridArrayTensor`,
  which allows for natural tensor algebra (e.g., `grad.T` for transpose). It
  intelligently chooses the finite difference scheme to try and produce a result
  located at the cell center.

  Args:
    v: A `GridVariable` (for a scalar field) or a sequence of `GridVariable`s
      (for a vector field).

  Returns:
    A `GridArrayTensor` representing the gradient. For a scalar input, this will
    be a rank-1 tensor (a vector). For a vector input, it will be a rank-2 tensor.
  """
  # If the input `v` is a sequence (i.e., a vector field), recursively call this
  # function on each component and stack the resulting gradient vectors into a tensor.
  if not isinstance(v, grids.GridVariable):
    return grids.GridArrayTensor(np.stack([gradient_tensor(u) for u in v], axis=-1))
    
  # If the input is a single GridVariable (a scalar field), compute its gradient vector.
  grad = []
  for axis in range(v.grid.ndim):
    offset = v.offset[axis]
    # Choose a differencing scheme based on the offset of the input data to
    # produce a result that is centered as much as possible.
    if np.isclose(offset, 0.0):
      # Data at the lower face: use a forward difference to move to the center.
      derivative = forward_difference(v, axis)
    elif np.isclose(offset, 1.0):
      # Data at the upper face: use a backward difference to move to the center.
      derivative = backward_difference(v, axis)
    elif np.isclose(offset, 0.5):
      # Data is already centered in this dimension, use a central difference.
      # No need to interpolate first, central_difference is already centered.
      derivative = central_difference(v, axis)
    else:
      raise ValueError(f'expected offset values in {{0, 0.5, 1}}, got {offset}')
    grad.append(derivative)
    
  # Return the list of gradient components as a GridArrayTensor.
  return grids.GridArrayTensor(grad)


def curl_2d(v: Sequence[GridVariable]) -> GridArray:
  """
  Approximates the curl (vorticity) of a 2D vector field `v`.

  In 2D, the curl is a scalar quantity `ω = ∂v/∂x - ∂u/∂y` that measures the
  local, instantaneous rotation of the fluid. Vorticity is a key quantity in
  the analysis of fluid flows.

  Args:
    v: A sequence of two `GridVariable`s representing the 2D vector field (u, v).

  Returns:
    A `GridArray` containing the scalar vorticity field.
  """
  # Validate input dimensions.
  if len(v) != 2:
    raise ValueError(f'Length of `v` is not 2: {len(v)}')
  grid = grids.consistent_grid(*v)
  if grid.ndim != 2:
    raise ValueError(f'Grid dimensionality is not 2: {grid.ndim}')
    
  # Compute `∂v/∂x` using a forward difference.
  dv_dx = forward_difference(v[1], axis=0)
  # Compute `∂u/∂y` using a forward difference.
  du_dy = forward_difference(v[0], axis=1)
  
  # The 2D curl is the difference between these two partial derivatives.
  # The subtraction is handled by the `__array_ufunc__` of the GridArray class.
  return dv_dx - du_dy


def curl_3d(
    v: Sequence[GridVariable]
) -> Tuple[GridArray, GridArray, GridArray]:
  """
  Approximates the curl of a 3D vector field `v`.

  In 3D, the curl is a vector given by the cross product of the del operator
  and the velocity vector: `∇ × v`. It describes the axis and magnitude of the
  local rotation.

  `∇ × v = (∂w/∂y - ∂v/∂z)î + (∂u/∂z - ∂w/∂x)ĵ + (∂v/∂x - ∂u/∂y)k̂`

  Args:
    v: A sequence of three `GridVariable`s representing the 3D vector field (u, v, w).

  Returns:
    A tuple of three `GridArray`s for the x, y, and z components of the curl vector.
  """
  # Validate input dimensions.
  if len(v) != 3:
    raise ValueError(f'Length of `v` is not 3: {len(v)}')
  grid = grids.consistent_grid(*v)
  if grid.ndim != 3:
    raise ValueError(f'Grid dimensionality is not 3: {grid.ndim}')
    
  # Compute each component of the curl vector using the formula above.
  # `v[0]` is u, `v[1]` is v, `v[2]` is w.
  # `axis=0` is x, `axis=1` is y, `axis=2` is z.
  
  # x-component: (∂w/∂y - ∂v/∂z)
  curl_x = (forward_difference(v[2], axis=1) - forward_difference(v[1], axis=2))
  # y-component: (∂u/∂z - ∂w/∂x)
  curl_y = (forward_difference(v[0], axis=2) - forward_difference(v[2], axis=0))
  # z-component: (∂v/∂x - ∂u/∂y)
  curl_z = (forward_difference(v[1], axis=0) - forward_difference(v[0], axis=1))
  
  return (curl_x, curl_y, curl_z)
