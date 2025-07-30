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
Functions for interpolating values on a grid.

This module provides various numerical schemes for interpolation. Interpolation is a
fundamental operation in CFD, used for two primary purposes:

1.  **Staggered Grid Transfers**: Moving data from one location on the staggered
    grid to another (e.g., from cell centers to cell faces).
2.  **Advection Schemes**: Estimating the value of a quantity at the "departure
    point" or on a control volume face as part of a finite volume advection
    calculation.

The module includes standard methods like linear interpolation, specialized CFD
schemes like upwinding, and high-resolution Total Variation Diminishing (TVD)
schemes that use flux limiters (e.g., Van Leer) to balance accuracy and stability.
"""

from typing import Callable, Optional, Sequence, Tuple, Union
from jax import lax
import jax.numpy as jnp
from jax_ib.base import grids
import numpy as np

# Type aliases for clarity
GridArray = grids.GridArray
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
# Defines the standard signature for an interpolation function.
InterpolationFn = Callable[
    [GridVariable, Tuple[float, ...], GridVariableVector, Optional[float]],
    GridVariable]
# A flux limiter is a function that takes a ratio of gradients and returns a weight.
FluxLimiter = Callable[[grids.Array], grids.Array]


def _linear_along_axis(
    c: GridVariable,
    offset: float,
    axis: int
) -> GridVariable:
  """
  Performs linear interpolation of `c` to a new `offset` along a single axis.

  This is a helper function that forms the building block for multi-dimensional
  linear interpolation.

  Args:
    c: The `GridVariable` to be interpolated.
    offset: The target offset along the specified `axis`.
    axis: The integer index of the axis to interpolate along.

  Returns:
    A new `GridVariable` with its data interpolated to the new offset.
  """
  # Calculate the distance to interpolate.
  offset_delta = offset - c.offset[axis]

  # If the offset is the same, no interpolation is needed.
  if np.isclose(offset_delta, 0):
    return c

  # Construct the new offset tuple for the output GridVariable.
  new_offset = tuple(offset if j == axis else o for j, o in enumerate(c.offset))

  # If the offset delta is a whole number, interpolation is just a simple shift.
  if np.isclose(offset_delta, int(offset_delta)):
    shifted_array = c.shift(int(round(offset_delta)), axis)
    return grids.GridVariable(
        array=grids.GridArray(data=shifted_array.data,
                              offset=new_offset,
                              grid=c.grid),
        bc=c.bc)

  # For fractional offsets, perform linear interpolation: y = w1*y1 + w2*y2.
  # Find the two neighboring grid points (floor and ceil).
  floor = int(np.floor(offset_delta))
  ceil = int(np.ceil(offset_delta))
  # Calculate the weights for each neighbor. The closer neighbor gets more weight.
  floor_weight = ceil - offset_delta
  ceil_weight = 1.0 - floor_weight
  # Get the data at the neighbor locations using `.shift()`
  floor_data = c.shift(floor, axis).data
  ceil_data = c.shift(ceil, axis).data
  # Compute the weighted average.
  data = floor_weight * floor_data + ceil_weight * ceil_data
  
  return grids.GridVariable(
      array=grids.GridArray(data, new_offset, c.grid), bc=c.bc)


def linear(
    c: GridVariable,
    offset: Tuple[float, ...],
    v: Optional[GridVariableVector] = None,
    dt: Optional[float] = None
) -> grids.GridVariable:
  """
  Performs multi-linear interpolation of `c` to a new `offset`.

  This function achieves multi-dimensional interpolation by applying 1D linear
  interpolation sequentially along each axis where the offset differs.

  Args:
    c: The quantity to be interpolated.
    offset: The target offset to which `c` will be interpolated.
    v: Velocity field (unused by this function, but part of the standard API).
    dt: Time step (unused by this function, but part of the standard API).

  Returns:
    A `GridVariable` containing the values of `c` at the new `offset`.
  """
  del v, dt  # Mark unused arguments.
  if len(offset) != len(c.offset):
    raise ValueError('`c.offset` and `offset` must have the same length')
  
  # Start with the original variable.
  interpolated = c
  # Sequentially apply 1D interpolation for each axis.
  for axis, target_offset_axis in enumerate(offset):
    interpolated = _linear_along_axis(interpolated, offset=target_offset_axis, axis=axis)
  return interpolated


def upwind(
    c: GridVariable,
    offset: Tuple[float, ...],
    v: GridVariableVector,
    dt: Optional[float] = None
) -> GridVariable:
  """
  Performs first-order upwind interpolation of `c` to `offset`.

  Upwind schemes are common in CFD for their stability. The value at the new
  location is taken from the "upwind" or "upstream" direction, as determined
  by the velocity `v`. If velocity is positive, we take the value from the
  cell behind (e.g., at `i-1`); if negative, we take it from the cell ahead
  (e.g., at `i+1`). This method is only first-order accurate and introduces
  numerical diffusion, but it prevents unphysical oscillations.

  Args:
    c: The quantity to be interpolated.
    offset: The target offset (typically a cell face). Must differ from
      `c.offset` along only one axis.
    v: The velocity field, used to determine the "upwind" direction.
    dt: Time step (unused).

  Returns:
    A `GridVariable` containing the upwind-interpolated values.
  """
  del dt  # Mark unused argument.
  if c.offset == offset: return c

  # Determine the single axis along which interpolation is occurring.
  interpolation_axes = tuple(
      axis for axis, (current, target) in enumerate(zip(c.offset, offset))
      if not np.isclose(current, target)
  )
  if len(interpolation_axes) != 1:
    raise grids.InconsistentOffsetError(
        'Upwind interpolation requires `c.offset` and `target offset` to '
        f'differ in exactly one entry. Got: {c.offset} and {offset}.')
  axis, = interpolation_axes
  
  # Get the velocity component normal to the target face.
  u = v[axis]
  
  offset_delta = u.offset[axis] - c.offset[axis]
  # Determine the "upwind" and "downwind" neighbors.
  floor = int(np.floor(offset_delta))  # Upwind neighbor for positive velocity
  ceil = int(np.ceil(offset_delta))   # Upwind neighbor for negative velocity
  
  # Use `jnp.where` to select data from the correct neighbor based on the sign
  # of the velocity at each point on the grid.
  array = grids.where(
      u.array > 0, c.shift(floor, axis), c.shift(ceil, axis)
  )
  
  grid = grids.consistent_grid(c, u)
  # Return a new GridVariable with the interpolated data. Note that the BC
  # is often simplified after interpolation, here using a default periodic BC.
  return grids.GridVariable(
      array=grids.GridArray(array.data, offset, grid),
      bc=boundaries.periodic_boundary_conditions(grid.ndim))


def lax_wendroff(
    c: GridVariable,
    offset: Tuple[float, ...],
    v: Optional[GridVariableVector] = None,
    dt: Optional[float] = None
) -> GridVariable:
  """
  Performs second-order Lax-Wendroff interpolation.

  This scheme achieves second-order accuracy by including a correction term to
  the first-order upwind value. This correction is derived from a Taylor series
  expansion in time. While more accurate than upwinding, as a linear second-order
  scheme, it is not guaranteed to be monotonic and can introduce spurious
  oscillations near sharp gradients (Godunov's theorem). It is often used as the
  high-order component in TVD schemes with flux limiters.

  Args:
    c: The quantity to be interpolated.
    offset: The target offset.
    v: The velocity field.
    dt: The time step, which is required for the correction term.

  Returns:
    A `GridVariable` with the interpolated values.
  """
  if c.offset == offset: return c
  # Determine the interpolation axis.
  interpolation_axes = tuple(
      axis for axis, (current, target) in enumerate(zip(c.offset, offset))
      if not np.isclose(current, target)
  )
  if len(interpolation_axes) != 1:
    raise grids.InconsistentOffsetError(
        'Lax-Wendroff requires offsets to differ in one entry.')
  axis, = interpolation_axes
  u = v[axis]
  
  offset_delta = u.offset[axis] - c.offset[axis]
  floor = int(np.floor(offset_delta))
  ceil = int(np.ceil(offset_delta))
  
  grid = grids.consistent_grid(c, u)
  # Calculate the Courant number `C = u * dt / dx`.
  courant_numbers = (dt / grid.step[axis]) * u.data
  
  # The formula for positive velocity: c_upwind + 0.5 * (1 - C) * (c_downwind - c_upwind)
  positive_u_case = (
      c.shift(floor, axis).data + 0.5 * (1 - courant_numbers) *
      (c.shift(ceil, axis).data - c.shift(floor, axis).data))
  # The formula for negative velocity.
  negative_u_case = (
      c.shift(ceil, axis).data - 0.5 * (1 + courant_numbers) *
      (c.shift(ceil, axis).data - c.shift(floor, axis).data))
      
  # Select the appropriate formula based on the sign of the velocity.
  array = grids.where(u.array > 0, positive_u_case, negative_u_case)
  
  return grids.GridVariable(
      array=grids.GridArray(array.data, offset, grid),
      bc=boundaries.periodic_boundary_conditions(grid.ndim))


def safe_div(x: Array, y: Array, default_numerator: float = 1.0) -> Array:
  """
  Performs division, returning `x / default_numerator` where `y` is zero.
  This is a utility to prevent division-by-zero errors and `NaN` propagation.
  """
  return x / jnp.where(y != 0, y, default_numerator)


def van_leer_limiter(r: Array) -> Array:
  """
  The Van Leer flux limiter function.

  Flux limiters are used in TVD schemes. They take `r`, the ratio of successive
  gradients, as input. The output `phi(r)` is a weighting factor that blends
  between a high-order and low-order scheme. `phi=0` gives the low-order (upwind)
  scheme, and `phi=1` would give a pure central difference scheme. The Van Leer
  limiter smoothly transitions between these to maintain stability.
  
  Args:
    r: The ratio of successive gradients `(c_i - c_{i-1}) / (c_{i+1} - c_i)`.

  Returns:
    The limiter value `phi(r)`.
  """
  # The formula is `phi(r) = (r + |r|) / (1 + |r|)`.
  # This can be simplified to `2r / (1 + r)` for `r > 0` and `0` otherwise.
  return jnp.where(r > 0, safe_div(2 * r, 1 + r), 0.0)


def apply_tvd_limiter(
    interpolation_fn: InterpolationFn,
    limiter: FluxLimiter = van_leer_limiter
) -> InterpolationFn:
  """
  A meta-function that creates a TVD interpolation scheme.

  It takes a high-order (but potentially oscillatory) interpolation function and a
  flux limiter, and returns a new, robust TVD interpolation function. The new
  function computes a value by adding a "limited" correction to the stable,
  first-order upwind scheme:
  `c_tvd = c_upwind + phi(r) * (c_high_order - c_upwind)`

  Args:
    interpolation_fn: The high-order scheme (e.g., `lax_wendroff`).
    limiter: The flux limiter function (e.g., `van_leer_limiter`).

  Returns:
    A new `InterpolationFn` that implements the TVD scheme.
  """
  def tvd_interpolation(
      c: GridVariable,
      offset: Tuple[float, ...],
      v: GridVariableVector,
      dt: float,
  ) -> GridVariable:
    """The generated TVD interpolation function."""
    # This implementation is currently restricted to interpolating to a single
    # adjacent face at a time.
    # Determine the interpolation axis.
    interpolation_axes = tuple(
        axis for axis, (current, target) in enumerate(zip(c.offset, offset))
        if not np.isclose(current, target)
    )
    if len(interpolation_axes) != 1:
        raise NotImplementedError('tvd_interpolation only supports interpolation along a single axis.')
    axis, = interpolation_axes

    # Compute the low-order (stable) and high-order (accurate) interpolated values.
    c_low = upwind(c, offset, v, dt)
    c_high = interpolation_fn(c, offset, v, dt)

    # Get the stencil of values needed to compute the ratio of gradients `r`.
    c_left = c.shift(-1, axis)
    c_right = c.shift(1, axis)
    c_next_right = c.shift(2, axis)

    # The definition of `r` depends on the flow direction.
    # For positive velocity, `r = (c_i - c_{i-1}) / (c_{i+1} - c_i)`.
    positive_u_r = safe_div(c.data - c_left.data, c_right.data - c.data)
    # For negative velocity, the stencil is shifted.
    negative_u_r = safe_div(c_next_right.data - c_right.data, c_right.data - c.data)
    
    # Evaluate the limiter `phi(r)` for both cases.
    positive_u_phi = grids.GridArray(limiter(positive_u_r), c_low.offset, c.grid)
    negative_u_phi = grids.GridArray(limiter(negative_u_r), c_low.offset, c.grid)
    
    u = v[axis]
    # Choose the correct limiter value based on the velocity direction.
    phi = grids.where(u.array > 0, positive_u_phi, negative_u_phi)
    
    # Apply the final TVD formula.
    c_interpolated = c_low.array + (c_high.array - c_low.array) * phi
    
    return grids.GridVariable(c_interpolated, c.bc)
  return tvd_interpolation


def point_interpolation(
    point: Array,
    c: GridArray,
    order: int = 1,
) -> jax.Array:
  """
  Interpolates the value of a `GridArray` `c` at an arbitrary off-grid point.
  
  This is useful for post-processing or for coupling with Lagrangian methods
  where data is needed at specific, non-grid-aligned locations. It uses
  `jax.scipy.ndimage.map_coordinates` which is JAX's equivalent of Scipy's
  general-purpose interpolation tool.

  Args:
    point: A 1D array of shape `(ndim,)` specifying the physical coordinates
      of the point to interpolate to.
    c: The N-dimensional `GridArray` containing the source data.
    order: The order of spline interpolation (0=nearest neighbor, 1=linear).

  Returns:
    The interpolated value at `point`.
  """
  point = jnp.asarray(point)

  # Get the physical domain boundaries and grid properties.
  domain_lower, domain_upper = zip(*c.grid.domain)
  domain_lower = jnp.array(domain_lower)
  domain_upper = jnp.array(domain_upper)
  shape = jnp.array(c.grid.shape)
  offset = jnp.array(c.offset)
  
  # Convert the physical `point` coordinates into fractional grid indices.
  # This is a linear mapping from the physical domain to the index space.
  # For example, a point at `domain_lower` maps to index `-offset`.
  index = (-offset + (point - domain_lower) * shape /
           (domain_upper - domain_lower))

  # Use `map_coordinates` to perform the interpolation at the fractional indices.
  # This function handles multi-dimensional spline interpolation efficiently.
  # Note: `mode` and `cval` from the original code are not passed, implying defaults.
  return jax.scipy.ndimage.map_coordinates(
      c.data, coordinates=index, order=order)
