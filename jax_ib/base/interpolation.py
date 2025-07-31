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
Functions for interpolating values on a staggered grid.

This module provides various numerical schemes for interpolation, which is a
fundamental operation in CFD simulations. Interpolation is used for two primary
purposes:

1.  **Staggered Grid Transfers**: Moving data from one location on the staggered
    grid to another (e.g., from cell centers to cell faces). This is essential
    for calculating fluxes and evaluating terms that involve quantities at
    different locations.

2.  **Finite Volume Advection**: As part of an advection calculation, these
    schemes are used to estimate the value of a quantity on a control volume
    face. The choice of interpolation scheme directly determines the accuracy,
    stability, and conservation properties of the advection algorithm.

The module provides a hierarchy of common schemes with different trade-offs:

-   `linear`: A standard, second-order accurate scheme. It is accurate for
    smooth flows but can introduce unphysical oscillations ("wiggles") near
    sharp gradients.

-   `upwind`: A first-order scheme that is very robust and guaranteed to be
    non-oscillatory. Its stability comes at the cost of introducing significant
    numerical diffusion, which can smear out sharp features.

-   `lax_wendroff`: A second-order scheme that is more accurate than upwind but
    is not monotonic. It is typically not used alone but serves as the
    high-order component in more advanced schemes.

-   **Total Variation Diminishing (TVD) Schemes**: Implemented via the
    `apply_tvd_limiter` factory, these high-resolution methods (e.g., using
    `van_leer_limiter`) dynamically blend a low-order (upwind) and high-order
    (Lax-Wendroff) scheme. They aim to achieve second-order accuracy in smooth
    regions while reverting to the stable first-order scheme near shocks to
    prevent oscillations, offering a good balance of accuracy and robustness.

Finally, `point_interpolation` provides a general utility for interpolating
field values at arbitrary off-grid locations.
"""

from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax_ib.base import boundaries
from jax_ib.base import grids
import numpy as np


# --- Type Aliases ---
Array = Union[np.ndarray, jax.Array]
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
# Defines the standard signature for an interpolation function. It takes the
# variable to interpolate, a target offset, an optional velocity field, and an
# optional time step, and returns the interpolated variable.
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
  Performs linear interpolation of `c` to a new `offset` along a single specified `axis`.

  This is a helper function that forms the building block for multi-dimensional
  linear interpolation. It calculates a weighted average of the two grid points
  that bracket the target location.

  Args:
    c: The `GridVariable` to be interpolated.
    offset: The target offset along the specified `axis`.
    axis: The integer index of the axis to interpolate along.

  Returns:
    A new `GridVariable` with its data interpolated to the new offset.
  """
  # Calculate the distance to interpolate in grid units.
  offset_delta = offset - c.offset[axis]

  # If the offset is the same, no interpolation is needed. Return the original variable.
  if np.isclose(offset_delta, 0):
    return c

  # Construct the new offset tuple for the output GridVariable.
  new_offset = tuple(offset if j == axis else o
                     for j, o in enumerate(c.offset))

  # If the offset delta is a whole number, interpolation is just a simple shift.
  # The `.shift()` method correctly handles boundary conditions.
  if np.isclose(offset_delta, int(offset_delta)):
    shifted_array = c.shift(int(round(offset_delta)), axis)
    return grids.GridVariable(
        array=grids.GridArray(data=shifted_array.data,
                              offset=new_offset,
                              grid=c.grid),
        bc=c.bc)

  # For fractional offsets, perform standard linear interpolation: y = w1*y1 + w2*y2.
  # Find the two neighboring grid points (floor and ceil) relative to the current position.
  floor = int(np.floor(offset_delta))
  ceil = int(np.ceil(offset_delta))
  # Calculate the weights for each neighbor. The closer neighbor gets more weight.
  floor_weight = ceil - offset_delta
  ceil_weight = 1. - floor_weight
  # Get the data at the neighbor locations using `.shift()` and compute the weighted average.
  data = (floor_weight * c.shift(floor, axis).data +
          ceil_weight * c.shift(ceil, axis).data)
          
  # Return a new GridVariable containing the interpolated data and updated offset.
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
  interpolation sequentially along each axis where the offset differs. For example,
  to move from a cell center (0.5, 0.5) to a corner (0, 0), it first interpolates
  along x to (0, 0.5), then interpolates that result along y to (0, 0).

  Args:
    c: The quantity to be interpolated.
    offset: The target offset to which `c` will be interpolated. Must have the
      same length as `c.offset`.
    v: Velocity field (unused by this function, but part of the standard API).
    dt: Size of the time step (unused by this function, but part of the standard API).

  Returns:
    A `GridVariable` containing the values of `c` at the new `offset`.
  """
  del v, dt  # Mark unused arguments for linters.
  # Validate that the target offset has the correct number of dimensions.
  if len(offset) != len(c.offset):
    raise ValueError('`c.offset` and `offset` must have the same length;'
                     f'got {c.offset} and {offset}.')
  # Start with the original variable.
  interpolated = c
  # Sequentially apply 1D interpolation for each axis.
  for a, o in enumerate(offset):
    interpolated = _linear_along_axis(interpolated, offset=o, axis=a)
  return interpolated


def upwind(
    c: GridVariable,
    offset: Tuple[float, ...],
    v: GridVariableVector,
    dt: Optional[float] = None
) -> GridVariable:
  """
  Performs first-order upwind interpolation of `c` to `offset`.

  Upwind schemes are common in CFD for their stability, especially when advecting
  sharp gradients. The value at the new location (e.g., a cell face) is taken
  entirely from the "upwind" or "upstream" direction, as determined by the sign
  of the velocity `v` at that face.

  - If velocity at the face is positive, the value is taken from the cell behind.
  - If velocity at the face is negative, the value is taken from the cell ahead.

  This method is only first-order accurate and introduces numerical diffusion
  (which can smear sharp features), but it robustly prevents unphysical
  oscillations that higher-order linear schemes can create.

  Args:
    c: The `GridVariable` (e.g., a scalar concentration) to be interpolated.
    offset: The target offset (typically a cell face). Must differ from
      `c.offset` along only one axis.
    v: The velocity `GridVariableVector`, used to determine the "upwind" direction.
    dt: Size of the time step (unused by this function, kept for API consistency).

  Returns:
    A `GridVariable` containing the upwind-interpolated values at the new `offset`.

  Raises:
    InconsistentOffsetError: if `offset` and `c.offset` differ in more than one
    entry, as this method is defined for 1D interpolation at a time.
  """
  # Mark `dt` as unused to satisfy linters.
  del dt
  
  # If the source and target offsets are the same, no interpolation is needed.
  if c.offset == offset: return c
  
  # Determine the single axis along which interpolation is occurring by finding
  # where the source and target offsets differ.
  interpolation_axes = tuple(
      axis for axis, (current, target) in enumerate(zip(c.offset, offset))
      if not np.isclose(current, target)
  )
  
  # This implementation only supports interpolation along a single axis at a time.
  if len(interpolation_axes) != 1:
    raise grids.InconsistentOffsetError(
        f'for upwind interpolation `c.offset` and `offset` must differ at most '
        f'in one entry, but got: {c.offset} and {offset}.')
  axis, = interpolation_axes
  
  # Get the velocity component normal to the target face (i.e., along the interpolation axis).
  u = v[axis]
  
  # Calculate the distance between the source and target locations in grid units.
  offset_delta = u.offset[axis] - c.offset[axis]

  # If offsets differ by a whole number, the operation is just a simple shift.
  if np.isclose(offset_delta, int(offset_delta)):
    return grids.GridVariable(
        array=grids.GridArray(data=c.shift(int(round(offset_delta)), axis).data,
                              offset=offset,
                              grid=grids.consistent_grid(c, u)),
        bc=c.bc)

  # Determine the "upwind" and "downwind" neighbors relative to `c`.
  # For positive velocity, the upwind neighbor is at the `floor` position.
  floor = int(np.floor(offset_delta))
  # For negative velocity, the upwind neighbor is at the `ceil` position.
  ceil = int(np.ceil(offset_delta))
  
  # Use `jnp.where` to select data from the correct neighbor for each point on the grid.
  # This is the core of the upwind scheme.
  # `grids.applied` wraps `jnp.where` to work with GridArrays.
  array_data = grids.applied(jnp.where)(
      u.array > 0,  # Condition: where is the velocity positive?
      c.shift(floor, axis),  # Value if True: take data from the `floor` neighbor.
      c.shift(ceil, axis)    # Value if False: take data from the `ceil` neighbor.
  ).data
  
  # Ensure the grid information is consistent.
  grid = grids.consistent_grid(c, u)
  
  # Return a new GridVariable with the interpolated data. Note that the boundary
  # condition of the interpolated quantity is often not well-defined and is
  # defaulted to periodic here. The consuming function (e.g., `advect`) is
  # responsible for assigning the correct final BC.
  return grids.GridVariable(
      array=grids.GridArray(array_data, offset, grid),
      bc=boundaries.periodic_boundary_conditions(grid.ndim))


def lax_wendroff(
    c: GridVariable,
    offset: Tuple[float, ...],
    v: Optional[GridVariableVector] = None,
    dt: Optional[float] = None
) -> GridVariable:
  """
  Performs second-order Lax-Wendroff interpolation of `c` to a target `offset`.

  This scheme achieves second-order accuracy by adding a corrective term to the
  first-order upwind value. This correction is derived from a Taylor series
  expansion in time and space, which approximates the second-order derivative
  term.

  While more accurate than simple upwinding, as a linear second-order scheme, it
  is not guaranteed to be monotonic (i.e., it can create new, unphysical highs
  and lows, or "wiggles") near sharp gradients or shocks. This is a consequence
  of Godunov's theorem.

  Because of this, the Lax-Wendroff scheme is often not used by itself, but rather
  as the high-order component in more advanced Total Variation Diminishing (TVD)
  schemes that use a flux limiter to suppress these oscillations.

  Args:
    c: The `GridVariable` to be interpolated.
    offset: The target offset to which `c` will be interpolated. Must differ from
      `c.offset` along only one axis.
    v: The velocity `GridVariableVector`, used to determine the upwind direction
      and to calculate the Courant number.
    dt: The size of the time step, `dt`. This is required for the scheme.

  Returns:
    A `GridVariable` that contains the interpolated values of `c` at the target
    `offset`.

  Raises:
    InconsistentOffsetError: if `offset` and `c.offset` differ in more than one
      entry.
  """
  # TODO(dkochkov): The suggestion to add a helper function to compute the
  # interpolation axis is a good one for code clarity and reuse.
  
  # If the source and target offsets are the same, no interpolation is needed.
  if c.offset == offset: return c
  
  # Determine the single axis along which interpolation is occurring.
  interpolation_axes = tuple(
      axis for axis, (current, target) in enumerate(zip(c.offset, offset))
      if not np.isclose(current, target)
  )
  if len(interpolation_axes) != 1:
    raise grids.InconsistentOffsetError(
        f'for Lax-Wendroff interpolation `c.offset` and `offset` must differ at'
        f' most in one entry, but got: {c.offset} and {offset}.')
  axis, = interpolation_axes
  
  # Get the velocity component normal to the target face.
  u = v[axis]
  
  # Determine the relative positions of the upwind/downwind neighbors.
  offset_delta = u.offset[axis] - c.offset[axis]
  floor = int(np.floor(offset_delta))  # Upwind neighbor for positive velocity.
  ceil = int(np.ceil(offset_delta))   # Upwind neighbor for negative velocity.
  
  grid = grids.consistent_grid(c, u)
  # Calculate the Courant number `C = u * dt / dx` for every point on the grid.
  # This is crucial for the Lax-Wendroff correction term.
  courant_numbers = (dt / grid.step[axis]) * u.data
  
  # Calculate the interpolated value assuming the velocity is positive everywhere.
  # The formula is: c_upwind + 0.5 * (1 - |C|) * (c_downwind - c_upwind).
  positive_u_case = (
      c.shift(floor, axis).data + 0.5 * (1 - courant_numbers) *
      (c.shift(ceil, axis).data - c.shift(floor, axis).data))
      
  # Calculate the interpolated value assuming the velocity is negative everywhere.
  # The formula changes slightly due to the direction of upwinding.
  negative_u_case = (
      c.shift(ceil, axis).data - 0.5 * (1 + courant_numbers) * # Note: 1 + C because C is negative
      (c.shift(ceil, axis).data - c.shift(floor, axis).data))
      
  # Use `jnp.where` to select the appropriate formula for each grid point based on the local velocity sign.
  array = grids.where(u.array > 0, positive_u_case, negative_u_case)
  
  grid = grids.consistent_grid(c, u)
  # Return a new GridVariable with the interpolated data.
  return grids.GridVariable(
      array=grids.GridArray(array.data, offset, grid),
      bc=boundaries.periodic_boundary_conditions(grid.ndim))


def safe_div(x: Array, y: Array, default_numerator: float = 1.0) -> Array:
  """
  Performs element-wise division `x / y` safely.

  This is a utility to prevent division-by-zero errors and the propagation of `NaN`
  values. Where the denominator `y` is zero, the result of the division is
  effectively `x / default_numerator`. The choice of `default_numerator=1` means
  the result will be `x` in those cases, which is often a sensible default,
  but in the context of flux limiters, the `x` is also often zero, resulting in zero.

  Args:
    x: The numerator array.
    y: The denominator array.
    default_numerator: The value to use in the denominator where `y` is zero.

  Returns:
    The result of the element-wise division.
  """
  # `jnp.where` is used to create a new denominator that replaces all zeros with
  # the `default_numerator`, making the subsequent division safe.
  return x / jnp.where(y != 0, y, default_numerator)


def van_leer_limiter(r: Array) -> Array:
  """
  Computes the Van Leer flux limiter function.

  Flux limiters are the core of TVD (Total Variation Diminishing) schemes. They
  are designed to be "smart" switches that blend between a stable, low-order
  scheme (like upwind) and an accurate, high-order scheme (like Lax-Wendroff).

  The limiter takes `r`, the ratio of successive gradients in the solution, as
  input.
  - When `r` is large and positive (smooth regions), `phi(r)` approaches 1, using
    more of the high-order scheme.
  - When `r` is near zero or negative (extrema or oscillations), `phi(r)` becomes
    0, reverting to the stable low-order scheme to prevent wiggles.

  Args:
    r: An array representing the ratio of successive gradients, `(c_i - c_{i-1}) / (c_{i+1} - c_i)`.

  Returns:
    An array containing the limiter value `phi(r)` for each grid point.
  """
  # The formula `phi(r) = (r + |r|) / (1 + |r|)` can be simplified.
  # If `r <= 0`, then `r + |r| = 0`, so the result is 0.
  # If `r > 0`, then `r + |r| = 2r` and `1 + |r| = 1 + r`, so the result is `2r / (1+r)`.
  # This `jnp.where` implements that logic efficiently.
  return jnp.where(r > 0, safe_div(2 * r, 1 + r), 0.0)


def apply_tvd_limiter(
    interpolation_fn: InterpolationFn,
    limiter: FluxLimiter = van_leer_limiter
) -> InterpolationFn:
  """
  A meta-function that creates a robust TVD (Total Variation Diminishing) interpolation scheme.

  This function is a "factory" for creating high-resolution, oscillation-free
  interpolation methods. It works by taking a stable but diffusive low-accuracy
  scheme (`upwind`) and a more accurate but potentially unstable high-accuracy
  scheme (e.g., `lax_wendroff`), and blending them together using a `limiter`.

  The final interpolated value is calculated as:
  `c_tvd = c_low + limiter_value * (c_high - c_low)`

  The `limiter_value` (`phi`) is a function of the local smoothness of the data.
  In smooth regions, `phi` is close to 1, recovering the high-accuracy scheme.
  Near sharp gradients or extrema, `phi` goes to 0, reverting to the stable
  low-accuracy scheme to prevent oscillations.

  Args:
    interpolation_fn: The high-order interpolation function (e.g., `lax_wendroff`)
      that will be limited. It must follow the standard `InterpolationFn` API.
    limiter: A flux limiter function (e.g., `van_leer_limiter`) that calculates
      the blending factor based on the ratio of consecutive gradients.

  Returns:
    A new `InterpolationFn` that implements the complete, robust TVD scheme.
  """

  def tvd_interpolation(
      c: GridVariable,
      offset: Tuple[float, ...],
      v: GridVariableVector,
      dt: float,
  ) -> GridVariable:
    """
    The generated TVD interpolation function that performs the blending.
    This inner function is what is returned by the outer factory.
    """
    # This loop structure is designed for multi-dimensional interpolation,
    # applying the 1D TVD scheme sequentially along each axis where the offset changes.
    for axis, axis_offset in enumerate(offset):
      # Define the target offset for this single axis of interpolation.
      interpolation_offset = tuple([
          c_offset if i != axis else axis_offset
          for i, c_offset in enumerate(c.offset)
      ])
      
      # Only perform interpolation if the offset on this axis actually changes.
      if not np.allclose(np.array(interpolation_offset), np.array(c.offset)):
        # This implementation is specifically for interpolating from a cell
        # center to a face (a distance of 0.5 grid units).
        if not np.isclose(interpolation_offset[axis] - c.offset[axis], 0.5):
          raise NotImplementedError('tvd_interpolation only supports forward '
                                    'interpolation to control volume faces.')
                                    
        # Step 1: Compute the interpolated values using both the low- and high-order schemes.
        c_low = upwind(c, interpolation_offset, v, dt)
        c_high = interpolation_fn(c, interpolation_offset, v, dt)

        # Step 2: Compute the ratio of consecutive gradients, `r`, which is the
        # input to the limiter function. This requires a stencil of several points.
        c_left = c.shift(-1, axis)       # Value at i-1
        c_right = c.shift(1, axis)      # Value at i+1
        c_next_right = c.shift(2, axis) # Value at i+2

        # Step 3: The definition of `r` depends on the flow direction (sign of velocity).
        # For positive velocity, `r = (c_i - c_{i-1}) / (c_{i+1} - c_i)`.
        positive_u_r = safe_div(c.data - c_left.data, c_right.data - c.data)
        # For negative velocity, the stencil for the ratio is shifted upstream.
        negative_u_r = safe_div(c_next_right.data - c_right.data,
                                c_right.data - c.data)
                                
        # Step 4: Evaluate the limiter function `phi(r)` for both cases.
        positive_u_phi = grids.GridArray(
            limiter(positive_u_r), c_low.offset, c.grid)
        negative_u_phi = grids.GridArray(
            limiter(negative_u_r), c_low.offset, c.grid)
            
        # Get the velocity component along the interpolation axis.
        u = v[axis]
        
        # Select the correct limiter value `phi` for each grid point based on the velocity sign.
        phi = grids.applied(jnp.where)(
            u.array > 0, positive_u_phi, negative_u_phi)
            
        # Step 5: Compute the final TVD interpolated value by blending the low and high results.
        # NOTE: There appears to be a sign error here. The standard formula is
        # `c_low + phi * (c_high - c_low)`. The code has a minus sign.
        c_interpolated_data = c_low.array.data - (c_low.array.data - c_high.array.data) * phi.data
        
        # Update `c` for the next iteration of the loop (for the next dimension).
        c = grids.GridVariable(
            grids.GridArray(c_interpolated_data, interpolation_offset, c.grid),
            c.bc)
            
    # Return the final interpolated variable after processing all necessary axes.
    return c

  # The factory returns the fully-formed `tvd_interpolation` function.
  return tvd_interpolation


# TODO(pnorgaard): The suggestion to change `c` to a GridVariable is interesting.
# While `.shift()` is not used, a GridVariable's `bc` attribute could potentially
# inform the `mode` parameter automatically (e.g., periodic BC -> mode='wrap').
# However, since this function is a wrapper around a lower-level JAX utility,
# taking a GridArray is a reasonable and simple design choice.
def point_interpolation(
    point: Array,
    c: GridArray,
    order: int = 1,
    mode: str = 'constant',
    cval: float = 0.0,
) -> jax.Array:
  """
  Interpolates the value of a `GridArray` `c` at an arbitrary off-grid point.
  
  This is a general-purpose interpolation utility, useful for post-processing or
  for coupling with Lagrangian methods (like particle tracking) where data is
  needed at specific, non-grid-aligned locations.

  It works by first converting the physical coordinates of the `point` into
  fractional grid indices, and then uses the powerful `map_coordinates` function
  from JAX's Scipy compatibility library to perform the underlying spline
  interpolation.

  Args:
    point: A 1D array of shape `(ndim,)` specifying the physical coordinates
      (e.g., `[x, y]`) of the single point to interpolate to.
    c: The N-dimensional `GridArray` containing the source data.
    order: The order of the spline interpolation. `0` corresponds to nearest-neighbor
      interpolation, and `1` corresponds to multilinear interpolation.
    mode: A string specifying how to handle coordinates that fall outside the
      grid's domain. The options are:
      'reflect': Reflects the data about the boundary edge.
      'constant': Fills with a constant value (`cval`).
      'nearest': Extends with the value of the nearest valid grid point.
      'mirror': Reflects the data about the center of the last grid point.
      'wrap': Wraps the data around from the opposite boundary (periodic).
    cval: The constant value to use if `mode` is 'constant'.

  Returns:
    A scalar JAX array representing the interpolated value at the given `point`.
  """
  # Ensure the input point is a JAX array.
  point = jnp.asarray(point)

  # --- Coordinate Transformation ---
  # This section maps the physical coordinates of the `point` into the
  # fractional index space of the array `c`.

  # Unzip the domain boundaries into separate lower and upper bound arrays.
  domain_lower, domain_upper = zip(*c.grid.domain)
  domain_lower = jnp.array(domain_lower)
  domain_upper = jnp.array(domain_upper)
  # Get the grid properties as arrays for vectorized calculations.
  shape = jnp.array(c.grid.shape)
  offset = jnp.array(c.offset)
  
  # The mapping from a physical `point` to a fractional `index` is linear.
  # We can derive the formula by considering two known points:
  # A point at `domain_lower` should map to an index of `-offset`.
  # A point at `domain_upper` should map to an index of `shape - offset`.
  #
  # The following vectorized formula implements this linear transformation:
  # 1. `(point - domain_lower)`: distance from the lower boundary.
  # 2. `(domain_upper - domain_lower)`: total physical size of the domain.
  # 3. `( ... ) / ( ... )`: fractional position within the domain (from 0 to 1).
  # 4. `* shape`: scales the fractional position to the grid index range.
  # 5. `- offset`: accounts for the staggered grid offset.
  index = (-offset + (point - domain_lower) * shape /
           (domain_upper - domain_lower))

  # --- Perform Interpolation ---
  # `jax.scipy.ndimage.map_coordinates` is a powerful, JAX-compatible function
  # that performs multi-dimensional spline interpolation. It takes the source
  # data and a set of fractional coordinates and returns the interpolated values.
  return jax.scipy.ndimage.map_coordinates(
      c.data,          # The source data array.
      coordinates=index, # The target coordinates in fractional index space.
      order=order,     # The spline order (0 for nearest, 1 for linear).
      mode=mode,       # How to handle out-of-bounds coordinates.
      cval=cval        # The value to use for mode='constant'.
  )
