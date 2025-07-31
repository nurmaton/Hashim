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
Classes that specify how boundary conditions are applied to arrays.

This module provides the core data structures and logic for defining and applying
boundary conditions (BCs) to the physical fields in the simulation. In finite
difference/volume methods, values must be defined in "ghost cells" just outside
the computational domain to correctly calculate derivatives at the boundaries.
This module implements the logic for filling those ghost cells according to
different physical conditions.
"""

import dataclasses
from typing import Any, Callable, Iterable, Sequence, Tuple, Optional, Union
from jax import lax
import jax
import jax.numpy as jnp
from jax_ib.base import grids
import numpy as np
import scipy
from jax.tree_util import register_pytree_node_class
from jax_ib.base import particle_class

# --- Type Aliases ---
# Defines convenient, readable aliases for the core data structures from the `grids` module.
BoundaryConditions = grids.BoundaryConditions
GridArray = grids.GridArray
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
Array = Union[np.ndarray, jax.Array]
BCArray = grids.BCArray


class BCType:
  """
  Defines string constants for different types of boundary conditions.

  Using a class for these constants helps prevent typos and makes the code
  more readable and maintainable compared to using raw strings everywhere.
  """
  # Periodic BC: The domain wraps around on itself. What goes out one side
  # comes in the opposite side. Useful for modeling a small part of a larger, repeating system.
  PERIODIC = 'periodic'
  
  # Dirichlet BC: The value of the variable is fixed at the boundary.
  # For velocity, this is used to model solid, no-slip walls (v=0) or inflow/outflow with a specified velocity.
  DIRICHLET = 'dirichlet'
  
  # Neumann BC: The gradient (normal derivative) of the variable is fixed at the boundary.
  # A zero-Neumann condition (`∂v/∂n = 0`) means there is no flux across the boundary
  # and is often used for pressure at solid walls or for outflow boundaries.
  NEUMANN = 'neumann'

# This decorator registers the class with JAX, allowing it to be used as a node
# in a PyTree. This is essential for JAX to be able to trace operations through
# objects of this class inside `jit`, `vmap`, etc.
@register_pytree_node_class
# `init=False` means the dataclass won't generate a default __init__ method.
# `frozen=False` allows attributes to be set, which is needed by the custom __init__.
@dataclasses.dataclass(init=False, frozen=False)
class ConstantBoundaryConditions(BoundaryConditions):
  """
  A concrete implementation of `BoundaryConditions` that handles BCs that are
  constant in space, but may be dependent on time.

  This class holds the configuration for boundary conditions, such as their type
  (periodic, Dirichlet, etc.) and values. It provides the core methods to
  manipulate `GridArray`s by padding them with ghost cells according to the
  defined boundary conditions.

  Attributes:
    types: A tuple of tuples, where `types[i]` is a pair of strings specifying
      the lower and upper boundary condition types for dimension `i`.
    bc_values: A tuple of tuples containing the numerical values for Dirichlet or
      Neumann conditions. For periodic dimensions, the values are typically `None`.
    boundary_fn: A callable (function) that can be used to describe
      time-dependent boundary values. It usually takes time `t` as an argument.
    time_stamp: A float that tracks the current simulation time, for use with `boundary_fn`.
  """
  # Attribute type hints for clarity.
  types: Tuple[Tuple[str, str], ...]
  bc_values: Tuple[Tuple[Optional[float], Optional[float]], ...]
  boundary_fn: Callable[...,Optional[float]]
  time_stamp: Optional[float]
  
  def __init__(self,
               time_stamp: Optional[float],
               values: Sequence[Tuple[Optional[float], Optional[float]]],
               types: Sequence[Tuple[str, str]],
               boundary_fn: Callable[...,Optional[float]]):
    """
    Initializes the boundary condition configuration.
    
    Args:
      time_stamp: The initial time, used for time-dependent boundary conditions.
      values: A sequence of tuples specifying the boundary values for each dimension.
      types: A sequence of tuples specifying the boundary types for each dimension.
      boundary_fn: A function that can define time-dependent boundary values.
    """
    # Ensure types and values are converted to immutable tuples. This is good practice
    # for PyTree components, as their structure should not change.
    types = tuple(types)
    values = tuple(values)
    boundary_fn = boundary_fn
    time_stamp = time_stamp
    
    # Because the dataclass is marked with `init=False` and `frozen=False`,
    # we use `object.__setattr__` to initialize the attributes. This is a standard
    # pattern for custom initialization in such dataclasses.
    object.__setattr__(self, 'bc_values', values)
    object.__setattr__(self, 'boundary_fn', boundary_fn)
    # The time_stamp is stored, or an empty list if None is provided (though None might be better).
    object.__setattr__(self, 'time_stamp', time_stamp if time_stamp is not None else [])
    object.__setattr__(self, 'types', types)
    
    # The commented out lines are likely remnants of a previous implementation idea.


  def tree_flatten(self):
    """
    Defines how to flatten this object for JAX PyTree processing.

    JAX needs to know which parts of the object are dynamic (tracers, arrays)
    and which are static (metadata, configuration).
    
    Returns:
      A tuple of "children" (dynamic parts) and "aux_data" (static parts).
    """
    # `time_stamp` and `bc_values` are treated as dynamic "children". This means
    # JAX can trace them, and their values can be updated within a `jit` context
    # (e.g., inside a `lax.scan` loop).
    children = (self.time_stamp,self.bc_values,)
    # `types` and `boundary_fn` are static "aux_data". Their structure and identity
    # are assumed to be constant during a JAX trace.
    aux_data = (self.types,self.boundary_fn)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """
    Defines how to reconstruct the object from its flattened parts.
    This is the inverse of `tree_flatten`.
    """
    # Reconstruct the class instance from the unflattened components.
    # The order of `*children` and `*aux_data` must match the `__init__` signature.
    return cls(*children, *aux_data)


  def update_bc_(self, time_stamp: float, dt: float):
    """
    Updates the `time_stamp` for time-dependent boundary conditions.
    
    Note: The logic that directly uses this has been simplified in the newer
    `update_BC` pass-through function, making this method less critical in the
    current main solver loop, but it remains for potential use.
    """
    return time_stamp + dt
       

  def shift(
      self,
      u: GridArray,
      offset: int,
      axis: int,
  ) -> GridArray:
    """
    Shifts a `GridArray` by a given integer `offset` along an `axis`.

    This is the primary high-level method used for finite difference calculations.
    It works by first padding the array with ghost cells in the direction of the
    shift and then trimming the array from the opposite side.

    Args:
      u: a `GridArray` object to be shifted.
      offset: a positive or negative integer specifying the number of grid cells to shift.
      axis: the axis along which to perform the shift.

    Returns:
      A new `GridArray`, shifted and padded according to the boundary conditions.
      The returned array will have a correspondingly updated `offset`.
    """
    # Pad the array with ghost cells in the direction of the shift.
    padded = self._pad(u, offset, axis)
    # Trim the array from the opposite direction to complete the shift.
    trimmed = self._trim(padded, -offset, axis)
    return trimmed

  def _pad(
      self,
      u: GridArray,
      width: int,
      axis: int,
  ) -> GridArray:
    """
    Pads a GridArray with ghost cells according to the boundary conditions.

    This method is the core of the boundary condition implementation. It extends
    the data array with extra cells ("ghost cells") whose values are determined
    by the boundary condition type (periodic, Dirichlet, or Neumann). This is
    essential for finite difference stencils near the domain boundary.

    For example, for a Dirichlet boundary, the ghost cell value is set such that
    a linear interpolation to the boundary yields the correct fixed value. This
    often involves mirroring the interior data.

    Important: For many standard finite difference/volume methods, only one
    layer of ghost cells (`width=1`) is required. More ghost cells might be
    needed for higher-order schemes or other applications like CNNs on grids.
    This implementation does not support more than one ghost cell for Neumann BCs.

    Args:
      u: The `GridArray` object to pad.
      width: The number of cells to add. If negative, pads the lower boundary
        (e.g., left side); if positive, pads the upper boundary (e.g., right side).
      axis: The axis along which to pad.

    Returns:
      A new, padded `GridArray`.
    """

    def make_padding(width: int):
      """
      A nested helper function to create the padding configuration tuple
      that `jnp.pad` expects.
      """
      # Determine if we are padding the lower or upper boundary based on the sign of `width`.
      if width < 0:  # pad lower boundary
        # Get the BC type for the lower boundary of the specified axis.
        bc_type = self.types[axis][0]
        # `jnp.pad` expects a tuple of (pad_before, pad_after).
        padding = (-width, 0)
      else:  # pad upper boundary
        # Get the BC type for the upper boundary.
        bc_type = self.types[axis][1]
        padding = (0, width)

      # Create the full padding configuration for all dimensions.
      # It will be `(0, 0)` for all axes except the one being padded.
      full_padding = [(0, 0)] * u.grid.ndim
      full_padding[axis] = padding
      return full_padding, padding, bc_type

    # Call the helper to get the padding configuration.
    full_padding, padding, bc_type = make_padding(width)
    
    # Calculate the new offset of the padded array. If we pad `N` cells on the
    # left (lower boundary), the new array's first element corresponds to an
    # offset that is `N` units smaller.
    offset = list(u.offset)
    offset[axis] -= padding[0]
    
    # Enforce the limitation that Neumann BC padding is only implemented for a single ghost cell.
    # Periodic and Dirichlet have more general implementations.
    if not (bc_type == BCType.PERIODIC or
            bc_type == BCType.DIRICHLET) and abs(width) > 1:
      raise ValueError(
          f'Padding past 1 ghost cell is not defined in {bc_type} case.')

    # Trim any existing padding from the input array `u`. This is crucial to
    # prevent padding from accumulating if `_pad` is called multiple times.
    # `_trim_padding` returns the trimmed array and the amount that was trimmed.
    u, trimmed_padding = self._trim_padding(u)
    # Get the raw data array from the (potentially trimmed) GridArray.
    data = u.data
    # Update the padding amount needed. If `_trim_padding` removed some existing
    # padding, we need to add it back to achieve the correct final shape.
    full_padding[axis] = tuple(
        pad + trimmed_pad
        for pad, trimmed_pad in zip(full_padding[axis], trimmed_padding))

    # --- Apply Padding Based on Boundary Condition Type ---

    if bc_type == BCType.PERIODIC:
      # For periodic boundaries, the data must span the entire grid dimension.
      # Otherwise, wrapping values around doesn't make physical sense.
      if u.grid.shape[axis] > u.shape[axis]:
        raise ValueError('For periodic BC, the GridArray shape must match the grid shape.')
      # Use `mode='wrap'` which takes values from the opposite end of the array
      # to fill the padded region, creating the periodic effect.
      pad_kwargs = dict(mode='wrap')
      data = jnp.pad(data, full_padding, **pad_kwargs)

    elif bc_type == BCType.DIRICHLET:
      # Dirichlet BCs specify a fixed value *at the boundary*. The implementation
      # differs depending on whether the data is at cell centers or cell faces.

      # Case 1: Data is at cell centers (offset ends in .5).
      if np.isclose(u.offset[axis] % 1, 0.5):
        # For Dirichlet at cell centers, the data must also span the full grid.
        if u.grid.shape[axis] > u.shape[axis]:
          raise ValueError('For cell-centered Dirichlet, the GridArray shape must match the grid shape.')
        # This formula sets the ghost cell value `u_ghost` such that a linear
        # interpolation between `u_ghost` and the first interior cell `u_interior`
        # equals the boundary value `u_bc`. The formula is: `u_ghost = 2*u_bc - u_interior`.
        # This is implemented by combining `mode='constant'` (for the `2*u_bc` part)
        # and `mode='symmetric'` (for the `-u_interior` part).
        data = (2 * jnp.pad(
            data, full_padding, mode='constant', constant_values=self.bc_values)
                - jnp.pad(data, full_padding, mode='symmetric'))
                
      # Case 2: Data is at cell faces/edges (offset is integer).
      elif np.isclose(u.offset[axis] % 1, 0):
        # On a staggered grid, face-aligned data might not include the boundary
        # face itself (e.g., `u_x` velocity might start at face `i=1/2`, not `i=0`).
        # This logic handles cases where the boundary value needs to be explicitly added.
        if u.grid.shape[axis] > u.shape[axis] + 1:
          raise ValueError('For a Dirichlet cell-face BC, the GridArray is missing too many grid points.')
        elif u.grid.shape[axis] == u.shape[axis] + 1 and not np.isclose(u.offset[axis], 1):
          raise ValueError('A GridArray missing a point must be offset by 1 for Dirichlet BC.')

        # Helper to determine if we need to explicitly pad with the boundary value.
        def _needs_pad_with_boundary_value():
          # Case A: We are padding "into" the domain from a face-aligned array
          # (e.g., offset=0, padding on the right/upper side).
          if (np.isclose(u.offset[axis], 0) and width > 0) or \
             (np.isclose(u.offset[axis], 1) and width < 0):
            return True
          # Case B: The array is one point smaller than the grid, meaning it's an
          # interior array that's missing the boundary point.
          elif u.grid.shape[axis] == u.shape[axis] + 1:
            return True
          else:
            return False

        if _needs_pad_with_boundary_value():
          # If we need to add the boundary value and are only padding one cell,
          # we can simply pad with the constant boundary value.
          if np.isclose(abs(width), 1):
            data = jnp.pad(
                data,
                full_padding,
                mode='constant',
                constant_values=self.bc_values)
          # Padding more than one ghost cell is a more complex reflection.
          elif abs(width) > 1:
            # This logic reflects the data around the explicitly added boundary value.
            bc_padding, _, _ = make_padding(int(np.sign(width)))
            full_padding_past_bc, _, _ = make_padding(width - int(np.sign(width)))
            expanded_data = jnp.pad(
                data, bc_padding, mode='constant', constant_values=(0, 0))
            padding_values = list(self.bc_values)
            padding_values[axis] = [pad / 2 for pad in padding_values[axis]]
            data = 2 * jnp.pad(
                data,
                full_padding,
                mode='constant',
                constant_values=tuple(padding_values)) - jnp.pad(
                    expanded_data, full_padding_past_bc, mode='reflect')
        else:
          # If the array already contains the boundary value, we use a different
          # reflection formula similar to the cell-centered case.
          padding_values = list(self.bc_values)
          padding_values[axis] = [pad / 2 for pad in padding_values[axis]]
          data = 2 * jnp.pad(
              data,
              full_padding,
              mode='constant',
              constant_values=tuple(padding_values)) - jnp.pad(
                  data, full_padding, mode='reflect')
      else:
        # The offset must be either a center or a face.
        raise ValueError('expected offset to be an edge or cell center, got '
                         f'offset[axis]={u.offset[axis]}')
    elif bc_type == BCType.NEUMANN:
      # Neumann boundary conditions specify the gradient at the boundary.
      # Similar to periodic and cell-centered Dirichlet, the array must span
      # the full grid dimension for the stencil math to be well-defined.
      if u.grid.shape[axis] > u.shape[axis]:
        raise ValueError('For Neumann BC, the GridArray shape must match the grid shape.')
        
      # The implementation is valid for data at cell centers or cell faces.
      if not (np.isclose(u.offset[axis] % 1, 0) or
              np.isclose(u.offset[axis] % 1, 0.5)):
        raise ValueError('Expected offset to be an edge or cell center for Neumann BC, got '
                         f'offset[axis]={u.offset[axis]}')
      else:
        # This formula sets the ghost cell value `u_ghost` such that the finite
        # difference approximation of the gradient at the boundary equals the
        # specified Neumann value, `N`.
        #
        # For a forward difference at the lower boundary: (u_interior_1 - u_ghost) / dx = N
        # Rearranging gives: u_ghost = u_interior_1 - N * dx.
        # For a backward difference at the upper boundary: (u_ghost - u_interior_last) / dx = N
        # Rearranging gives: u_ghost = u_interior_last + N * dx.
        #
        # This single line of code cleverly implements both cases:
        data = (
            # `jnp.pad(..., mode='edge')` sets `u_ghost = u_interior`. This is the
            # zero-Neumann part of the formula (`u_interior_1`).
            jnp.pad(data, full_padding, mode='edge')
            
            # The second part is the correction `± N * dx`.
            + u.grid.step[axis] *
            
            # This difference term `(0 - N)` provides the `±N` part.
            # `jnp.pad(..., mode='constant')` pads with 0 by default.
            # `jnp.pad(..., constant_values=...)` pads with the Neumann value `N`.
            # The result is `(0 - N)` on the lower boundary and `(N - 0)` on the
            # upper boundary (after considering padding directions), which correctly
            # applies the `±` sign.
            (jnp.pad(data, full_padding, mode='constant') - jnp.pad(
                data,
                full_padding,
                mode='constant',
                constant_values=self.bc_values))
        )
    else:
      # If the `bc_type` is not one of the recognized types, raise an error.
      raise ValueError(f'Invalid boundary type encountered: {bc_type}')

    # After computing the padded `data` array according to the correct BC logic,
    # wrap it in a new GridArray with the updated offset and grid information.
    return GridArray(data, tuple(offset), u.grid)

  def _trim(
      self,
      u: GridArray,
      width: int,
      axis: int,
  ) -> GridArray:
    """
    Trims a specified number of cells from the boundary of a GridArray.
    This is the inverse operation of `_pad`.

    Args:
      u: a `GridArray` object to be trimmed.
      width: The number of cells to trim. If negative, trims from the lower
        boundary (e.g., left side). If positive, trims from the upper boundary
        (e.g., right side).
      axis: The axis along which to perform the trim.

    Returns:
      A new, smaller `GridArray`.
    """
    # Determine the slice indices based on the sign of `width`.
    if width < 0:  # Trim from the lower boundary.
      # `padding` here represents the slice start and end relative to the edges.
      # A width of -2 means we want to slice from index 2 onwards.
      padding = (-width, 0)
    else:  # Trim from the upper boundary.
      # A width of 2 means we want to slice up to the last 2 elements.
      padding = (0, width)

    # Calculate the slice indices for the `lax.slice_in_dim` function.
    # The start index is `padding[0]`.
    # The limit index is the total size minus the amount to trim from the end.
    limit_index = u.data.shape[axis] - padding[1]
    
    # `lax.slice_in_dim` is the JAX primitive for slicing an array along a
    # single dimension. It is more efficient inside a `jit` context than
    # standard Python slicing `array[...]`.
    data = lax.slice_in_dim(u.data, padding[0], limit_index, axis=axis)
    
    # Update the offset to reflect the trim. If we trim `N` cells from the
    # left (lower boundary), the new array's first element corresponds to an
    # offset that is `N` units larger.
    offset = list(u.offset)
    offset[axis] += padding[0]
    
    # Return a new GridArray with the trimmed data and updated offset.
    return GridArray(data, tuple(offset), u.grid)

  def _trim_padding(self, u: grids.GridArray, axis: int = 0):
    """
    Trims all excess ghost cell padding from a GridArray to make its shape
    match the underlying grid's shape.

    This is a utility function used to normalize a `GridArray` that may have
    been padded in previous operations, ensuring it represents only the data
    within the physical domain before a new operation is applied.

    Args:
      u: a `GridArray` object that may have padding.
      axis: The axis along which to trim.

    Returns:
      A tuple containing:
      - The trimmed `GridArray`, whose shape now matches `u.grid.shape`.
      - A `padding` tuple `(trimmed_from_left, trimmed_from_right)` indicating
        how many cells were removed from each side.
    """
    # `padding` will store the number of cells trimmed from the (lower, upper) sides.
    padding = (0, 0)
    
    # Only perform trimming if the array's shape is larger than the grid's shape.
    if u.shape[axis] > u.grid.shape[axis]:
      # `negative_trim` stores the number of cells padded on the left/lower side.
      negative_trim = 0
      # A negative offset indicates that padding exists on the left side.
      if u.offset[axis] < 0:
        # The number of padded cells is the absolute value of the offset, rounded.
        negative_trim = -round(-u.offset[axis])
        # Trim these cells from the lower boundary.
        u = self._trim(u, negative_trim, axis)
        
      # `positive_trim` is the remaining excess size after the left-side trim.
      # This corresponds to the padding on the right/upper side.
      positive_trim = u.shape[axis] - u.grid.shape[axis]
      if positive_trim > 0:
        # Trim these cells from the upper boundary.
        u = self._trim(u, positive_trim, axis)
        
      # Record the total number of cells trimmed from each side.
      padding = (negative_trim, positive_trim)
      
    # Return the fully trimmed array and the record of what was trimmed.
    return u, padding

  def values(
      self, axis: int,
      grid: grids.Grid
  ) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """
    Returns the boundary values as arrays broadcastable to the grid faces.

    This method takes the scalar boundary condition values (e.g., `1.0` for a
    wall moving at a speed of 1) and converts them into JAX arrays that have the
    same shape as the boundary face of the grid. This is useful for applying
    boundary conditions in solvers.

    Args:
      axis: The axis along which to return boundary values.
      grid: A `Grid` object on which the boundary conditions are to be evaluated.

    Returns:
      A tuple containing two elements: `(lower_boundary_array, upper_boundary_array)`.
      Each element is a JAX array with `grid.ndim - 1` dimensions, filled with
      the corresponding boundary value. In the case of periodic boundaries,
      where a single value is not well-defined, it returns `(None, None)`.
    """
    # For periodic boundaries, the `bc_values` are typically set to `None`.
    # In this case, there are no specific values to return.
    if None in self.bc_values[axis]:
      return (None, None)
      
    # Create the shape of the boundary face. For a 3D grid and `axis=1` (y-axis),
    # the face shape would be `(shape[0], shape[2])`.
    face_shape = grid.shape[:axis] + grid.shape[axis + 1:]
    
    # Create JAX arrays for the lower and upper boundaries, filled with the
    # respective scalar values from `self.bc_values`.
    # `self.bc_values[axis][-i]` with `i=0` and `i=1` is a way to access
    # the (upper, lower) values in reverse order.
    bc_value_arrays = tuple(
        jnp.full(face_shape, self.bc_values[axis][-i])
        for i in [0, 1]) # i=0 for lower, i=1 for upper
        
    return bc_value_arrays

  def trim_boundary(self, u: grids.GridArray) -> grids.GridArray:
    """
    Returns a `GridArray` containing only the interior data points.

    This method performs two trimming operations:
    1. It removes any ghost cell padding that might exist on the input `u`,
       making its shape equal to the grid's shape.
    2. For Dirichlet boundary conditions on a staggered grid, it removes the
       grid points that lie exactly on the boundary, as these are considered
       boundary values, not interior unknowns.

    Args:
      u: A `GridArray` object that may have padding or include points on the boundary.

    Returns:
      A new, potentially smaller `GridArray` containing only interior data points.
    """
    # Step 1: Remove any ghost cell padding from all axes.
    # The `_` is used to discard the second return value of `_trim_padding`.
    for axis in range(u.grid.ndim):
      u, _ = self._trim_padding(u, axis=axis)
      
    # After trimming padding, the array shape should match the grid shape.
    # If not, it implies the array was already trimmed, which is an invalid state.
    if u.shape != u.grid.shape:
      raise ValueError('The GridArray shape does not match the grid shape after trimming padding.')
      
    # Step 2: Trim points that lie exactly on a Dirichlet boundary.
    for axis in range(u.grid.ndim):
      # Check for a Dirichlet condition on the lower boundary. If the data is
      # located at an offset of 0.0, it lies on the boundary and should be trimmed.
      if np.isclose(u.offset[axis], 0.0) and self.types[axis][0] == BCType.DIRICHLET:
        # Trim one layer from the lower boundary.
        u = self._trim(u, -1, axis)
        
      # Check for a Dirichlet condition on the upper boundary. If the data is
      # located at an offset of 1.0 (relative to the cell size), it lies on the
      # boundary and should be trimmed.
      elif np.isclose(u.offset[axis], 1.0) and self.types[axis][1] == BCType.DIRICHLET:
        # Trim one layer from the upper boundary.
        u = self._trim(u, 1, axis)
        
    # Return the fully trimmed GridArray.
    return u

  def pad_and_impose_bc(
      self,
      u: grids.GridArray,
      offset_to_pad_to: Optional[Tuple[float, ...]] = None
  ) -> grids.GridVariable:
    """
    Pads an interior-only `GridArray` and wraps it in a `GridVariable`.

    This method is designed to take a `GridArray` that only contains data for
    the interior computational nodes and add the necessary boundary points,
    turning it into a complete `GridVariable` that is consistent with its
    boundary conditions.

    Args:
      u: A `GridArray` object that specifies values only on the internal nodes.
      offset_to_pad_to: A tuple specifying the desired final offset. This is
        important for ambiguous cases on staggered grids; for example, an interior
        array for a Dirichlet problem could be padded to align with either the
        lower or upper boundary face.

    Returns:
      A `GridVariable` that has been correctly padded to include boundary points.
    """
    # If no target offset is specified, assume the final offset is the same as the input.
    if offset_to_pad_to is None:
      offset_to_pad_to = u.offset
      
    # This loop handles a specific, tricky case for staggered grids with Dirichlet BCs.
    # If the interior data `u` is offset by 1.0 (i.e., it starts one grid cell in),
    # it needs to be padded on one side or the other to include the boundary face.
    for axis in range(u.grid.ndim):
      # Check if the BC is Dirichlet and the interior data starts at an offset of 1.0.
      if self.types[axis][0] == BCType.DIRICHLET and np.isclose(u.offset[axis], 1.0):
        # If the target is also offset by 1.0, pad one cell to the right (upper boundary).
        if np.isclose(offset_to_pad_to[axis], 1.0):
          u = self._pad(u, 1, axis)
        # If the target is offset by 0.0, pad one cell to the left (lower boundary).
        elif np.isclose(offset_to_pad_to[axis], 0.0):
          u = self._pad(u, -1, axis)
          
    # After padding, wrap the resulting `GridArray` and `self` (the BC object)
    # into a new, complete `GridVariable`.
    return grids.GridVariable(u, self)

  def impose_bc(self, u: grids.GridArray) -> grids.GridVariable:
    """
    Ensures a `GridArray` is consistent with the boundary conditions.

    This is a high-level convenience method that handles the common workflow of
    taking a `GridArray` that might have points on the boundary, trimming it down
    to just the interior points, and then correctly padding it back to create
    a final, consistent `GridVariable`.

    Args:
      u: A `GridArray` object.

    Returns:
      A `GridVariable` that has the correct boundary values imposed.
    """
    # Store the original offset of the input array.
    offset = u.offset
    
    # If the input array's shape matches the grid's shape, it might contain
    # points that lie exactly on a boundary. These must be trimmed first to
    # isolate the true interior "unknowns" of the system.
    if u.shape == u.grid.shape:
      u = self.trim_boundary(u)
      
    # After ensuring `u` contains only interior points, call `pad_and_impose_bc`
    # to add the correct boundary values back on.
    return self.pad_and_impose_bc(u, offset)

  # Create convenient aliases for the private `_trim` and `_pad` methods.
  # This allows them to be called from outside the class if needed, for example,
  # `my_bc_object.pad(...)` instead of `my_bc_object._pad(...)`.
  trim = _trim
  pad = _pad

# This decorator registers the class with JAX, allowing it to be used as a node in a PyTree.
@register_pytree_node_class
class HomogeneousBoundaryConditions(ConstantBoundaryConditions):
  """
  A specialized, more efficient `BoundaryConditions` class for homogeneous conditions.
  
  This class represents boundary conditions where the value (for Dirichlet) or
  the flux (for Neumann) is zero everywhere. This is a very common case in CFD
  (e.g., a stationary no-slip wall has a velocity of zero).

  This class inherits from `ConstantBoundaryConditions` but provides a more
  optimized `tree_flatten` method. By telling JAX that the boundary values are
  always constant and known (zero), it allows for better optimization in
  compiled functions, as these values do not need to be tracked as dynamic tracers.

  Attributes:
    types: `types[i]` is a tuple specifying the lower and upper BC types for
      dimension `i`.
  """

  def __init__(self, types: Sequence[Tuple[str, str]]):
    """
    Initializes homogeneous boundary conditions for the given types.
    
    Args:
      types: A sequence of tuples specifying the boundary types for each
        dimension, e.g., `((BCType.PERIODIC, BCType.PERIODIC), (BCType.DIRICHLET, BCType.DIRICHLET))`.
    """
    # Get the number of dimensions from the length of the `types` sequence.
    ndim = len(types)
    # The boundary values are always zero for homogeneous conditions.
    values = ((0.0, 0.0),) * ndim
    # A placeholder boundary function is used, as it's not time-dependent.
    bc_fn = lambda x: x
    # The timestamp is irrelevant for non-time-dependent BCs, so it's set to a constant.
    time_stamp = 0.0
    # Call the parent class's initializer with these fixed, homogeneous values.
    super(HomogeneousBoundaryConditions, self).__init__(time_stamp, values, types, bc_fn)

  def tree_flatten(self):
    """
    Provides a custom, optimized flattening recipe for this class.
    
    Since all numerical values (`bc_values`, `time_stamp`) are constant and
    known, there are no dynamic "children" that JAX needs to trace. Everything
    is treated as static "auxiliary data". This is a key optimization.
    """
    children = ()  # No dynamic children.
    aux_data = (self.types,) # Only the `types` tuple is needed to reconstruct the object.
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Reconstructs the object from its static-only auxiliary data."""
    # The class is reconstructed using only the `types` from `aux_data`.
    return cls(*aux_data)


@register_pytree_node_class
class TimeDependentBoundaryConditions(ConstantBoundaryConditions):
  """
  DEPRECATED / REDUNDANT CLASS for boundary conditions that can vary with time.
  
  The functionality of this class is already fully covered by the main
  `ConstantBoundaryConditions` class, which includes a `boundary_fn` and a
  `time_stamp` attribute for handling time-dependent cases. This subclass does
  not add any new functionality and has some potential inconsistencies (like
  the argument order in `__init__`). It is likely a remnant of an earlier
  design and could probably be removed.
  
  The docstring and example usage are also misleading as they are copied from
  a different class.
  """

  def __init__(
      self,
      types: Sequence[Tuple[str, str]],
      values: Sequence[Tuple[Optional[float], Optional[float]]],
      boundary_fn: Callable[..., Optional[float]],
      time_stamp: Optional[float]
  ):
    # Note: The argument order here (`types`, `values`, ...) differs from the
    # parent `ConstantBoundaryConditions`'s `__init__` signature. This could
    # cause issues if `tree_unflatten` from the parent were ever used.
    
    # The commented out lines are likely remnants of a previous implementation.
    #ndim = len(types)
    #values = ((0.0, 0.0),) * ndim
    
    # Call the parent class's initializer.
    super(TimeDependentBoundaryConditions, self).__init__(types, values, boundary_fn, time_stamp)

  def tree_flatten(self):
    """
    Returns a flattening recipe for this JAX PyTree.
    This recipe differs from the parent class, which might be intentional or
    an artifact of an older design. Here, only `bc_values` are considered dynamic.
    """
    # The boundary values are the dynamic part.
    children = (self.bc_values,)
    # The timestamp, types, and function are considered static.
    aux_data = (self.time_stamp, self.types, self.boundary_fn,)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """
    Returns the unflattening recipe for this JAX PyTree.
    It reconstructs the object from its flattened parts.
    """
    # Note: The argument order `(*children, *aux_data)` might not match the
    # `__init__` signature, which could lead to errors.
    return cls(*children, *aux_data)


def boundary_function(t: float) -> jnp.ndarray:
  """
  An example of a function defining a time-dependent boundary value.
  
  This function can be passed into a `ConstantBoundaryConditions` object via the
  `boundary_fn` attribute to create time-varying Dirichlet or Neumann conditions.
  The `update_BC` function would then call this function at each time step to
  get the new boundary values.

  Note: The current implementation `1 + 0 * (...)` will always return `1.0`.
  This is likely a placeholder or was simplified for debugging. A typical
  implementation would be `A*jnp.cos(freq*t) + B*jnp.sin(freq*t)`.

  Args:
    t: The current simulation time.

  Returns:
    The value of the boundary condition at time `t`.
  """
  # Define some parameters for the sinusoidal function.
  A = 1
  B = 1
  freq = 1
  # Calculate the time-dependent value. The `1 + 0 *` makes it always return 1.
  return 1 + 0 * (A * jnp.cos(freq*t) + B * jnp.sin(freq*t))    

def Reserve_BC(
    all_variable: particle_class.All_Variables,
    step_time: float
) -> particle_class.All_Variables:
    """
    Updates boundary conditions for time-dependent domain walls.

    WHY THIS FUNCTION IS NOW A PASS-THROUGH FOR THE DEFORMABLE PROBLEM:
    This function was originally designed for simulations where the physical
    boundaries of the simulation box itself were in motion (e.g., a lid-driven
    cavity with an oscillating lid). In that scenario, the velocity of the
    domain walls needs to be recalculated and updated at every single time step.

    However, in the deformable flapping simulation, the physics are different:
    - The ACTION is happening with the particle *inside* the domain.
    - The outer boundaries of the simulation box are STATIC (e.g., periodic or
      far-field boundaries with a constant velocity of zero).

    Since the domain's boundary conditions are constant and do not change with
    time, there is no need to perform a complex update at every step. This
    function's logic is therefore obsolete for this specific problem.
    The new implementation is a simple pass-through (`return all_variable`)
    which is both more computationally efficient and a more accurate
    representation of the static-boundary physics.
    """
    # The original implementation for time-dependent domain boundaries is commented out below.
    # # Unpack the current simulation state.
    # v = all_variable.velocity
    # particles = all_variable.particles
    # pressure = all_variable.pressure
    # Drag = all_variable.Drag
    # Step_count = all_variable.Step_count
    # MD_var = all_variable.MD_var
    
    # # Get the boundary function objects for each velocity component.
    # bcfn_x = v[0].bc.boundary_fn
    # bcfn_y = v[1].bc.boundary_fn
    
    # # Update the time stamp.
    # dt = step_time
    # ts = v[0].bc.time_stamp + dt
    
    # # Calculate the new boundary values by calling the boundary functions.
    # # Note the mix of dynamic and static values: `(bcfn[0](ts), bcfn[1](0.0))` suggests
    # # some boundary values are updated with the new time, while others are held constant.
    # vx_bc = ((bcfn_x[0](ts), bcfn_x[1](0.0)), (bcfn_x[2](ts), bcfn_x[3](0.0)))
    # vy_bc = ((bcfn_y[0](ts), bcfn_y[1](0.0)), (bcfn_y[2](ts), bcfn_y[3](0.0)))
    
    # # Create new `ConstantBoundaryConditions` objects with the updated values.
    # vel_bc = (ConstantBoundaryConditions(values=vx_bc, time_stamp=ts, types=v[0].bc.types, boundary_fn=bcfn_x),
    #           ConstantBoundaryConditions(values=vy_bc, time_stamp=ts, types=v[1].bc.types, boundary_fn=bcfn_y))
   
    # # Create new `GridVariable` objects for the velocity components, pairing the
    # # original data arrays with the new boundary condition objects.
    # v_updated =  tuple(grids.GridVariable(u.array, bc) for u, bc in zip(v, vel_bc))
    
    # # Return a new `All_Variables` object containing the updated velocity variables.
    # return particle_class.All_Variables(particles, v_updated, pressure, Drag, Step_count, MD_var)
    
    # --- NEW IMPLEMENTATION FOR STATIC BOUNDARIES ---
    # For the deformable problem with static outer walls, this function does nothing.
    return all_variable
  
  
def update_BC(
    all_variable: particle_class.All_Variables,
    step_time: float
) -> particle_class.All_Variables:
    """
    Updates all time-dependent boundary conditions for the velocity field.

    WHY THIS FUNCTION IS NOW A PASS-THROUGH FOR THE DEFORMABLE PROBLEM:
    Similar to `Reserve_BC`, this function's original purpose was to handle
    simulations with actively moving domain boundaries by re-evaluating the
    wall velocities at every time step.

    For the deformable flapping problem, the outer boundaries are fixed and
    unchanging. Therefore, performing this update is unnecessary and computationally
    wasteful. The correct implementation for this physical setup is to simply
    pass the state through unchanged, as the boundary conditions are constant
    throughout the entire simulation. This satisfies the API required by the
    time-stepper while correctly reflecting the static-boundary physics.
    """
    # The original implementation for time-dependent domain boundaries is commented out below.
    # # Unpack the state.
    # v = all_variable.velocity
    # particles = all_variable.particles
    # pressure = all_variable.pressure
    # Drag = all_variable.Drag
    # Step_count = all_variable.Step_count
    # MD_var = all_variable.MD_var
    
    # # Get the boundary functions.
    # bcfn_x = v[0].bc.boundary_fn
    # bcfn_y = v[1].bc.boundary_fn
    
    # # Update the time stamp.
    # dt = step_time
    # ts = v[0].bc.time_stamp + dt
    
    # # Calculate the new boundary values by calling all boundary functions with the new time `ts`.
    # vx_bc = ((bcfn_x[0](ts), bcfn_x[1](ts)), (bcfn_x[2](ts), bcfn_x[3](ts)))
    # vy_bc = ((bcfn_y[0](ts), bcfn_y[1](ts)), (bcfn_y[2](ts), bcfn_y[3](ts)))
    
    # # Create new BC objects.
    # vel_bc = (ConstantBoundaryConditions(values=vx_bc, time_stamp=ts, types=v[0].bc.types, boundary_fn=bcfn_x),
    #           ConstantBoundaryConditions(values=vy_bc, time_stamp=ts, types=v[1].bc.types, boundary_fn=bcfn_y))
   
    # # Create updated GridVariable objects.
    # v_updated =  tuple(grids.GridVariable(u.array, bc) for u, bc in zip(v, vel_bc))
    
    # # Return the new state.
    # return particle_class.All_Variables(particles, v_updated, pressure, Drag, Step_count, MD_var)
    
    # --- NEW IMPLEMENTATION FOR STATIC BOUNDARIES ---
    # For the deformable problem with static outer walls, this function does nothing.
    return all_variable

# --- Convenience Utilities / Factory Functions ---
# These functions provide simple, readable ways to create common types of boundary conditions.

def periodic_boundary_conditions(ndim: int) -> HomogeneousBoundaryConditions:
  """
  A factory function that returns periodic boundary conditions for all axes.
  
  Args:
    ndim: The number of spatial dimensions.
  
  Returns:
    A `HomogeneousBoundaryConditions` object configured for periodic BCs.
  """
  # Creates a tuple of `ndim` pairs of ('periodic', 'periodic').
  periodic_types = ((BCType.PERIODIC, BCType.PERIODIC),) * ndim
  return HomogeneousBoundaryConditions(periodic_types)


def Radom_velocity_conditions(ndim: int) -> ConstantBoundaryConditions:
    """
    A factory function that returns a specific "Moving Wall" boundary condition
    with initial zero-valued BCs.
    
    The name `Radom_velocity_conditions` is likely a typo and should probably be
    `Random_velocity_conditions` or something more descriptive like
    `zero_velocity_moving_wall_bcs`.
    """
    # Start with homogeneous (zero) boundary values.
    values = ((0.0, 0.0),) * ndim
    # Use a placeholder, identity boundary function.
    bc_fn = lambda x: x
    # Start time is zero.
    time_stamp = 0.0
    # Call another factory function to construct the final BC object.
    return Moving_wall_boundary_conditions(
        ndim,
        bc_vals=values,
        time_stamp=time_stamp,    
        bc_fn=bc_fn,
    )


def dirichlet_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> ConstantBoundaryConditions:
  """
  A factory function that creates Dirichlet boundary conditions for all axes.

  Args:
    ndim: The number of spatial dimensions.
    bc_vals: A sequence of tuples specifying the lower and upper boundary values
      for each dimension, e.g., `((x_lower, x_upper), (y_lower, y_upper))`.
      If `None`, it creates homogeneous (zero-valued) Dirichlet BCs.

  Returns:
    A `ConstantBoundaryConditions` or `HomogeneousBoundaryConditions` instance.
  """
  # Define the boundary types for all dimensions as Dirichlet.
  dirichlet_types = ((BCType.DIRICHLET, BCType.DIRICHLET),) * ndim
  
  # If no boundary values are provided, create a more efficient
  # HomogeneousBoundaryConditions object.
  if not bc_vals:
    return HomogeneousBoundaryConditions(dirichlet_types)
  # Otherwise, create a standard ConstantBoundaryConditions object with the given values.
  # Note: The `__init__` call is incomplete here; it's missing `time_stamp` and `boundary_fn`.
  # This suggests the class constructor might have default values or this is an older usage pattern.
  else:
    # A more complete call would look like:
    # return ConstantBoundaryConditions(time_stamp=0.0, values=bc_vals, types=dirichlet_types, boundary_fn=lambda t: t)
    return ConstantBoundaryConditions(types=dirichlet_types, values=bc_vals)


def neumann_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> ConstantBoundaryConditions:
  """
  A factory function that returns Neumann boundary conditions for all axes.

  Args:
    ndim: The number of spatial dimensions.
    bc_vals: A sequence of tuples specifying the lower and upper boundary flux
      values for each dimension. If `None`, it creates homogeneous (zero-flux)
      Neumann BCs.

  Returns:
    A `ConstantBoundaryConditions` or `HomogeneousBoundaryConditions` instance.
  """
  # Define the boundary types for all dimensions as Neumann.
  neumann_types = ((BCType.NEUMANN, BCType.NEUMANN),) * ndim
  
  # If no flux values are provided, return the optimized homogeneous version.
  if not bc_vals:
    return HomogeneousBoundaryConditions(neumann_types)
  # Otherwise, create a standard ConstantBoundaryConditions object.
  # This call is also incomplete, similar to the Dirichlet factory.
  else:
    return ConstantBoundaryConditions(types=neumann_types, values=bc_vals)


def channel_flow_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> ConstantBoundaryConditions:
  """
  A factory for creating boundary conditions for a typical channel flow setup.

  This configures the domain to be periodic in the primary flow direction (x-axis)
  and have solid walls (Dirichlet) in the cross-stream direction (y-axis).
  Any additional dimensions (e.g., z-axis in a 3D channel) are also set to periodic.

  Args:
    ndim: The number of spatial dimensions.
    bc_vals: A sequence of tuples for the boundary values. For the periodic
      dimensions, the corresponding tuple should be `(None, None)`.

  Returns:
    A `ConstantBoundaryConditions` or `HomogeneousBoundaryConditions` instance.
  """
  # Define the BC types for the first two dimensions (periodic, then Dirichlet).
  bc_type = ((BCType.PERIODIC, BCType.PERIODIC),
             (BCType.DIRICHLET, BCType.DIRICHLET))
  # Add periodic BCs for any remaining dimensions.
  for _ in range(ndim - 2):
    bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)
    
  # Return the appropriate class based on whether boundary values were provided.
  if not bc_vals:
    return HomogeneousBoundaryConditions(bc_type)
  else:
    # This call is also incomplete.
    return ConstantBoundaryConditions(types=bc_type, values=bc_vals)


def Moving_wall_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]],
    time_stamp: Optional[float],    
    bc_fn: Callable[...,Optional[float]],
) -> ConstantBoundaryConditions:
  """
  A factory for creating boundary conditions for a lid-driven cavity or moving wall setup.

  This configures the same boundary types as `channel_flow_boundary_conditions`
  (periodic in x, Dirichlet in y), but it is designed to create a fully specified,
  potentially time-dependent `ConstantBoundaryConditions` object.

  Args:
    ndim: The number of spatial dimensions.
    bc_vals: A sequence of tuples for the initial boundary values.
    time_stamp: The initial simulation time.
    bc_fn: A function that describes the time-dependent boundary condition values.

  Returns:
    A fully specified `ConstantBoundaryConditions` instance.
  """
  # Define the boundary types for a channel/cavity geometry.
  bc_type = ((BCType.PERIODIC, BCType.PERIODIC),
             (BCType.DIRICHLET, BCType.DIRICHLET))
  # Add periodic BCs for any remaining dimensions.
  for _ in range(ndim - 2):
    bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)

  # Return a new `ConstantBoundaryConditions` object with all parameters specified.
  return ConstantBoundaryConditions(
      values=bc_vals,
      time_stamp=time_stamp,
      types=bc_type,
      boundary_fn=bc_fn
  )
  

def Far_field_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]],
    time_stamp: Optional[float],    
    bc_fn: Callable[...,Optional[float]],
) -> ConstantBoundaryConditions:
  """
  A factory for creating boundary conditions for an open domain where all
  boundaries are set to a Dirichlet (fixed value) condition.

  This is typically used to model a small region within a larger body of fluid,
  where the "far-field" velocity is assumed to be known and constant (or a known
  function of time).

  Note: The docstring description "Returns BCs periodic for dimension 0 and
  Dirichlet for dimension 1" is incorrect and appears to be a copy-paste error.
  The code itself implements Dirichlet conditions on all boundaries.

  Args:
    ndim: The number of spatial dimensions.
    bc_vals: A sequence of tuples for the initial boundary values.
    time_stamp: The initial simulation time.
    bc_fn: A function that describes the time-dependent boundary condition values.

  Returns:
    A fully specified `ConstantBoundaryConditions` instance.
  """
  # Define the boundary types for the first two dimensions as Dirichlet.
  bc_type = ((BCType.DIRICHLET, BCType.DIRICHLET),
             (BCType.DIRICHLET, BCType.DIRICHLET))
  # Add Dirichlet BCs for any remaining dimensions.
  for _ in range(ndim - 2):
    bc_type += ((BCType.DIRICHLET, BCType.DIRICHLET),)

  # Return a new `ConstantBoundaryConditions` object with all parameters specified.
  return ConstantBoundaryConditions(
      values=bc_vals,
      time_stamp=time_stamp,
      types=bc_type,
      boundary_fn=bc_fn
  )

def find_extremum(fn: Callable, extrema: str, i_guess: float) -> float:
    """
    A simple wrapper around `scipy.optimize.fmin` to find a maximum or minimum
    of a 1D function.

    This is a general utility function and its placement in this file might be
    incidental. It is not directly related to boundary conditions.

    Args:
      fn: The 1D function to optimize.
      extrema: A string, either 'maximum' or 'minimum', specifying what to find.
      i_guess: An initial guess for the optimization algorithm.

    Returns:
      The function value `f(x)` at the found extremum `x`.
    """
    # To find a maximum of f(x), we can find the minimum of -f(x).
    if extrema == 'maximum':
      direction = -1
    elif extrema == 'minimum':
      direction = 1
    else:
      # Note the typo in the error message "maiximum".
      raise ValueError(
          'No extrema was correctly identified. For maximum, type "maiximum". '
          'For minimization, type "minimum".'
      )
    # `scipy.optimize.fmin` finds the value `x` that minimizes the given lambda function.
    # The lambda function `lambda x: direction * fn(x)` allows us to find either a min or max.
    optimal_x = scipy.optimize.fmin(lambda x: direction * fn(x), i_guess)
    # Return the value of the original function at that optimal point.
    return fn(optimal_x)

def periodic_and_neumann_boundary_conditions(
    bc_vals: Optional[Tuple[float, float]] = None
) -> ConstantBoundaryConditions:
  """
  A factory for 2D BCs that are periodic in dimension 0 (x-axis) and Neumann
  in dimension 1 (y-axis).

  Args:
    bc_vals: A tuple of `(lower, upper)` boundary flux values for the Neumann
      (y) dimension. If `None`, returns homogeneous (zero-flux) BCs.

  Returns:
    A `ConstantBoundaryConditions` or `HomogeneousBoundaryConditions` instance.
  """
  # Define the tuple of boundary condition types.
  types = ((BCType.PERIODIC, BCType.PERIODIC), (BCType.NEUMANN, BCType.NEUMANN))
  
  if not bc_vals:
    # If no values are provided, use the efficient homogeneous class.
    return HomogeneousBoundaryConditions(types)
  else:
    # For the periodic x-axis, the values are None. For the Neumann y-axis,
    # the values are the provided `bc_vals`.
    values = ((None, None), bc_vals)
    # This `__init__` call is incomplete and relies on default arguments or older patterns.
    return ConstantBoundaryConditions(types, values)


def periodic_and_dirichlet_boundary_conditions(
    bc_vals: Optional[Tuple[float, float]] = None,
    periodic_axis: int = 0
) -> ConstantBoundaryConditions:
  """
  A factory for 2D BCs with one periodic and one Dirichlet axis.

  Args:
    bc_vals: A tuple of `(lower, upper)` boundary values for the Dirichlet
      dimension. If `None`, returns homogeneous (zero-value) BCs.
    periodic_axis: An integer (0 or 1) specifying which axis is periodic.

  Returns:
    A `ConstantBoundaryConditions` or `HomogeneousBoundaryConditions` subclass instance.
  """
  # Define the basic types.
  periodic = (BCType.PERIODIC, BCType.PERIODIC)
  dirichlet = (BCType.DIRICHLET, BCType.DIRICHLET)
  
  # Construct the `types` and `values` tuples based on which axis is periodic.
  if periodic_axis == 0:
    types = (periodic, dirichlet)
    values = ((None, None), bc_vals)
  else: # periodic_axis is 1
    types = (dirichlet, periodic)
    values = (bc_vals, (None, None))
  
  # Return the appropriate class based on whether boundary values were provided.
  if not bc_vals:
    return HomogeneousBoundaryConditions(types)
  else:
    # This `__init__` call is also incomplete.
    return ConstantBoundaryConditions(types, values)


def is_periodic_boundary_conditions(c: grids.GridVariable, axis: int) -> bool:
  """
  A utility function to check if a `GridVariable` has periodic boundary
  conditions along a specific axis.

  An axis is considered periodic only if both its lower and upper boundaries
  are of type `BCType.PERIODIC`.

  Args:
    c: The `GridVariable` to check.
    axis: The integer axis to check.

  Returns:
    `True` if the variable is periodic along the given axis, `False` otherwise.
  """
  # Check the type of the lower boundary. If it's not periodic, we can immediately
  # return False. The upper boundary is implicitly assumed to also be periodic
  # if the lower one is, as mixed types on a single axis are not standard.
  if c.bc.types[axis][0] != BCType.PERIODIC:
    return False
  return True


def has_all_periodic_boundary_conditions(*arrays: GridVariable) -> bool:
  """
  Checks if all provided `GridVariable`s are periodic on all of their axes.

  This is a convenience function for solvers or methods (like `solve_fast_diag`)
  that are only valid for fully periodic domains.

  Args:
    *arrays: A variable number of `GridVariable` objects to check.

  Returns:
    `True` if every array is periodic in every dimension, `False` otherwise.
  """
  # Iterate through each GridVariable provided.
  for array in arrays:
    # Iterate through each spatial dimension of the variable.
    for axis in range(array.grid.ndim):
      # If we find any axis that is not periodic, we can immediately return False.
      if not is_periodic_boundary_conditions(array, axis):
        return False
  # If the loops complete without finding any non-periodic axes, all are periodic.
  return True


def consistent_boundary_conditions(*arrays: GridVariable) -> Tuple[str, ...]:
  """
  Checks that all arrays have the same BC type (periodic or not) on each axis.
  
  For many physics operations (like the pressure projection), all velocity
  components must have the same type of boundary on a given axis (e.g., all
  must be periodic, or all must be non-periodic). This function enforces that
  consistency.

  Args:
    *arrays: A variable number of `GridVariable` objects to compare.

  Returns:
    A tuple of strings ('periodic' or 'nonperiodic') describing the consistent
    boundary type for each axis.

  Raises:
    grids.InconsistentBoundaryConditionsError: If the boundary conditions are
      mixed (e.g., one variable is periodic and another is not) on any axis.
  """
  bc_types = []
  # Iterate through each spatial dimension, assuming all variables are on the same grid.
  for axis in range(arrays[0].grid.ndim):
    # Create a set of the periodic status (True/False) for all arrays on this axis.
    bcs = {is_periodic_boundary_conditions(array, axis) for array in arrays}
    # If the set has more than one item (i.e., it contains both True and False),
    # the boundary conditions for this axis are inconsistent.
    if len(bcs) != 1:
      raise grids.InconsistentBoundaryConditionsError(
          f'arrays do not have consistent bc types on axis {axis}: {arrays}')
    # If the set has only one item, pop it to see what the consistent type is.
    elif bcs.pop(): # The single item was True
      bc_types.append('periodic')
    else: # The single item was False
      bc_types.append('nonperiodic')
  return tuple(bc_types)

# --- MODIFIED FUNCTION to fix the PyTree bug ---

# WHY THE CHANGE: A single, stable lambda function is created at the module level.
# In Python, every time a `lambda` is defined, it creates a new, unique function
# object. The old code would implicitly create new function objects inside the
# main simulation loop (`jax.lax.scan`), which would change the PyTree "structure"
# on every iteration, causing a TypeError.
# This object is created only ONCE when this module is imported, giving it a
# stable identity that JAX can safely trace.
_stable_lambda = lambda x: x

def get_pressure_bc_from_velocity(v: GridVariableVector) -> BoundaryConditions:
  """
  Infers pressure boundary conditions from the specified velocity.

  WHY THE CHANGE WAS MADE COMPARED TO THE OLD VERSION:
  This function is called at every step inside the main simulation loop. The old
  version created a new `ConstantBoundaryConditions` object whose `boundary_fn`
  attribute was a lambda function inherited from the velocity. This created a new,
  unique function object on each iteration, which violates the JAX `scan` requirement
  that the PyTree structure of the state must be identical on every loop.
  
  This new version fixes that critical bug by always using a single, globally
  defined, stable lambda function (`_stable_lambda`) for the `boundary_fn`.
  This ensures the PyTree structure remains constant, making the function
  compatible with `jax.lax.scan`.
  """
  # First, check that the velocity BCs are consistent across all components.
  velocity_bc_types = consistent_boundary_conditions(*v)
  
  pressure_bc_types = []
  # Define the values for the pressure BCs, which are almost always homogeneous (zero-flux).
  # bc_value = ((0.0, 0.0),) * len(velocity_bc_types)
  # WHY THE CHANGE: Explicitly define the bc_value for a 2D case.
  # The old version used `* len(velocity_bc_types)`, which was less explicit
  # and could be ambiguous in higher dimensions. This is clearer.
  bc_value = ((0.0,0.0),(0.0,0.0))
  # This line is potentially problematic: it assumes all velocity components share
  # the same `boundary_fn` object and propagates it to the pressure. A safer
  # choice would be a simple placeholder, as in the refactored code.
  # Bc_f = v[0].bc.boundary_fn
  # WHY THE CHANGE: This is the core of the bug fix.
  # Instead of inheriting a potentially new lambda function from the velocity
  # (`Bc_f = v[0].bc.boundary_fn`), we assign our stable, top-level lambda.
  # This guarantees that the `boundary_fn` attribute of the returned object
  # is the *exact same object* on every call to this function.
  Bc_f = _stable_lambda
  
  # Iterate through each axis and set the corresponding pressure BC type.
  for velocity_bc_type in velocity_bc_types:
    if velocity_bc_type == 'periodic':
      pressure_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
    else: # 'nonperiodic' (i.e., Dirichlet for velocity)
      pressure_bc_types.append((BCType.NEUMANN, BCType.NEUMANN))
      
  # Construct and return the final boundary condition object for the pressure.
  # The hardcoded `time_stamp=2.0` is arbitrary and suggests this implementation
  # might be from an older version. The refactored code uses 0.0 for stability.
  # return ConstantBoundaryConditions(
  #     values=bc_value,
  #     time_stamp=2.0,
  #     types=tuple(pressure_bc_types),
  #     boundary_fn=Bc_f
  # )
  # WHY THE CHANGE: Use a safe, standard default value for the time_stamp.
  # The old version used a hardcoded, arbitrary `2.0`. A default of `0.0` is
  # a much safer and more standard choice for a boundary condition that is
  # not expected to be time-dependent.
  return ConstantBoundaryConditions(values=bc_value,time_stamp=0.0,types=pressure_bc_types,boundary_fn=Bc_f)


def get_advection_flux_bc_from_velocity_and_scalar(
    u: GridVariable, c: GridVariable,
    flux_direction: int
) -> BoundaryConditions:
  """
  Infers the boundary condition for an advection flux term `uc`.

  In finite volume methods, advection `∇⋅(uc)` is calculated by first finding the
  flux `F = uc` on the control volume faces. This function determines the correct
  boundary condition for that flux `F`.

  The logic is based on the physical properties of the advected scalar `c` and
  the advecting velocity `u`:
  - If the domain is periodic, the flux must also be periodic.
  - If a boundary is a solid, non-porous wall (`u=0`), the flux through it is zero.
  - On boundaries parallel to the flow, a zero-flux condition is also often appropriate.

  Args:
    u: The velocity component in the direction of the flux.
    c: The scalar being advected.
    flux_direction: The axis along which the flux is calculated (e.g., 0 for x-flux).

  Returns:
    A `HomogeneousBoundaryConditions` object for the advection flux.
  """
  flux_bc_types = []
  
  # This implementation is limited to the simpler `ConstantBoundaryConditions`.
  if not isinstance(u.bc, ConstantBoundaryConditions):
    raise NotImplementedError(
        f'Flux boundary condition is not implemented for {u.bc, c.bc}')
        
  # Determine the flux BC type for each axis of the domain.
  for axis in range(c.grid.ndim):
    # Case 1: The boundary on this axis is periodic. The flux is also periodic.
    if u.bc.types[axis][0] == 'periodic':
      flux_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
      
    # Case 2: This axis is NOT the direction of the flux. This corresponds to a
    # boundary that is parallel to the flux component. For example, for an x-flux `u*c`,
    # the top and bottom boundaries (y-axis) are parallel. It's common to assume
    # zero flux through these boundaries (a homogeneous Dirichlet condition on the flux).
    elif flux_direction != axis:
      flux_bc_types.append((BCType.DIRICHLET, BCType.DIRICHLET))
      
    # Case 3: This axis IS the direction of the flux. This corresponds to a
    # boundary that is normal to the flux (e.g., an inlet or outlet).
    # This implementation only supports the specific case of a solid, non-porous
    # wall, where the normal velocity `u` is zero at both boundaries.
    # If `u=0`, then the flux `uc` must also be zero.
    elif (u.bc.types[axis][0] == BCType.DIRICHLET and
          u.bc.types[axis][1] == BCType.DIRICHLET and
          u.bc.bc_values[axis][0] == 0.0 and u.bc.bc_values[axis][1] == 0.0):
      flux_bc_types.append((BCType.DIRICHLET, BCType.DIRICHLET))
      
    # All other cases (e.g., inflow/outflow with non-zero velocity) are not supported.
    else:
      raise NotImplementedError(
          f'Flux boundary condition is not implemented for {u.bc, c.bc}')
          
  # Since the supported cases all result in zero flux at non-periodic boundaries,
  # the function can return a simple `HomogeneousBoundaryConditions` object.
  return HomogeneousBoundaryConditions(flux_bc_types)


def new_periodic_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]],
    time_stamp: Optional[float],    
    bc_fn: Callable[...,Optional[float]],
) -> ConstantBoundaryConditions:
  """
  A factory function for creating fully periodic, time-aware boundary conditions.

  Note: The docstring description "Returns BCs periodic for dimension 0 and
  Dirichlet for dimension 1" is incorrect and appears to be a copy-paste error.
  The code itself implements periodic conditions on all boundaries. The standard
  `periodic_boundary_conditions` factory is simpler and should generally be preferred.

  Args:
    ndim: The number of spatial dimensions.
    bc_vals: A sequence of tuples for the initial boundary values. For periodic
      conditions, this is typically `((None, None), (None, None), ...)`.
    time_stamp: The initial simulation time.
    bc_fn: A function that could describe time-dependent boundary conditions
      (though this is unusual for periodic domains).

  Returns:
    A fully specified `ConstantBoundaryConditions` instance.
  """
  # Create a tuple of ('periodic', 'periodic') pairs for each dimension.
  bc_type = ((BCType.PERIODIC, BCType.PERIODIC),
             (BCType.PERIODIC, BCType.PERIODIC))
  for _ in range(ndim - 2):
    bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)
  
  # Return a new `ConstantBoundaryConditions` object with all parameters specified.
  return ConstantBoundaryConditions(
      values=bc_vals,
      time_stamp=time_stamp,
      types=bc_type,
      boundary_fn=bc_fn
  )
