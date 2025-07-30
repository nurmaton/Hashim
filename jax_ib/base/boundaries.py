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
"""Classes that specify how boundary conditions are applied to arrays."""

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

# Type aliases for clarity and conciseness.
BoundaryConditions = grids.BoundaryConditions
GridArray = grids.GridArray
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
Array = Union[np.ndarray, jax.Array]
BCArray = grids.BCArray


class BCType:
  """Defines string constants for different types of boundary conditions."""
  PERIODIC = 'periodic'  # Value wraps around from the end to the beginning.
  DIRICHLET = 'dirichlet'  # Fixed value at the boundary.
  NEUMANN = 'neumann'      # Fixed gradient at the boundary.

@register_pytree_node_class
@dataclasses.dataclass(init=False, frozen=False)
class ConstantBoundaryConditions(BoundaryConditions):
  """
  Stores and applies boundary conditions for a PDE variable.

  This class holds the configuration for boundary conditions, such as their type
  (periodic, Dirichlet, etc.) and values. It provides methods to manipulate
  GridArrays by padding them according to these boundary conditions. This is
  a core component for finite difference/volume methods where values in
  "ghost cells" outside the main domain need to be defined.

  This class is registered as a JAX PyTree, allowing it to be used seamlessly
  within JAX transformations like jit, vmap, and scan.
  """
  # `types` stores the boundary condition type for each dimension's lower and upper boundary.
  # Example: ((BCType.PERIODIC, BCType.PERIODIC), (BCType.DIRICHLET, BCType.DIRICHLET))
  types: Tuple[Tuple[str, str], ...]
  
  # `bc_values` stores the boundary values corresponding to the types.
  # For periodic boundaries, this is typically unused.
  bc_values: Tuple[Tuple[Optional[float], Optional[float]], ...]

  # `boundary_fn` is a callable that can describe time-dependent boundary values.
  boundary_fn: Callable[...,Optional[float]]

  # `time_stamp` tracks the current simulation time, for use with `boundary_fn`.
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
    # Ensure types and values are immutable tuples.
    types = tuple(types)
    values = tuple(values)

    # Because the dataclass is marked with init=False, we use object.__setattr__
    # to initialize the attributes of the frozen dataclass.
    object.__setattr__(self, 'bc_values', values)
    object.__setattr__(self, 'boundary_fn', boundary_fn)
    object.__setattr__(self, 'time_stamp', time_stamp if time_stamp is not None else [])
    object.__setattr__(self, 'types', types)


  def tree_flatten(self):
    """
    Defines how to flatten this object for JAX PyTree processing.

    JAX needs to know which parts of the object are dynamic (tracers, arrays)
    and which are static (metadata, configuration).
    
    Returns:
      A tuple of "children" (dynamic parts) and "aux_data" (static parts).
    """
    # `time_stamp` and `bc_values` are treated as dynamic children, allowing them
    # to be updated within JAX transformations.
    children = (self.time_stamp, self.bc_values,)
    # `types` and `boundary_fn` are static auxiliary data, as they define the
    # structure and behavior, which are not expected to change during a JAX scan.
    aux_data = (self.types, self.boundary_fn)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """
    Defines how to reconstruct the object from its flattened parts.

    Args:
      aux_data: The static data (types, boundary_fn).
      children: The dynamic data (time_stamp, bc_values).

    Returns:
      An instance of the class.
    """
    # Reconstruct the class instance from the unflattened components.
    # The order of *children and *aux_data must match the __init__ signature.
    return cls(*children, *aux_data)


  def update_bc_(self, time_stamp: float, dt: float):
    """
    Updates the time_stamp for time-dependent boundary conditions.
    
    Note: The logic that uses this has been simplified in `update_BC`,
    making this method less critical in the current version.
    """
    return time_stamp + dt


  def shift(
      self,
      u: GridArray,
      offset: int,
      axis: int,
  ) -> GridArray:
    """
    Shifts a GridArray by a given integer offset along an axis.

    This operation is fundamental for calculating finite differences. For example,
    to compute a central difference `u(i+1) - u(i-1)`, one can use
    `u.shift(1, axis) - u.shift(-1, axis)`.

    Args:
      u: The GridArray to be shifted.
      offset: The integer amount to shift by (can be positive or negative).
      axis: The axis along which to perform the shift.

    Returns:
      A new GridArray, shifted and padded according to the boundary conditions.
    """
    # Pad the array in the direction of the shift.
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

    This method is the core of boundary condition implementation. It extends the
    data array with extra cells ("ghost cells") whose values are determined
    by the boundary condition type (periodic, Dirichlet, or Neumann).

    Args:
      u: The GridArray to pad.
      width: The number of cells to add. If negative, pads the lower boundary;
             if positive, pads the upper boundary.
      axis: The axis along which to pad.

    Returns:
      A new, padded GridArray.
    """

    # Helper to determine padding configuration based on width and axis.
    def make_padding(width):
      if width < 0:  # Pad the lower boundary (e.g., left side)
        bc_type = self.types[axis][0]
        padding = (-width, 0)
      else:  # Pad the upper boundary (e.g., right side)
        bc_type = self.types[axis][1]
        padding = (0, width)
      
      # jnp.pad expects a padding configuration for all dimensions.
      full_padding = [(0, 0)] * u.grid.ndim
      full_padding[axis] = padding
      return full_padding, padding, bc_type

    full_padding, padding, bc_type = make_padding(width)
    offset = list(u.offset)
    offset[axis] -= padding[0] # Update the array's offset to reflect the new padded data.

    # Neumann padding is typically only defined for one layer of ghost cells.
    if not (bc_type == BCType.PERIODIC or
            bc_type == BCType.DIRICHLET) and abs(width) > 1:
      raise ValueError(
          f'Padding past 1 ghost cell is not defined in {bc_type} case.')

    # Trim any existing padding before applying new padding to avoid compounding.
    u, trimmed_padding = self._trim_padding(u)
    data = u.data
    # Adjust the final padding amount based on what was trimmed.
    full_padding[axis] = tuple(
        pad + trimmed_pad
        for pad, trimmed_pad in zip(full_padding[axis], trimmed_padding))

    # --- Apply padding based on BC type ---

    if bc_type == BCType.PERIODIC:
      # Periodic BCs require the array to cover the full domain dimension.
      if u.grid.shape[axis] > u.shape[axis]:
        raise ValueError('the GridArray shape does not match the grid.')
      # 'wrap' mode implements periodic padding by taking values from the other side.
      data = jnp.pad(data, full_padding, mode='wrap')

    elif bc_type == BCType.DIRICHLET:
      # Logic for cell-centered data (offset ends in .5)
      if np.isclose(u.offset[axis] % 1, 0.5):
        if u.grid.shape[axis] > u.shape[axis]:
          raise ValueError('the GridArray shape does not match the grid.')
        # This formula ensures that a linear interpolation to the boundary
        # results in the desired boundary value.
        data = (2 * jnp.pad(
            data, full_padding, mode='constant', constant_values=self.bc_values)
                - jnp.pad(data, full_padding, mode='symmetric'))
      
      # Logic for cell-face data (staggered grid, offset ends in .0)
      elif np.isclose(u.offset[axis] % 1, 0):
        if u.grid.shape[axis] > u.shape[axis] + 1:
          raise ValueError('For a dirichlet cell-face boundary condition, ' +
                           'the GridArray has more than 1 grid point missing.')
        elif u.grid.shape[axis] == u.shape[axis] + 1 and not np.isclose(
            u.offset[axis], 1):
          raise ValueError('For a dirichlet cell-face boundary condition, ' +
                           'the GridArray has more than 1 grid point missing.')

        # Determines if we need to explicitly pad with the boundary value itself.
        def _needs_pad_with_boundary_value():
          if (np.isclose(u.offset[axis], 0) and
              width > 0) or (np.isclose(u.offset[axis], 1) and width < 0):
            return True
          elif u.grid.shape[axis] == u.shape[axis] + 1:
            return True
          else:
            return False

        if _needs_pad_with_boundary_value():
          # For a single ghost cell, pad with the constant boundary value.
          if np.isclose(abs(width), 1):
            data = jnp.pad(
                data,
                full_padding,
                mode='constant',
                constant_values=self.bc_values)
          # For more than one ghost cell, the logic becomes more complex.
          elif abs(width) > 1:
            bc_padding, _, _ = make_padding(int(width / abs(width)))
            full_padding_past_bc, _, _ = make_padding(
                (abs(width) - 1) * int(width / abs(width)))
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
          # If the boundary value is already part of the array, use reflection.
          padding_values = list(self.bc_values)
          padding_values[axis] = [pad / 2 for pad in padding_values[axis]]
          data = 2 * jnp.pad(
              data,
              full_padding,
              mode='constant',
              constant_values=tuple(padding_values)) - jnp.pad(
                  data, full_padding, mode='reflect')
      else:
        raise ValueError('expected offset to be an edge or cell center, got '
                         f'offset[axis]={u.offset[axis]}')
    
    elif bc_type == BCType.NEUMANN:
      # Neumann BCs also require the full grid dimension.
      if u.grid.shape[axis] > u.shape[axis]:
        raise ValueError('the GridArray shape does not match the grid.')
      if not (np.isclose(u.offset[axis] % 1, 0) or
              np.isclose(u.offset[axis] % 1, 0.5)):
        raise ValueError('expected offset to be an edge or cell center, got '
                         f'offset[axis]={u.offset[axis]}')
      else:
        # This formula sets the ghost cell value such that the finite difference
        # approximation of the gradient at the boundary equals the Neumann value.
        data = (
            jnp.pad(data, full_padding, mode='edge') + u.grid.step[axis] *
            (jnp.pad(data, full_padding, mode='constant') - jnp.pad(
                data,
                full_padding,
                mode='constant',
                constant_values=self.bc_values)))
    else:
      raise ValueError('invalid boundary type')

    return GridArray(data, tuple(offset), u.grid)

  def _trim(
      self,
      u: GridArray,
      width: int,
      axis: int,
  ) -> GridArray:
    """
    Trims cells from the boundary of a GridArray. This is the inverse of _pad.

    Args:
      u: The GridArray to trim.
      width: The number of cells to trim. If negative, trims the lower boundary;
             if positive, trims the upper boundary.
      axis: The axis along which to trim.

    Returns:
      A new, trimmed GridArray.
    """
    if width < 0: # Trim from the lower boundary
      padding = (-width, 0)
    else: # Trim from the upper boundary
      padding = (0, width)
    
    # Calculate the slice indices.
    limit_index = u.data.shape[axis] - padding[1]
    # Use lax.slice_in_dim for JAX-compatible slicing.
    data = lax.slice_in_dim(u.data, padding[0], limit_index, axis=axis)
    # Update offset to reflect the trimmed data.
    offset = list(u.offset)
    offset[axis] += padding[0]
    return GridArray(data, tuple(offset), u.grid)

  def _trim_padding(self, u: grids.GridArray, axis=0):
    """
    Trims all excess padding from a GridArray to make it match the grid shape.

    Args:
      u: The potentially padded GridArray.
      axis: The axis to trim.

    Returns:
      A tuple containing the trimmed GridArray and the amounts that were trimmed.
    """
    padding = (0, 0)
    # If the array is larger than the grid, it has padding.
    if u.shape[axis] > u.grid.shape[axis]:
      negative_trim = 0
      if u.offset[axis] < 0:
        # Calculate and trim padding on the negative/lower side.
        negative_trim = -round(-u.offset[axis])
        u = self._trim(u, negative_trim, axis)
      # Calculate and trim any remaining padding on the positive/upper side.
      positive_trim = u.shape[axis] - u.grid.shape[axis]
      if positive_trim > 0:
        u = self._trim(u, positive_trim, axis)
      padding = (negative_trim, positive_trim)
    return u, padding

  def values(
      self, axis: int,
      grid: grids.Grid) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """Returns the boundary values as arrays broadcastable to the grid faces."""
    if None in self.bc_values[axis]:
      # Periodic boundaries don't have a single "value", so return None.
      return (None, None)
    # Create arrays of the boundary values with shapes matching the grid boundary face.
    bc_values = tuple(
        jnp.full(grid.shape[:axis] +
                 grid.shape[axis + 1:], self.bc_values[axis][-i])
        for i in [0, 1])
    return bc_values

  def trim_boundary(self, u: grids.GridArray) -> grids.GridArray:
    """
    Returns a GridArray with boundary points removed.

    For Dirichlet conditions on a staggered grid, a value might lie directly on
    the boundary. This function removes such points, returning only the
    interior values.

    Args:
      u: A GridArray that may include boundary points.

    Returns:
      A GridArray containing only interior data points.
    """
    # First, remove any ghost cell padding.
    for axis in range(u.grid.ndim):
      u, _ = self._trim_padding(u, axis)
    if u.shape != u.grid.shape:
      raise ValueError('the GridArray has already been trimmed.')
    # Next, trim points that lie exactly on a Dirichlet boundary.
    for axis in range(u.grid.ndim):
      # Trim lower boundary if it's Dirichlet and offset is 0.
      if np.isclose(u.offset[axis],
                    0.0) and self.types[axis][0] == BCType.DIRICHLET:
        u = self._trim(u, -1, axis)
      # Trim upper boundary if it's Dirichlet and offset is 1.
      elif np.isclose(u.offset[axis],
                      1.0) and self.types[axis][1] == BCType.DIRICHLET:
        u = self._trim(u, 1, axis)
    return u

  def pad_and_impose_bc(
      self,
      u: grids.GridArray,
      offset_to_pad_to: Optional[Tuple[float,
                                       ...]] = None) -> grids.GridVariable:
    """Pads an interior-only GridArray and wraps it in a GridVariable."""
    if offset_to_pad_to is None:
      offset_to_pad_to = u.offset
    # Special handling for Dirichlet boundaries on staggered grids.
    for axis in range(u.grid.ndim):
      if self.types[axis][0] == BCType.DIRICHLET and np.isclose(
          u.offset[axis], 1.0):
        if np.isclose(offset_to_pad_to[axis], 1.0):
          u = self._pad(u, 1, axis)
        elif np.isclose(offset_to_pad_to[axis], 0.0):
          u = self._pad(u, -1, axis)
    # Return a GridVariable, which pairs the data array with its BC object.
    return grids.GridVariable(u, self)

  def impose_bc(self, u: grids.GridArray) -> grids.GridVariable:
    """
    Ensures a GridArray is consistent with the boundary conditions.

    This is a convenience method that first trims any boundary values from the
    input array and then pads it correctly, returning a complete GridVariable.
    """
    offset = u.offset
    # If the array already has the shape of the grid, it might contain
    # points on the boundary that need to be trimmed first.
    if u.shape == u.grid.shape:
      u = self.trim_boundary(u)
    return self.pad_and_impose_bc(u, offset)

  # Alias common methods for convenience.
  trim = _trim
  pad = _pad


@register_pytree_node_class
class HomogeneousBoundaryConditions(ConstantBoundaryConditions):
  """
  A specialized, more efficient BC class for homogeneous conditions.
  
  This represents boundaries where the value (Dirichlet) or flux (Neumann) is
  zero. By overriding the PyTree flattening, we can tell JAX that the values
  and timestamp are truly constant and not dynamic, leading to better optimization.
  """

  def __init__(self, types: Sequence[Tuple[str, str]]):
    """
    Initializes homogeneous boundary conditions.
    
    Args:
      types: A sequence of tuples specifying the boundary types for each dimension.
    """
    ndim = len(types)
    # Values are always (0.0, 0.0) for homogeneous conditions.
    values = ((0.0, 0.0),) * ndim
    # A placeholder boundary function.
    bc_fn = lambda x: x
    # Timestamp is irrelevant, so set to 0.0.
    time_stamp = 0.0
    # Initialize the parent class with these constant values.
    super(HomogeneousBoundaryConditions, self).__init__(time_stamp, values, types, bc_fn)

  def tree_flatten(self):
    """
    Custom flattening recipe for efficiency.
    
    Since all values are constant and known, there are no dynamic "children".
    Everything is static "aux_data". This prevents JAX from tracking these
    values as dynamic tracers inside compiled functions.
    """
    children = ()  # No dynamic children.
    aux_data = (self.types,) # Only the `types` tuple is needed to reconstruct.
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Reconstructs the object from its static-only data."""
    # The class is reconstructed using only the `types` from aux_data.
    return cls(*aux_data)


@register_pytree_node_class
class TimeDependentBoundaryConditions(ConstantBoundaryConditions):
  """
  Boundary conditions that can vary with time.
  
  Note: The current implementation has some inconsistencies in argument order
  between `__init__` and `tree_unflatten` which may need correction. This
  class is less used since the `update_BC` logic was simplified.
  """

  def __init__(self, types: Sequence[Tuple[str, str]],values: Sequence[Tuple[Optional[float], Optional[float]]],boundary_fn: Callable[..., Optional[float]],time_stamp: Optional[float]):
    # The argument order here (`types`, `values`, ...) differs from the parent `__init__`.
    # This may cause issues if called directly.
    super(TimeDependentBoundaryConditions, self).__init__(types, values,boundary_fn,time_stamp)

  def tree_flatten(self):
    """Returns flattening recipe for JAX PyTree."""
    # Here, bc_values are considered dynamic children.
    children = (self.bc_values,)
    # Timestamp, types, and the function are static auxiliary data.
    aux_data = (self.time_stamp, self.types, self.boundary_fn,)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Returns unflattening recipe for JAX PyTree."""
    # Reconstructs using children and aux_data. Note the potential argument order mismatch.
    return cls(*children, *aux_data)


def boundary_function(t):
  """An example of a function defining a time-dependent boundary value."""
  A=1
  B = 1
  freq = 1
  # A simple sinusoidal function of time `t`.
  return 1+0*(A*jnp.cos(freq*t)+B*jnp.sin(freq*t))

# --- MODIFIED FUNCTIONS TO FIX THE BUGS ---

def Reserve_BC(all_variable: particle_class.All_Variables, step_time: float) -> particle_class.All_Variables:
    """
    Pass-through function. Obsolete for static BCs in the deformable model.
    
    This function was previously used to handle complex, time-dependent boundary
    conditions. It has been replaced by this simple pass-through because the
    current simulation setups (e.g., with static or periodic boundaries) do
    not require dynamic updates to BC values during the simulation loop.
    This simplifies the code and avoids potential bugs.
    """
    return all_variable

def update_BC(all_variable: particle_class.All_Variables, step_time: float) -> particle_class.All_Variables:
    """
    Pass-through function. The logic for time-dependent BCs is not needed.

    Similar to `Reserve_BC`, this function is now a placeholder. The original
    logic for updating boundary conditions based on a `boundary_fn` and `time_stamp`
    has been removed as it is not needed for the current primary use case of
    static periodic boundaries.
    """
    return all_variable

# --- END MODIFIED FUNCTIONS ---

def periodic_boundary_conditions(ndim: int) -> ConstantBoundaryConditions:
  """Factory function to create periodic BCs for a given number of dimensions."""
  return HomogeneousBoundaryConditions(
      ((BCType.PERIODIC, BCType.PERIODIC),) * ndim)

def Radom_velocity_conditions(ndim: int) -> ConstantBoundaryConditions:
    """Factory function to create moving wall BCs with initial zero velocity."""
    values = ((0.0, 0.0),) * ndim
    bc_fn = lambda x: x  # Placeholder function
    time_stamp = 0.0
    return Moving_wall_boundary_conditions(
    ndim,
    bc_vals=values,
    time_stamp=time_stamp,
    bc_fn=bc_fn,)

def dirichlet_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> ConstantBoundaryConditions:
  """Factory function for Dirichlet BCs."""
  if not bc_vals:
    # If no values are provided, create homogeneous (zero-value) Dirichlet BCs.
    return HomogeneousBoundaryConditions(
        ((BCType.DIRICHLET, BCType.DIRICHLET),) * ndim)
  else:
    # Otherwise, create ConstantBoundaryConditions with the specified values.
    return ConstantBoundaryConditions(
        ((BCType.DIRICHLET, BCType.DIRICHLET),) * ndim, bc_vals)

def neumann_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> ConstantBoundaryConditions:
  """Factory function for Neumann BCs."""
  if not bc_vals:
    # If no values provided, create homogeneous (zero-flux) Neumann BCs.
    return HomogeneousBoundaryConditions(
        ((BCType.NEUMANN, BCType.NEUMANN),) * ndim)
  else:
    # Otherwise, create ConstantBoundaryConditions with the specified flux values.
    return ConstantBoundaryConditions(
        ((BCType.NEUMANN, BCType.NEUMANN),) * ndim, bc_vals)

def channel_flow_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> ConstantBoundaryConditions:
  """
  Creates BCs for a typical channel flow setup.
  
  This sets the first axis (X) to be periodic and the second axis (Y) to be
  Dirichlet (solid walls). Any further dimensions are also set to periodic.
  """
  bc_type = ((BCType.PERIODIC, BCType.PERIODIC),
             (BCType.DIRICHLET, BCType.DIRICHLET))
  for _ in range(ndim - 2):
    bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)
  if not bc_vals:
    return HomogeneousBoundaryConditions(bc_type)
  else:
    return ConstantBoundaryConditions(bc_type, bc_vals)

def Moving_wall_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]],
    time_stamp: Optional[float],
    bc_fn: Callable[...,Optional[float]],
) -> ConstantBoundaryConditions:
  """
  Creates BCs for a setup with moving walls, typically for lid-driven cavity.
  
  Sets the first axis (X) to periodic and the second (Y) to Dirichlet.
  """
  bc_type = ((BCType.PERIODIC, BCType.PERIODIC),
             (BCType.DIRICHLET, BCType.DIRICHLET))
  for _ in range(ndim - 2):
    bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)
  # Returns a time-aware ConstantBoundaryConditions object.
  return ConstantBoundaryConditions(values=bc_vals,time_stamp=time_stamp,types=bc_type,boundary_fn=bc_fn)

def Far_field_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]],
    time_stamp: Optional[float],
    bc_fn: Callable[...,Optional[float]],
) -> ConstantBoundaryConditions:
  """
  Creates BCs for an open domain with far-field Dirichlet conditions on all sides.
  """
  bc_type = ((BCType.DIRICHLET, BCType.DIRICHLET),
             (BCType.DIRICHLET, BCType.DIRICHLET))
  for _ in range(ndim - 2):
    bc_type += ((BCType.DIRICHLET, BCType.DIRICHLET),)
  return ConstantBoundaryConditions(values=bc_vals,time_stamp=time_stamp,types=bc_type,boundary_fn=bc_fn)

def find_extremum(fn,extrema,i_guess):
    """
    A simple wrapper around scipy.optimize.fmin to find a maximum or minimum.
    
    Args:
      fn: The function to optimize.
      extrema: String, either 'maximum' or 'minimum'.
      i_guess: Initial guess for the optimization.
    
    Returns:
      The function value at the found extremum.
    """
    if extrema == 'maximum':
      direc = -1  # To find a maximum, we minimize the negative of the function.
    elif extrema == 'minimum':
      direc = 1
    else:
      raise ValueError('No extrema was correctly identified. For maximum, type "maiximum". For minimization, type "minimum". ')
    # Call scipy's optimizer and return the optimal function value.
    return fn(scipy.optimize.fmin(lambda x: direc*fn(x), i_guess))

def periodic_and_neumann_boundary_conditions(
    bc_vals: Optional[Tuple[float,
                            float]] = None) -> ConstantBoundaryConditions:
  """Creates BCs periodic on axis 0 and Neumann on axis 1."""
  if not bc_vals:
    return HomogeneousBoundaryConditions(
        ((BCType.PERIODIC, BCType.PERIODIC), (BCType.NEUMANN, BCType.NEUMANN)))
  else:
    return ConstantBoundaryConditions(
        ((BCType.PERIODIC, BCType.PERIODIC), (BCType.NEUMANN, BCType.NEUMANN)),
        ((None, None), bc_vals)) # `None` for the periodic axis values.

def periodic_and_dirichlet_boundary_conditions(
    bc_vals: Optional[Tuple[float, float]] = None,
    periodic_axis=0) -> ConstantBoundaryConditions:
  """Creates BCs with one periodic and one Dirichlet axis."""
  periodic = (BCType.PERIODIC, BCType.PERIODIC)
  dirichlet = (BCType.DIRICHLET, BCType.DIRICHLET)
  if periodic_axis == 0:
    types = (periodic, dirichlet)
    values = ((None, None), bc_vals)
  else:
    types = (dirichlet, periodic)
    values = (bc_vals, (None, None))
  
  if not bc_vals:
    return HomogeneousBoundaryConditions(types)
  else:
    return ConstantBoundaryConditions(types, values)

def is_periodic_boundary_conditions(c: grids.GridVariable, axis: int) -> bool:
  """Checks if a GridVariable has periodic BCs along a specific axis."""
  # It's periodic only if both lower and upper boundaries are periodic.
  if c.bc.types[axis][0] != BCType.PERIODIC:
    return False
  return True

def has_all_periodic_boundary_conditions(*arrays: GridVariable) -> bool:
  """Checks if all provided GridVariables are periodic on all their axes."""
  for array in arrays:
    for axis in range(array.grid.ndim):
      if not is_periodic_boundary_conditions(array, axis):
        return False
  return True

def consistent_boundary_conditions(*arrays: GridVariable) -> Tuple[str, ...]:
  """
  Checks that all arrays have the same BC type (periodic or not) on each axis.
  
  For many physics operations (like the pressure projection), all velocity
  components must have the same type of boundary on a given axis.

  Raises:
    InconsistentBoundaryConditionsError: If BCs are mixed on any axis.
  
  Returns:
    A tuple of strings ('periodic' or 'nonperiodic') for each axis.
  """
  bc_types = []
  for axis in range(arrays[0].grid.ndim):
    # Create a set of the periodic status for all arrays on this axis.
    bcs = {is_periodic_boundary_conditions(array, axis) for array in arrays}
    # If the set has more than one item, the BCs are inconsistent.
    if len(bcs) != 1:
      raise grids.InconsistentBoundaryConditionsError(
          f'arrays do not have consistent bc: {arrays}')
    elif bcs.pop():
      bc_types.append('periodic')
    else:
      bc_types.append('nonperiodic')
  return tuple(bc_types)

# --- MODIFIED FUNCTION to fix the PyTree bug ---

# Create a single, top-level lambda function. A top-level object has a stable
# identity, which is crucial for JAX's PyTree comparison inside `jit`.
_stable_lambda = lambda x: x

def get_pressure_bc_from_velocity(v: grids.GridVariableVector) -> BoundaryConditions:
  """
  Returns the appropriate pressure boundary conditions for a given velocity field.

  The rule is:
  - If velocity is periodic on an axis, pressure is also periodic.
  - If velocity is non-periodic (e.g., Dirichlet for a solid wall), the
    pressure gradient normal to that wall is zero (Neumann).

  This function also incorporates a fix for a common JAX PyTree error. By using
  a stable, top-level function (`_stable_lambda`) for `boundary_fn`, we ensure
  the PyTree structure of the returned BC object is the same every time this
  function is called within a `jit`-compiled context, preventing tracer errors.
  """
  velocity_bc_types = consistent_boundary_conditions(*v)
  pressure_bc_types = []
  # Pressure BC values are typically homogeneous (zero).
  bc_value = ((0.0,0.0),(0.0,0.0))
  
  # Use the stable, top-level lambda function. Creating a new lambda inside this
  # function on each call would result in a different object, changing the
  # PyTree structure and causing an error in `lax.scan` or `jit`.
  Bc_f = _stable_lambda

  for velocity_bc_type in velocity_bc_types:
    if velocity_bc_type == 'periodic':
      pressure_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
    else:
      pressure_bc_types.append((BCType.NEUMANN, BCType.NEUMANN))
      
  # The time_stamp can be a fixed value as it's not used for these BC types.
  # Using a constant here also helps maintain a stable PyTree structure.
  return ConstantBoundaryConditions(values=bc_value,time_stamp=0.0,types=pressure_bc_types,boundary_fn=Bc_f)

# --- END MODIFIED FUNCTION ---

def get_advection_flux_bc_from_velocity_and_scalar(
    u: GridVariable, c: GridVariable,
    flux_direction: int) -> BoundaryConditions:
  """Infers the boundary condition for an advection flux term."""
  flux_bc_types = []
  if not isinstance(u.bc, ConstantBoundaryConditions):
    raise NotImplementedError(
        f'Flux boundary condition is not implemented for {u.bc, c.bc}')
  for axis in range(c.grid.ndim):
    # If the domain is periodic, the flux is also periodic.
    if u.bc.types[axis][0] == 'periodic':
      flux_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
    # For flux parallel to a wall, the flux is typically zero.
    elif flux_direction != axis:
      flux_bc_types.append((BCType.DIRICHLET, BCType.DIRICHLET))
    # For flux normal to a non-porous wall (zero velocity), the flux is zero.
    elif (u.bc.types[axis][0] == BCType.DIRICHLET and
          u.bc.types[axis][1] == BCType.DIRICHLET and
          u.bc.bc_values[axis][0] == 0.0 and u.bc.bc_values[axis][1] == 0.0):
      flux_bc_types.append((BCType.DIRICHLET, BCType.DIRICHLET))
    else:
      # Other cases (e.g., inflow/outflow) are not implemented.
      raise NotImplementedError(
          f'Flux boundary condition is not implemented for {u.bc, c.bc}')
  return HomogeneousBoundaryConditions(flux_bc_types)

def new_periodic_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]],
    time_stamp: Optional[float],
    bc_fn: Callable[...,Optional[float]],
) -> ConstantBoundaryConditions:
  """
  A factory function for creating time-aware periodic boundary conditions.
  """
  # Define periodic types for all dimensions.
  bc_type = ((BCType.PERIODIC, BCType.PERIODIC),) * ndim
  
  return ConstantBoundaryConditions(
      values=bc_vals,
      time_stamp=time_stamp,
      types=bc_type,
      boundary_fn=bc_fn
  )
