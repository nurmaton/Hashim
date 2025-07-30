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
Core data structures for staggered grids and the variables defined on them.

This module provides the fundamental classes for representing the discretized
computational domain and the physical quantities (like velocity and pressure)
that are defined on that domain. The key concepts are:

- `Grid`: Describes the physical size, shape, and resolution of the computational
  domain.
- `GridArray`: A container that holds a JAX array of data and annotates it with
  its physical location (offset) on a `Grid`. This is the basic building block.
- `BoundaryConditions`: An abstract base class for defining how to handle values
  at the edges of the domain.
- `GridVariable`: The most important high-level class. It bundles a `GridArray`
  with a `BoundaryConditions` object, creating a self-contained representation
  of a physical field. This is the primary data type used by the physics solvers.
"""
# This import allows a class to use its own name in type hints before it is fully defined.
from __future__ import annotations

import dataclasses
import numbers
import operator
from typing import Any, Callable, Optional, Sequence, Tuple, Union
from jax import core
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np

# TODO(jamieas): The suggestion to move common types to a separate module is good practice for larger libraries.
# TODO(shoyer): The suggestion to add `jnp.ndarray` to the `Array` type alias is a good clarification.

# --- Type Aliases ---
# Defines convenient, readable aliases for common complex types.
Array = Union[np.ndarray, jax.Array]
IntOrSequence = Union[int, Sequence[int]]
# A generic type hint for any JAX PyTree. A PyTree is a nested structure of containers
# like lists, tuples, and dicts with JAX arrays at the leaves.
PyTree = Any


@register_pytree_node_class
@dataclasses.dataclass
class BCArray(np.lib.mixins.NDArrayOperatorsMixin):
  """
  DEPRECATED or SPECIAL-USE data container.
  
  This class appears to be an older or specialized version of GridArray that
  only contains the data, without offset or grid information. In the current
  codebase, `GridArray` is the standard class for array data, as the offset
  and grid metadata are crucial for most operations. It is likely kept for
  backward compatibility or for specific internal uses where the extra metadata
  is not needed.
  """
  # The raw numerical data as a JAX or NumPy array.
  data: Array


  def tree_flatten(self):
    """
    Returns the flattening recipe for this class, required for JAX PyTree compatibility.
    It tells JAX that the `data` attribute is the dynamic "child" to be traced.
    """
    # The "children" are the parts of the PyTree that JAX will trace (e.g., arrays).
    children = (self.data,)
    # "aux_data" are the static parts needed to reconstruct the object. This class has none.
    aux_data = None
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """
    Returns the unflattening recipe, telling JAX how to reconstruct the class
    from its flattened parts.
    """
    # Reconstruct the class instance using the children.
    return cls(*children)

  @property
  def dtype(self):
    """A property to conveniently access the data type of the underlying array."""
    return self.data.dtype

  @property
  def shape(self) -> Tuple[int, ...]:
    """A property to conveniently access the shape of the underlying array."""
    return self.data.shape

  # A tuple of types that are allowed in arithmetic operations with this class.
  # It includes primitive numbers, NumPy/JAX arrays, and JAX's internal tracer types.
  _HANDLED_TYPES = (numbers.Number, np.ndarray, jax.Array,
                    core.ShapedArray, jax.core.Tracer)

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """
    Defines how NumPy universal functions (like `+`, `*`, `sin`, etc.) operate
    on `BCArray` objects. This is part of NumPy's `NDArrayOperatorsMixin`.
    """
    # Check if all inputs are of a type we know how to handle.
    for x in inputs:
      if not isinstance(x, self._HANDLED_TYPES + (BCArray,)):
        return NotImplemented # Fall back to default behavior for unknown types.
        
    # We only support standard function calls (e.g., np.add(a, b)), not more
    # complex ufunc methods like `reduce` or `accumulate`.
    if method != '__call__':
      return NotImplemented
      
    try:
      # Get the JAX equivalent of the NumPy ufunc (e.g., `jnp.add` for `np.add`).
      # This ensures that operations on these objects are JAX-jittable.
      func = getattr(jnp, ufunc.__name__)
    except AttributeError:
      return NotImplemented # If there's no JAX equivalent, we can't proceed.
      
    # Extract the raw data arrays from any BCArray inputs.
    arrays = [x.data if isinstance(x, BCArray) else x for x in inputs]
    # Apply the JAX function to the raw data.
    result = func(*arrays)
    
    # Re-wrap the raw result array(s) in a new BCArray.
    if isinstance(result, tuple):
      # Handle ufuncs that return multiple arrays (e.g., `np.divmod`).
      return tuple(BCArray(r) for r in result)
    else:
      return BCArray(result)

@register_pytree_node_class
@dataclasses.dataclass
class GridArray(np.lib.mixins.NDArrayOperatorsMixin):
  """
  A data array associated with a specific location (offset) on a grid.

  This class is the fundamental container for data in the simulation. It bundles
  a raw JAX/NumPy array with metadata describing where that data "lives" on the
  discretized domain. This is crucial for staggered grids, where different
  quantities (like x-velocity, y-velocity, and pressure) are stored at
  different locations within a grid cell.

  By registering this class as a JAX PyTree and using NumPy's `NDArrayOperatorsMixin`,
  we can perform standard arithmetic operations (e.g., `array1 + array2`, `array * 2`)
  directly on `GridArray` objects, and JAX will correctly trace these operations
  through the underlying data.

  Attributes:
    data: The raw numerical data as a JAX or NumPy array.
    offset: A tuple describing the location of the data points within a grid
      cell. For example, `(0.5, 0.5)` is the cell center, while `(1.0, 0.5)` is
      the center of the cell's right face.
    grid: The `Grid` object that this data is defined on.
  """
  # The raw numerical data.
  data: Array
  # A tuple describing the location within a grid cell (e.g., (0.5, 0.5) for cell center).
  offset: Tuple[float, ...]
  # The `Grid` object that this data is defined on.
  grid: Grid

  def tree_flatten(self):
    """
    Defines how to flatten this object for JAX PyTree processing.
    The `data` array is the dynamic "child" to be traced by JAX.
    The `offset` and `grid` are static "auxiliary data" that describe the
    structure but do not change during JAX transformations.
    """
    # The "children" are the parts of the PyTree that JAX will trace (e.g., arrays).
    children = (self.data,)
    # "aux_data" are the static parts needed to reconstruct the object.
    aux_data = (self.offset, self.grid)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Defines how to reconstruct the object from its flattened parts."""
    # Reconstruct the class instance using the children and the static aux_data.
    return cls(*children, *aux_data)

  @property
  def dtype(self):
    """A property to conveniently access the data type of the underlying array."""
    return self.data.dtype

  @property
  def shape(self) -> Tuple[int, ...]:
    """A property to conveniently access the shape of the underlying array."""
    return self.data.shape

  # A tuple of types that are allowed in arithmetic operations with this class.
  _HANDLED_TYPES = (numbers.Number, np.ndarray, jax.Array,
                    core.ShapedArray, jax.core.Tracer)

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """
    Defines how NumPy universal functions (like `+`, `*`, `sin`, etc.) operate
    on `GridArray` objects.
    """
    # Check if all inputs are of a type we know how to handle.
    for x in inputs:
      if not isinstance(x, self._HANDLED_TYPES + (GridArray,)):
        return NotImplemented
        
    # We only support standard function calls, not other ufunc methods.
    if method != '__call__':
      return NotImplemented
      
    try:
      # Get the JAX equivalent of the NumPy ufunc.
      func = getattr(jnp, ufunc.__name__)
    except AttributeError:
      return NotImplemented
      
    # Extract the raw data arrays from any GridArray inputs.
    arrays = [x.data if isinstance(x, GridArray) else x for x in inputs]
    # Apply the JAX function to the raw data.
    result = func(*arrays)
    
    # Determine the metadata for the output GridArray. The assumption is that
    # an operation on multiple GridArrays is only valid if they share a
    # consistent offset and grid.
    grid_array_inputs = [x for x in inputs if isinstance(x, GridArray)]
    offset = consistent_offset(*grid_array_inputs)
    grid = consistent_grid(*grid_array_inputs)
    
    # The commented out line is likely a remnant of a previous implementation.
    #grid = inputs.grid#consistent_grid(*[x for x in inputs])
    
    # Re-wrap the raw result array(s) in a new GridArray.
    if isinstance(result, tuple):
      # Handle ufuncs that return multiple arrays (e.g., `np.divmod`).
      return tuple(GridArray(r, offset, grid) for r in result)
    else:
      return GridArray(result, offset, grid)


# A type alias for a tuple of GridArrays. This is commonly used to represent
# a vector field, where each element of the tuple is a component (e.g., u, v, w).
GridArrayVector = Tuple[GridArray, ...]


class GridArrayTensor(np.ndarray):
  """
  A NumPy array where each element is a `GridArray`.
  
  This class is designed to represent tensor fields, where each component of
  the tensor (e.g., the stress tensor components `T_xx`, `T_xy`) is a `GridArray`
  defined on the grid. By subclassing `np.ndarray`, it allows for standard matrix
  operations like `.T` (transpose) and `.dot` to be used directly on the field,
  which is syntactically convenient for expressing tensor calculus.

  Example usage:
    # grad is a 2x2 GridArrayTensor, where each element is a GridArray.
    grad = fd.gradient_tensor(uv)
    # Standard NumPy operations like transpose work directly on the tensor object.
    strain_rate = (grad + grad.T) / 2.
    # Matrix operations like dot product and trace also work.
    nu_smag = np.sqrt(np.trace(strain_rate.dot(strain_rate)))
  """

  def __new__(cls, arrays):
    # This creates a NumPy array with `dtype=object` to hold the GridArray
    # objects, and then casts the array's view to this `GridArrayTensor` class type.
    return np.asarray(arrays, dtype=object).view(cls)


# Register GridArrayTensor as a JAX PyTree so it can be used in jitted functions.
# This tells JAX how to break the tensor down into its leaves (the individual
# GridArrays) and how to reconstruct it.
jax.tree_util.register_pytree_node(
    GridArrayTensor,
    # Flatten function: convert the tensor to a flat list of its elements and save its shape.
    lambda tensor: (tensor.ravel().tolist(), tensor.shape),
    # Unflatten function: reconstruct the tensor from the list of elements and shape.
    lambda shape, arrays: GridArrayTensor(np.asarray(arrays).reshape(shape)),
)


@dataclasses.dataclass(init=False, frozen=False)
class BoundaryConditions:
  """
  Abstract base class for boundary conditions on a PDE variable.
  
  This class defines the "contract" or interface that all concrete boundary
  condition classes (like `ConstantBoundaryConditions` in `boundaries.py`) must
  implement. By defining these abstract methods, we ensure that any BC object
  will have the necessary functionality (e.g., `.shift()`, `.pad()`) that the
  numerical solvers can reliably call.

  The methods are defined but raise `NotImplementedError`, forcing subclasses
  to provide their own specific implementations.

  Attributes:
    types: A tuple of tuples, where `types[i]` is a pair of strings specifying
      the lower and upper boundary condition types (e.g., 'periodic', 'dirichlet')
      for dimension `i`.
  """
  # Defines the expected attribute for storing BC types.
  types: Tuple[Tuple[str, str], ...]

  def shift(
      self,
      u: GridArray,
      offset: int,
      axis: int,
  ) -> GridArray:
    """
    Shifts a GridArray by a given integer offset, applying boundary conditions.

    This is the primary method for accessing neighboring data, which is essential
    for finite difference calculations. The implementation in a subclass will
    involve "padding" the array with ghost cells whose values are determined
    by the specific boundary condition (e.g., wrapping for periodic, reflecting
    for Neumann).

    Args:
      u: a `GridArray` object to be shifted.
      offset: A positive or negative integer specifying the number of grid
        cells to shift.
      axis: The axis along which to perform the shift.

    Returns:
      A new `GridArray`, shifted and padded according to the boundary conditions.
      The returned array will have a correspondingly updated `offset`.
    """
    # This base class provides the interface but not the implementation.
    raise NotImplementedError(
        'shift() must be implemented in a BoundaryConditions subclass.')

  def values(self, axis: int, grid: Grid) -> Tuple[Optional[Array], Optional[Array]]:
    """
    Returns the boundary values as arrays defined on the grid faces.

    This method is used to get the actual numerical values at the boundaries.
    For example, for a Dirichlet condition, this would return arrays of the
    prescribed fixed values. For periodic or Neumann conditions, it might return None.

    Args:
      axis: The axis along which to return boundary values.
      grid: The `Grid` object on which the boundary values are defined.

    Returns:
      A tuple of (lower_boundary_values, upper_boundary_values). Each element is
      an array whose shape matches the boundary face, or None if not applicable.
    """
    # This base class provides the interface but not the implementation.
    raise NotImplementedError(
        'values() must be implemented in a BoundaryConditions subclass.')

  def pad(
      self,
      u: GridArray,
      width: int,
      axis: int,
  ) -> GridArray:
    """
    Pads a GridArray with a specified number of ghost cells along an axis.

    This is a lower-level function that is often used by the `.shift()` method.
    It extends the data array with extra cells and fills their values according
    to the specific boundary condition rules.

    Args:
      u: A `GridArray` object to pad.
      width: The number of ghost cells to add. A negative value pads the lower
        boundary, and a positive value pads the upper boundary.
      axis: The axis along which to pad.

    Returns:
      A new, larger `GridArray` that includes the ghost cells.
    """
    # This base class provides the interface but not the implementation.
    raise NotImplementedError(
        'pad() must be implemented in a BoundaryConditions subclass.')

  def trim_boundary(self, u: GridArray) -> GridArray:
    """
    Removes boundary-coincident points from a `GridArray`.

    This is important for staggered grids where a variable might live exactly
    on a Dirichlet boundary and needs to be excluded from certain calculations
    that only involve interior points.

    Args:
      u: A `GridArray` object that may include points on the boundary.

    Returns:
      A new, potentially smaller `GridArray` containing only interior points.
    """
    # This base class provides the interface but not the implementation.
    raise NotImplementedError(
        'trim_boundary() must be implemented in a BoundaryConditions subclass.')

  def impose_bc(self, u: GridArray) -> GridVariable:
    """

    Returns a `GridVariable` with its data made consistent with its boundary conditions.

    This is a general-purpose method to ensure that the values at the edges of a
    `GridArray` correctly reflect the boundary conditions before it is used in
    a calculation.

    Args:
      u: A `GridArray` object.

    Returns:
      A `GridVariable` where the data has been adjusted to satisfy the BCs.
    """
    # This base class provides the interface but not the implementation.
    raise NotImplementedError(
        'impose_bc() must be implemented in a BoundaryConditions subclass.')


@register_pytree_node_class
@dataclasses.dataclass
class GridVariable:
  """
  Associates a `GridArray` with its corresponding `BoundaryConditions`.

  This is the main high-level data structure for representing a physical field
  (like a velocity component or the pressure field) in the simulation. By bundling
  the data array and its grid metadata (`GridArray`) with the rules for how to
  behave at the domain edges (`BoundaryConditions`), it creates a complete,
  self-contained representation of a discretized variable.

  Operations that require knowledge of neighboring cells, like the finite difference
  `.shift()` method, are conveniently exposed here, delegating their complex
  implementation to the attached `bc` object.

  Attributes:
    array: The `GridArray` containing the numerical data, its offset, and its grid.
    bc: The `BoundaryConditions` object that defines the behavior at the domain boundaries.
  """
  # The GridArray object holding the data.
  array: GridArray
  # The BoundaryConditions object associated with this variable.
  bc: BoundaryConditions

  def __post_init__(self):
    """
    A validation check that runs automatically after the dataclass is initialized.
    It ensures that the provided `array` and `bc` are consistent.
    """
    # Check that the `array` attribute is indeed a GridArray object.
    if not isinstance(self.array, GridArray):
      raise ValueError(
          f'Expected array type to be GridArray, got {type(self.array)}')
    # Check that the number of dimensions in the boundary conditions matches
    # the number of dimensions of the grid.
    if len(self.bc.types) != self.grid.ndim:
      raise ValueError(
          'Incompatible dimension between grid and bc, grid dimension = '
          f'{self.grid.ndim}, bc dimension = {len(self.bc.types)}')

  def tree_flatten(self):
    """
    Returns the flattening recipe for this class, required for JAX PyTree compatibility.
    Both the `array` and `bc` objects are treated as "children", meaning JAX
    will trace through both of them.
    """
    children = (self.array, self.bc)
    aux_data = None
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """
    Returns the unflattening recipe, telling JAX how to reconstruct the class
    from its flattened parts.
    """
    return cls(*children)

  # --- Convenience Properties ---
  # These properties allow direct access to the attributes of the contained
  # `GridArray` object, making the code cleaner (e.g., `var.offset` instead
  # of `var.array.offset`).

  @property
  def dtype(self):
    """Returns the data type of the underlying array."""
    return self.array.dtype

  @property
  def shape(self) -> Tuple[int, ...]:
    """Returns the shape of the underlying array."""
    return self.array.shape

  @property
  def data(self) -> Array:
    """Returns the raw numerical data array."""
    return self.array.data

  @property
  def offset(self) -> Tuple[float, ...]:
    """Returns the offset of the variable on the grid."""
    return self.array.offset

  @property
  def grid(self) -> Grid:
    """Returns the `Grid` object this variable is defined on."""
    return self.array.grid

  def shift(
      self,
      offset: int,
      axis: int,
  ) -> GridArray:
    """
    Shifts the variable's data by a given offset along an axis.

    This is the primary method used for finite difference calculations. It
    delegates the complex logic of padding the array with ghost cells to the
    `shift` method of its `BoundaryConditions` object.

    Args:
      offset: A positive or negative integer specifying the number of grid
        cells to shift.
      axis: The axis along which to perform the shift.

    Returns:
      A `GridArray` containing the shifted data. Note that it returns a
      `GridArray`, not a `GridVariable`, because the boundary conditions of the
      shifted result are not well-defined.
    """
    return self.bc.shift(self.array, offset, axis)

  def _interior_grid(self) -> Grid:
    """
    Calculates the `Grid` corresponding to only the interior points of the domain.
    This is a helper method used in situations where boundary points should be excluded.
    """
    grid = self.array.grid
    domain = list(grid.domain)
    shape = list(grid.shape)
    for axis in range(self.grid.ndim):
      # Periodic boundaries are all considered "interior".
      if self.bc.types[axis][0] == 'periodic':
        continue
      # For Dirichlet/Neumann boundaries, if the variable lies on a cell face
      # (offset 0.0 or 1.0), the grid size is reduced by one.
      if np.isclose(self.array.offset[axis], 1.0):
        shape[axis] -= 1
        domain[axis] = (domain[axis][0], domain[axis][1] - grid.step[axis])
      elif np.isclose(self.array.offset[axis], 0.0):
        shape[axis] -= 1
        domain[axis] = (domain[axis][0] + grid.step[axis], domain[axis][1])
    # Return a new Grid object with the adjusted shape and domain.
    return Grid(tuple(shape), domain=tuple(domain))

  def trim_boundary(self) -> GridArray:
    """
    Returns a `GridArray` containing only the interior data points.

    This method removes points that lie exactly on a non-periodic boundary.
    It delegates the actual trimming implementation to the `trim_boundary`
    method of its `BoundaryConditions` object.

    Returns:
      A new, potentially smaller `GridArray` containing only interior points.
    """
    return self.bc.trim_boundary(self.array)

  def impose_bc(self) -> GridVariable:
    """
    Returns a `GridVariable` with its data made consistent with its boundary conditions.

    For variables on a staggered grid, some data points may lie exactly on the
    boundary. This method ensures those points have the correct values as
    defined by the boundary condition (e.g., for a Dirichlet BC). It delegates
    the implementation to its `bc` object.
    """
    return self.bc.impose_bc(self.array)


# A type alias for a tuple of GridVariables. This is the standard way to represent
# a vector field (like velocity) where each component is a `GridVariable`.
GridVariableVector = Tuple[GridVariable, ...]


def applied(func):
  """
  A decorator that converts a standard array function into one that operates on `GridArray` objects.

  Many JAX and NumPy functions (e.g., `jnp.where`, `jnp.sqrt`) are designed to
  work on raw arrays. This decorator wraps such a function, allowing it to be
  called directly with `GridArray` objects as arguments.

  The wrapper automatically:
  1. Extracts the raw `.data` arrays from the input `GridArray` arguments.
  2. Calls the original function with these raw arrays.
  3. Takes the raw array result and wraps it back into a new `GridArray`,
     preserving a consistent grid and offset.

  Args:
    func: The array-based function to be wrapped (e.g., `jnp.where`).

  Returns:
    A new function that can be called with `GridArray` objects.
  """

  def wrapper(*args, **kwargs):
    # This function is intended for `GridArray` objects. Using it with `GridVariable`
    # would discard the crucial boundary condition information, so we raise an error.
    for arg in args + tuple(kwargs.values()):
      if isinstance(arg, GridVariable):
        raise ValueError('grids.applied() cannot be used with GridVariable')

    # Collect all GridArray objects from the input arguments.
    grid_array_args = [
        arg for arg in args + tuple(kwargs.values())
        if isinstance(arg, GridArray)
    ]
    # Ensure that all GridArray inputs share the same offset and grid.
    offset = consistent_offset(*grid_array_args)
    grid = consistent_grid(*grid_array_args)
    
    # Extract the raw `.data` arrays from the inputs, leaving other arguments as is.
    raw_args = [arg.data if isinstance(arg, GridArray) else arg for arg in args]
    raw_kwargs = {
        k: v.data if isinstance(v, GridArray) else v for k, v in kwargs.items()
    }
    
    # Call the original function with the raw data.
    data = func(*raw_args, **raw_kwargs)
    
    # Wrap the raw output data in a new GridArray with the consistent metadata.
    return GridArray(data, offset, grid)

  return wrapper


# Create convenient aliases for commonly used `applied` functions.
# This allows for cleaner syntax, e.g., `grids.where(...)` instead of `grids.applied(jnp.where)(...)`.
where = applied(jnp.where)


def averaged_offset(
    *arrays: Union[GridArray, GridVariable]
) -> Tuple[float, ...]:
  """
  Returns the element-wise average of the offsets of the given arrays.
  This is useful for determining the offset of a result from a finite difference
  stencil, which is typically located at the center of the stencil points.
  """
  # `np.mean` is used because offsets are static metadata, not traced JAX arrays.
  if not arrays: return () # Handle the case of no input arrays.
  offset = np.mean([array.offset for array in arrays], axis=0)
  return tuple(offset.tolist())


def control_volume_offsets(
    c: Union[GridArray, GridVariable]
) -> Tuple[Tuple[float, ...], ...]:
  """
  Returns the offsets for the faces of a control volume centered on `c`.

  This is a key utility for finite volume methods. On a staggered grid, fluxes
  are computed at the faces of a control volume. If a scalar `c` is at the cell
  center (offset `(0.5, 0.5)`), this function returns the offsets of the right
  face `(1.0, 0.5)` and the top face `(0.5, 1.0)`.
  """
  # This list comprehension elegantly calculates the face offsets. For each
  # dimension `j`, it creates a new offset tuple by adding 0.5 to the j-th
  # component of the input offset `c.offset`.
  return tuple(
      tuple(o + .5 if i == j else o
            for i, o in enumerate(c.offset))
      for j in range(len(c.offset)))


# --- Custom Exception Classes ---
# Defining custom exceptions makes error messages more specific and informative.

class InconsistentOffsetError(Exception):
  """Raised when combining arrays that have different offsets."""


def consistent_offset(
    *arrays: Union[GridArray, GridVariable]
) -> Tuple[float, ...]:
  """
  Checks that all input arrays have the same offset and returns that offset.
  If the offsets are not identical, it raises an `InconsistentOffsetError`.
  This is crucial for ensuring that arithmetic operations are physically meaningful.
  """
  if not arrays: return () # Handle the case of no input arrays.
  # Create a set of all unique offsets from the input arrays.
  offsets = {array.offset for array in arrays}
  # If the set has more than one element, the offsets are inconsistent.
  if len(offsets) != 1:
    raise InconsistentOffsetError(
        f'arrays do not have a unique offset: {offsets}')
  # If the set has exactly one element, return it.
  offset, = offsets
  return offset


class InconsistentGridError(Exception):
  """Raised when combining arrays defined on different grids."""


def consistent_grid(*arrays: Union[GridArray, GridVariable]) -> Grid:
  """
  Checks that all input arrays are defined on the same grid and returns that grid.
  If the grids are not identical, it raises an `InconsistentGridError`.
  """
  if not arrays: return None # Handle the case of no input arrays.
  # Create a set of all unique grids from the input arrays.
  grids = {array.grid for array in arrays}
  # If the set has more than one element, the grids are inconsistent.
  if len(grids) != 1:
    raise InconsistentGridError(f'arrays do not have a unique grid: {grids}')
  # If the set has exactly one element, return it.
  grid, = grids
  return grid
  # The commented out line is likely a remnant of a previous, less safe implementation.
  #return arrays[0].grid


class InconsistentBoundaryConditionsError(Exception):
  """Raised when combining `GridVariable`s with different boundary conditions."""


def consistent_boundary_conditions(*arrays: GridVariable) -> BoundaryConditions:
  """
  Checks that all input variables have the same boundary conditions and returns them.
  If the BCs are not identical, it raises an `InconsistentBoundaryConditionsError`.
  """
  if not arrays: return None # Handle the case of no input arrays.
  # Create a set of all unique boundary condition objects.
  bcs = {array.bc for array in arrays}
  # If the set has more than one element, the BCs are inconsistent.
  if len(bcs) != 1:
    raise InconsistentBoundaryConditionsError(
        f'arrays do not have a unique bc: {bcs}')
  # If the set has exactly one element, return it.
  bc, = bcs
  return bc


@dataclasses.dataclass(init=False, frozen=True)
class Grid:
  """
  Describes the size, shape, and physical domain of the computational grid.

  This class defines the discretized space on which the simulation takes place.
  It is immutable (`frozen=True`) because the grid is assumed to be static
  throughout the simulation. All attributes are static metadata.

  The grid is defined by providing its `shape` (number of cells) and either its
  physical `domain` or its cell `step` size.

  Attributes:
    shape: A tuple of integers giving the number of grid cells in each dimension.
    step: A tuple of floats giving the physical size (e.g., `dx`) of each
      grid cell in each dimension.
    domain: A tuple of pairs `((x_min, x_max), (y_min, y_max), ...)`
      defining the physical boundaries of the simulation domain.
  """
  # Attribute type hints for clarity.
  shape: Tuple[int, ...]
  step: Tuple[float, ...]
  domain: Tuple[Tuple[float, float], ...]

  def __init__(
      self,
      shape: Sequence[int],
      step: Optional[Union[float, Sequence[float]]] = None,
      domain: Optional[Union[float, Sequence[Tuple[float, float]]]] = None,
  ):
    """
    Constructs a grid object. You must provide `shape` and EITHER `step` OR `domain`.
    """
    # Ensure the shape is a tuple of integers.
    shape = tuple(operator.index(s) for s in shape)
    # Use object.__setattr__ because the dataclass is frozen.
    object.__setattr__(self, 'shape', shape)

    # --- Logic to determine step and domain based on user input ---
    if step is not None and domain is not None:
      raise TypeError('Cannot provide both `step` and `domain` to Grid constructor')
    elif domain is not None:
      # If the domain is provided, parse it into a standard format.
      if isinstance(domain, (int, float)): # e.g., domain=10.0 for a 2D grid
        domain = ((0, domain),) * len(shape) # becomes ((0, 10.0), (0, 10.0))
      else:
        # Perform validation checks on the domain format.
        if len(domain) != self.ndim:
          raise ValueError(f'length of domain does not match ndim: {len(domain)} vs {self.ndim}')
        for bounds in domain:
          if len(bounds) != 2:
            raise ValueError(f'domain must be a sequence of (lower, upper) pairs: {domain}')
      domain = tuple((float(lower), float(upper)) for lower, upper in domain)
      
    else:
      # If the step is provided (or defaulted), derive the domain from it.
      if step is None: step = 1.0 # Default to unit step size.
      if isinstance(step, numbers.Number): # e.g., step=0.1
        step = (step,) * self.ndim         # becomes (0.1, 0.1, ...)
      elif len(step) != self.ndim:
        raise ValueError(f'length of step does not match ndim: {len(step)} vs {self.ndim}')
      # Domain starts at 0 and has total size = step * num_cells.
      domain = tuple(
          (0.0, float(step_ * size)) for step_, size in zip(step, shape))

    # Set the final domain attribute.
    object.__setattr__(self, 'domain', domain)

    # The step size is always re-derived from the final domain and shape to
    # ensure perfect consistency and avoid floating point errors.
    step = tuple(
        (upper - lower) / size for (lower, upper), size in zip(domain, shape))
    object.__setattr__(self, 'step', step)

  @property
  def ndim(self) -> int:
    """Returns the number of dimensions of this grid (e.g., 2 for 2D)."""
    return len(self.shape)

  @property
  def cell_center(self) -> Tuple[float, ...]:
    """Returns the offset `(0.5, 0.5, ...)` for the center of each grid cell."""
    return self.ndim * (0.5,)

  @property
  def cell_faces(self) -> Tuple[Tuple[float, ...]]:
    """
    Returns the offsets for the cell faces, used for staggering velocity.
    For 2D, this would be `((1.0, 0.5), (0.5, 1.0))` (the right and top faces).
    """
    d = self.ndim
    # This is a clever way to generate the face offsets: the identity matrix
    # gives the components (1,0,0..), (0,1,0..), etc. Adding a matrix of ones
    # and dividing by 2 results in (1.0, 0.5, ...), (0.5, 1.0, ...), etc.
    offsets = (np.eye(d) + np.ones([d, d])) / 2.
    return tuple(tuple(float(o) for o in offset) for offset in offsets)

  def stagger(self, v: Tuple[Array, ...]) -> Tuple[GridArray, ...]:
    """
    Takes a tuple of raw velocity component arrays and places them on the
    staggered cell faces, returning a tuple of `GridArray`s.
    """
    offsets = self.cell_faces
    return tuple(GridArray(u, o, self) for u, o in zip(v, offsets))

  def center(self, v: PyTree) -> PyTree:
    """
    Takes a PyTree of raw arrays (e.g., pressure) and places them at the cell
    center, returning a PyTree of `GridArray`s.
    """
    offset = self.cell_center
    return jax.tree_map(lambda u: GridArray(u, offset, self), v)

  def axes(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
    """
    Returns a tuple of 1D arrays, where each array contains the physical
    coordinates of the grid points along one axis for a given offset.
    """
    if offset is None: offset = self.cell_center
    if len(offset) != self.ndim:
      raise ValueError(f'unexpected offset length: {len(offset)} vs {self.ndim}')
    # Formula for coordinates: x_i = x_lower + (i + offset) * dx
    return tuple(lower + (jnp.arange(length) + offset_i) * step
                 for (lower, _), offset_i, length, step in zip(
                     self.domain, offset, self.shape, self.step))

  def fft_axes(self) -> Tuple[Array, ...]:
    """
    Returns the tuple of Fourier-space frequency coordinates for each axis.
    This is used by FFT-based solvers (e.g., for the Poisson equation). The
    result corresponds to the frequencies `k` in the Fourier basis `e^(ikx)`.
    """
    freq_axes = tuple(
        jnp.fft.fftfreq(n, d=s) for (n, s) in zip(self.shape, self.step))
    return freq_axes

  def rfft_axes(self) -> Tuple[Array, ...]:
    """
    Returns Fourier-space frequency coordinates for real-to-complex FFTs (`rfft`).
    The last axis is handled differently for real FFTs as it only stores the
    non-negative frequencies due to Hermitian symmetry.
    """
    fft_axes = tuple(
        jnp.fft.fftfreq(n, d=s)
        for (n, s) in zip(self.shape[:-1], self.step[:-1]))
    rfft_axis = (jnp.fft.rfftfreq(self.shape[-1], d=self.step[-1]),)
    return fft_axes + rfft_axis

  def mesh(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
    """
    Returns a tuple of N-D arrays containing the grid coordinates at every point.
    This is equivalent to `jnp.meshgrid(..., indexing='ij')` and is useful for
    initializing fields from an analytical function of space.
    """
    # First, get the 1D coordinate axes.
    axes = self.axes(offset)
    # Then, broadcast them into N-D arrays.
    return tuple(jnp.meshgrid(*axes, indexing='ij'))

  def rfft_mesh(self) -> Tuple[Array, ...]:
    """
    Returns a tuple of N-D arrays of Fourier-space frequency coordinates.
    This is the frequency-space equivalent of the `.mesh()` method.
    """
    rfft_axes = self.rfft_axes()
    return tuple(jnp.meshgrid(*rfft_axes, indexing='ij'))

  def eval_on_mesh(self,
                   fn: Callable[..., Array],
                   offset: Optional[Sequence[float]] = None) -> GridArray:
    """
    A convenience method to evaluate a function of space `f(x, y, ...)` on the
    grid and automatically wrap the result in a `GridArray`.
    """
    if offset is None: offset = self.cell_center
    # Create the mesh for the given offset, pass it to the function, and
    # wrap the resulting data array in a GridArray with the correct metadata.
    return GridArray(fn(*self.mesh(offset)), offset, self)


def domain_interior_masks(grid: Grid) -> Tuple[np.ndarray, ...]:
  """
  Returns boolean masks that identify the interior cell faces of the domain.

  This utility function generates a set of masks, one for each face direction
  (e.g., right faces, top faces). Each mask is a grid-sized array where the
  value is `1` if the corresponding face is in the interior of the domain and
  `0` if it lies exactly on a physical boundary.

  These masks are useful for applying operations only to interior faces, for
  example, when implementing certain boundary conditions or in post-processing.

  Args:
    grid: The `Grid` object defining the simulation domain.

  Returns:
    A tuple of NumPy arrays. For a 2D grid, this would be
    `(right_face_mask, top_face_mask)`.
  """
  # A list to store the generated mask for each face direction.
  masks = []
  
  # Iterate through the offsets of the cell faces (e.g., (1.0, 0.5), (0.5, 1.0) in 2D).
  for offset in grid.cell_faces:
    # Generate the meshgrid of physical coordinates for this set of faces.
    # Note: This uses np.isclose, suggesting it's intended for setup and not
    # for use inside a JAX-jitted function. If needed in JAX, jnp should be used.
    mesh = grid.mesh(offset)
    
    # Initialize the mask for this face direction to all ones (assuming all are interior).
    mask = np.ones(grid.shape, dtype=int)
    
    # Iterate through each dimension (x, y, ...) of the grid.
    for i, x_coords in enumerate(mesh):
      # For the current dimension `i`, check which points in the mesh lie on the
      # lower or upper physical boundaries of the domain.
      
      # `np.isclose` creates a boolean array, True where a face is on the lower boundary.
      # `np.invert` flips this to be True for interior points.
      # `.astype('int')` converts True/False to 1/0.
      is_not_on_lower_boundary = (np.invert(np.isclose(x_coords, grid.domain[i][0]))).astype('int')
      is_not_on_upper_boundary = (np.invert(np.isclose(x_coords, grid.domain[i][1]))).astype('int')
      
      # A face is interior to *this dimension* if it is not on the lower OR upper boundary.
      # By multiplying the masks, we accumulate the conditions. A final mask value
      # of 1 means the face was not on a boundary in *any* dimension.
      mask = mask * is_not_on_lower_boundary * is_not_on_upper_boundary
      
    # Add the completed mask for this face direction to the list.
    masks.append(mask)
    
  # Return the list of masks as a tuple.
  return tuple(masks)
