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
Utility methods for manipulating array-like objects and JAX PyTrees.

This module provides a collection of helper functions that perform common array
operations. These include basic manipulations like slicing and concatenating,
as well as more specialized numerical building blocks like functions to create
Laplacian operator matrices and perform 1D interpolation. These utilities are
designed to work with JAX PyTrees, allowing them to operate on complex nested
data structures, not just single arrays.
"""

from typing import Any, Callable, List, Tuple, Union

import jax
import jax.numpy as jnp
from jax_ib.base import boundaries
from jax_ib.base import grids
import numpy as np
import scipy.linalg

# Type aliases for clarity.
PyTree = Any
Array = Union[np.ndarray, jax.Array]


def _normalize_axis(axis: int, ndim: int) -> int:
  """
  Validates an axis index and converts negative indices to positive ones.
  For example, for a 3D array, an `axis` of -1 becomes 2.
  """
  if not -ndim <= axis < ndim:
    raise ValueError(f'invalid axis {axis} for ndim {ndim}')
  if axis < 0:
    # Convert negative axis to its positive equivalent.
    axis += ndim
  return axis


def slice_along_axis(
    inputs: PyTree,
    axis: int,
    idx: Union[slice, int],
    expect_same_dims: bool = True
) -> PyTree:
  """
  Slices all arrays within a PyTree along a specified axis.

  Args:
    inputs: A JAX PyTree containing arrays to be sliced.
    axis: The axis along which to perform the slice.
    idx: An integer index or a slice object defining the portion to extract.
    expect_same_dims: If True, raises an error if arrays in the PyTree have
      different numbers of dimensions.

  Returns:
    A new PyTree with the same structure as `inputs`, but with all its
    leaf arrays sliced.
  """
  # Deconstruct the PyTree into a flat list of arrays and a definition for reconstruction.
  arrays, tree_def = jax.tree_flatten(inputs)
  ndims = set(a.ndim for a in arrays)
  if expect_same_dims and len(ndims) != 1:
    raise ValueError('arrays in `inputs` expected to have same ndims, but have '
                     f'{ndims}. To allow this, pass expect_same_dims=False')
  sliced = []
  for array in arrays:
    ndim = array.ndim
    # Create a tuple of slice objects. It will be `slice(None)` (i.e., `:`) for
    # all axes except for the target `axis`, which gets the specified `idx`.
    slc = tuple(idx if j == _normalize_axis(axis, ndim) else slice(None)
                for j in range(ndim))
    sliced.append(array[slc])
  # Reconstruct the PyTree from the new list of sliced arrays.
  return jax.tree_unflatten(tree_def, sliced)


def split_along_axis(
    inputs: PyTree,
    split_idx: int,
    axis: int,
    expect_same_dims: bool = True
) -> Tuple[PyTree, PyTree]:
  """
  Splits a PyTree into two PyTrees at a given index along an axis.
  """
  # The first part of the split is from the beginning to `split_idx`.
  first_slice = slice_along_axis(
      inputs, axis, slice(0, split_idx), expect_same_dims)
  # The second part is from `split_idx` to the end.
  second_slice = slice_along_axis(
      inputs, axis, slice(split_idx, None), expect_same_dims)
  return first_slice, second_slice


def concat_along_axis(pytrees: Tuple[PyTree, ...], axis: int) -> PyTree:
  """
  Concatenates a sequence of PyTrees along a specified axis.
  All PyTrees in the sequence must have the same structure.
  """
  # Defines a function that concatenates its arguments along the specified axis.
  concat_leaves_fn = lambda *args: jnp.concatenate(args, axis)
  # `jax.tree_map` applies this function to the corresponding leaves of each PyTree.
  return jax.tree_map(concat_leaves_fn, *pytrees)


def block_reduce(
    array: Array,
    block_size: Tuple[int, ...],
    reduction_fn: Callable[[Array], Array]
) -> Array:
  """
  Applies a reduction function to non-overlapping blocks of an array.

  This is useful for coarse-graining or down-sampling data. For example, you
  could use `jnp.mean` as the `reduction_fn` to compute the average value over
  2x2 blocks of an image.

  Args:
    array: The input array.
    block_size: A tuple specifying the size of the blocks in each dimension.
      Must evenly divide the shape of the `array`.
    reduction_fn: A function to apply to each block (e.g., `jnp.mean`, `jnp.sum`).

  Returns:
    The reduced array.
  """
  new_shape = []
  for b, s in zip(block_size, array.shape):
    multiple, residual = divmod(s, b)
    if residual != 0:
      raise ValueError('`block_size` must divide `array.shape`;'
                       f'got {block_size}, {array.shape}.')
    # Reshape the array to have new axes corresponding to the blocks.
    # e.g., (100, 100) -> (50, 2, 50, 2) for a block_size of (2, 2).
    new_shape += [multiple, b]
  
  # Apply the reduction function over the newly created block axes.
  # This vmap construction effectively applies the reduction to each block.
  reshaped_array = array.reshape(new_shape)
  # Reduce along the last block axis, then the next-to-last, and so on.
  reduced_array = reduction_fn(reshaped_array, axis=tuple(range(1, 2 * array.ndim, 2)))
  return reduced_array


def laplacian_matrix(size: int, step: float) -> np.ndarray:
  """
  Creates a 1D finite difference Laplacian operator matrix with periodic BC.

  This matrix represents the second derivative `d^2/dx^2` on a uniform grid,
  discretized using a second-order central difference stencil `[1, -2, 1]/step^2`.
  The use of `scipy.linalg.circulant` naturally enforces periodic boundary conditions.
  """
  column = np.zeros(size)
  column[0] = -2 / step**2
  column[1] = column[-1] = 1 / step**2
  return scipy.linalg.circulant(column)

def laplacian_matrix_neumann(size: int, step: float) -> np.ndarray:
  """
  Creates a 1D finite difference Laplacian matrix with homogeneous Neumann BC.

  For a Neumann condition (zero-flux), the stencil at the boundary is modified.
  This matrix is constructed using `toeplitz` and then the corner elements are
  adjusted to reflect the Neumann condition `u_ghost = u_interior`.
  """
  column = np.zeros(size)
  column[0] = -2 / step ** 2
  column[1] = 1 / step ** 2
  matrix = scipy.linalg.toeplitz(column)
  # Adjust the boundary stencil for the zero-gradient condition.
  matrix[0, 0] = matrix[-1, -1] = -1 / step**2
  return matrix

def _laplacian_boundary_dirichlet_cell_centered(laplacians: List[Array],
                                                grid: grids.Grid, axis: int,
                                                side: str) -> None:
  """
  Modifies a 1D Laplacian matrix to enforce a homogeneous Dirichlet BC.
  This function assumes the variable is located at cell centers.

  For a homogeneous Dirichlet condition (`u=0` at the wall) and a cell-centered
  variable, the ghost cell value is the negative of the first interior cell
  value (`u_ghost = -u_interior`). This changes the stencil `[1, -2, 1]` at the
  boundary to `[-1, -2, 1]`, so the diagonal entry `L[0,0]` is modified from -2
  to -3. This function performs that modification in-place.
  """
  # This modification reflects the change in the stencil at the boundary.
  if side == 'lower':
    laplacians[axis][0, 0] = laplacians[axis][0, 0] - 1 / grid.step[axis]**2
  else:
    laplacians[axis][-1, -1] = laplacians[axis][-1, -1] - 1 / grid.step[axis]**2
  # Remove the periodic wrap-around connections.
  laplacians[axis][0, -1] = 0.0
  laplacians[axis][-1, 0] = 0.0
  return


def _laplacian_boundary_neumann_cell_centered(laplacians: List[Array],
                                              grid: grids.Grid, axis: int,
                                              side: str) -> None:
  """
  Modifies a 1D Laplacian matrix to enforce a homogeneous Neumann BC.
  This function assumes the variable is located at cell centers.
  
  For a homogeneous Neumann condition (`du/dx=0`) and a cell-centered variable,
  the ghost cell value is equal to the first interior cell (`u_ghost = u_interior`).
  This changes the stencil `[1, -2, 1]` to `[2, -2, 1]`, so the diagonal `L[0,0]`
  is modified from -2 to -1. This function performs that modification in-place.
  """
  if side == 'lower':
    laplacians[axis][0, 0] = laplacians[axis][0, 0] + 1 / grid.step[axis]**2
  else:
    laplacians[axis][-1, -1] = laplacians[axis][-1, -1] + 1 / grid.step[axis]**2
  # Remove the periodic wrap-around connections.
  laplacians[axis][0, -1] = 0.0
  laplacians[axis][-1, 0] = 0.0
  return


def laplacian_matrix_w_boundaries(
    grid: grids.Grid,
    offset: Tuple[float, ...],
    bc: boundaries.BoundaryConditions,
) -> List[Array]:
  """
  Constructs a list of 1D Laplacian matrices, one for each grid dimension,
  that correctly incorporate the specified boundary conditions.

  This is a factory function that starts with periodic Laplacians and then
  calls helper methods to modify them for Dirichlet or Neumann conditions.
  This is used by matrix-based Poisson solvers.

  Args:
    grid: The `Grid` object defining the domain.
    offset: The offset of the variable on which the Laplacian will act.
    bc: The `BoundaryConditions` object for the variable.

  Returns:
    A list of NumPy arrays, where each array is the 1D Laplacian matrix for
    one dimension.
  """
  if not isinstance(bc, boundaries.ConstantBoundaryConditions):
    raise NotImplementedError(
        f'Explicit laplacians are not implemented for {bc}.')
  
  # Start with periodic Laplacian matrices for all dimensions.
  laplacians = list(map(laplacian_matrix, grid.shape, grid.step))

  # Iterate through each axis and apply modifications based on the BC type.
  for axis in range(grid.ndim):
    # Logic for cell-centered variables (offset ends in .5).
    if np.isclose(offset[axis], 0.5):
      for i, side in enumerate(['lower', 'upper']):
        if bc.types[axis][i] == boundaries.BCType.NEUMANN:
          _laplacian_boundary_neumann_cell_centered(
              laplacians, grid, axis, side)
        elif bc.types[axis][i] == boundaries.BCType.DIRICHLET:
          _laplacian_boundary_dirichlet_cell_centered(
              laplacians, grid, axis, side)
    # Logic for cell-faced variables (offset ends in .0).
    elif np.isclose(offset[axis] % 1, 0.):
      if (bc.types[axis][0] == boundaries.BCType.DIRICHLET and
          bc.types[axis][1] == boundaries.BCType.DIRICHLET):
        # For a variable on the interior of a Dirichlet domain, the effective
        # grid size for the solver is smaller.
        laplacians[axis] = laplacians[axis][:-1, :-1]
      elif boundaries.BCType.NEUMANN in bc.types[axis]:
        raise NotImplementedError(
            'edge-aligned Neumann boundaries are not implemented.')
  return laplacians


def unstack(array: Array, axis: int) -> Tuple[Array, ...]:
  """
  Splits an array into a tuple of slices along a given axis.
  This is a convenience wrapper around `jnp.split`.
  """
  squeeze_fn = lambda x: jnp.squeeze(x, axis=axis)
  return tuple(squeeze_fn(x) for x in jnp.split(array, array.shape[axis], axis))


def interp1d(
    x: Array,
    y: Array,
    axis: int = -1,
    fill_value: Union[str, Array] = jnp.nan,
    assume_sorted: bool = True,
) -> Callable[[Array], jax.Array]:
  """
  Creates a JAX-compatible 1D linear interpolation function.

  Given a set of data points `(x, y)`, this function returns a new function
  that can be called with new `x_new` values to find the corresponding `y_new`
  values via linear interpolation. This is a JAX reimplementation of a common
  pattern like `scipy.interpolate.interp1d`.

  Example:
    x = jnp.linspace(0, 10)
    y = jnp.sin(x)
    f = interp1d(x, y)
    y_new = f(1.23)  # Approximates jnp.sin(1.23)

  Args:
    x: A 1D JAX array of data point coordinates.
    y: A JAX array of data point values.
    axis: The axis of `y` that corresponds to the `x` dimension.
    fill_value: Value to use for points outside the interpolation range.
      Can be a scalar, 'extrapolate' (linear extrapolation), or
      'constant_extrapolate' (use the value of the nearest endpoint).
    assume_sorted: If True, `x` must be monotonically increasing. If False,
      the data will be sorted internally.

  Returns:
    A callable function that performs the interpolation.
  """
  # --- Input validation and setup ---
  # (Code for validation is self-explanatory)
  allowed_fill_value_strs = {'constant_extrapolate', 'extrapolate'}
  if isinstance(fill_value, str) and fill_value not in allowed_fill_value_strs:
      raise ValueError(f'`fill_value` "{fill_value}" not in {allowed_fill_value_strs}')
  
  x = jnp.asarray(x)
  y = jnp.asarray(y)
  if not assume_sorted:
    ind = jnp.argsort(x)
    x = x[ind]
    y = jnp.take(y, ind, axis=axis)
  
  axis = _normalize_axis(axis, ndim=y.ndim)
  n_x = x.shape[0]

  def interp_func(x_new: jax.Array) -> jax.Array:
    """The returned interpolation function."""
    x_new_original_shape = x_new.shape
    x_new = jnp.ravel(x_new)

    # Find the indices of the known `x` points that bracket each `x_new` point.
    # `jnp.searchsorted` efficiently finds these indices for a sorted array.
    x_new_clipped = jnp.clip(x_new, jnp.min(x), jnp.max(x)) # Clip for stable indexing.
    above_idx = jnp.minimum(n_x - 1, jnp.searchsorted(x, x_new_clipped, side='right'))
    below_idx = jnp.maximum(0, above_idx - 1)

    # Get the x and y values of the bracketing points.
    x_above = jnp.take(x, above_idx)
    x_below = jnp.take(x, below_idx)
    y_above = jnp.take(y, above_idx, axis=axis)
    y_below = jnp.take(y, below_idx, axis=axis)

    # Calculate the slope of the line segment between the bracketing points.
    # The `expand` helper reshapes the denominator to allow broadcasting with `y`.
    expand = lambda arr: jnp.reshape(arr, arr.shape + (1,) * (y.ndim - axis - 1))
    slope = (y_above - y_below) / expand(jnp.maximum(x_above - x_below, 1e-9)) # Add epsilon for stability.

    # Apply the linear interpolation formula: y_new = y_below + slope * (x_new - x_below)
    # The exact logic depends on the fill_value behavior for out-of-bounds points.
    if isinstance(fill_value, str) and fill_value == 'extrapolate':
        delta_x = expand(x_new - x_below)
        y_new = y_below + delta_x * slope
    else: # Default behavior (NaN fill or constant extrapolation)
        delta_x = expand(x_new_clipped - x_below)
        y_new = y_below + delta_x * slope
        # If not extrapolating, use `jnp.where` to replace out-of-bounds values.
        if not (isinstance(fill_value, str) and fill_value == 'constant_extrapolate'):
            is_out_of_bounds = (x_new < jnp.min(x)) | (x_new > jnp.max(x))
            y_new = jnp.where(expand(is_out_of_bounds), expand(fill_value), y_new)
            
    # Reshape the result to match the expected output shape.
    return jnp.reshape(y_new, y.shape[:axis] + x_new_original_shape + y.shape[axis + 1:])

  return interp_func
