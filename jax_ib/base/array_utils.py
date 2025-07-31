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

This module provides a collection of helper functions for common and specialized
array operations required throughout the fluid dynamics solver. These utilities
are designed to work with JAX PyTrees, allowing them to operate transparently
on complex, nested data structures like velocity vectors, not just single arrays.

The functionalities provided can be categorized as:

1.  **PyTree Slicing and Dicing**: Functions like `slice_along_axis`,
    `split_along_axis`, and `concat_along_axis` provide the ability to
    manipulate entire physical fields. For example, one could use these to
    extract a boundary layer from a velocity field or to stitch together
    subdomains for parallel processing.

2.  **Matrix-based Operator Construction**: Functions like `laplacian_matrix`
    and `laplacian_matrix_w_boundaries` are fundamental building blocks for
    matrix-based (fast diagonalization) Poisson solvers. They construct the
    explicit matrix representation of the Laplacian operator, correctly
    modified to enforce various physical boundary conditions (Periodic,
    Dirichlet, Neumann).

3.  **Numerical Primitives**: Includes specialized functions like `block_reduce`
    for coarse-graining data, `gram_schmidt_qr` for matrix orthogonalization
    tasks, and a JAX-native `interp1d` for 1D linear interpolation, which is
    essential for tasks like interpolating forces or velocities onto Lagrangian
    markers.
"""

from typing import Any, Callable, List, Tuple, Union

import jax
import jax.numpy as jnp
from jax_ib.base import boundaries
from jax_ib.base import grids
import numpy as np
import scipy.linalg

# --- Type Aliases ---
# A generic type hint for any JAX PyTree. A PyTree is a nested structure of
# containers like lists, tuples, and dicts with arrays (or other "leaves") at the end.
PyTree = Any
# A type hint for an array, which can be either a NumPy or a JAX array.
Array = Union[np.ndarray, jax.Array]


def _normalize_axis(axis: int, ndim: int) -> int:
  """
  Validates an axis index and converts negative indices to positive ones.

  This is a standard helper function for array manipulation. In Python, an axis
  can be specified with a negative index (e.g., -1 for the last axis). This
  function converts such negative indices to their positive equivalent
  (e.g., for a 3D array, -1 becomes 2) and raises an error for invalid indices.

  Args:
    axis: The axis index to normalize.
    ndim: The total number of dimensions of the array.

  Returns:
    A non-negative, valid axis index.
  """
  # Check if the axis is within the valid range [-ndim, ndim-1].
  if not -ndim <= axis < ndim:
    raise ValueError(f'invalid axis {axis} for ndim {ndim}')
  # If the axis is negative, add `ndim` to convert it to a positive index.
  if axis < 0:
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

  This function allows you to apply a slice operation uniformly to all arrays
  in a complex, nested data structure. For example, you could take the first
  half of all velocity components in a `GridVariableVector` with a single call.

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
  # Deconstruct the PyTree into a flat list of arrays (`leaves`) and a
  # `tree_def` object that describes how to reconstruct the original structure.
  arrays, tree_def = jax.tree_flatten(inputs)
  
  # Optional validation: check if all arrays in the PyTree have the same rank.
  ndims = set(a.ndim for a in arrays)
  if expect_same_dims and len(ndims) != 1:
    raise ValueError('arrays in `inputs` expected to have same ndims, but have '
                     f'{ndims}. To allow this, pass expect_same_dims=False')
                     
  sliced = []
  # Iterate through the flat list of arrays.
  for array in arrays:
    ndim = array.ndim
    # Create a tuple of slice objects for standard NumPy/JAX slicing.
    # It will be `slice(None)` (i.e., `:`) for all axes except for the target
    # `axis`, which gets the specified `idx`.
    slc = tuple(idx if j == _normalize_axis(axis, ndim) else slice(None)
                for j in range(ndim))
    # Apply the slice and append the result.
    sliced.append(array[slc])
    
  # Reconstruct the original PyTree structure from the new list of sliced arrays.
  return jax.tree_unflatten(tree_def, sliced)


def split_along_axis(
    inputs: PyTree,
    split_idx: int,
    axis: int,
    expect_same_dims: bool = True
) -> Tuple[PyTree, PyTree]:
  """
  Splits a PyTree into two PyTrees at a given index along an axis.

  This is a convenience function built on top of `slice_along_axis`.

  Args:
    inputs: A PyTree of arrays to split.
    split_idx: The index along `axis` where the second split begins.
    axis: The axis along which to split the arrays.
    expect_same_dims: If True, enforces that all arrays have the same rank.

  Returns:
    A tuple containing two new PyTrees: (`inputs` before `split_idx`,
    `inputs` from `split_idx` onwards).
  """

  # The first part of the split is from the beginning of the axis up to (but not
  # including) `split_idx`.
  first_slice = slice_along_axis(
      inputs, axis, slice(0, split_idx), expect_same_dims)
  # The second part is from `split_idx` to the end of the axis.
  second_slice = slice_along_axis(
      inputs, axis, slice(split_idx, None), expect_same_dims)
  return first_slice, second_slice


def split_axis(
    inputs: PyTree,
    axis: int,
    keep_dims: bool = False
) -> Tuple[PyTree, ...]:
  """
  Splits all arrays within a PyTree into a tuple of PyTrees along a specified axis.

  This function takes a PyTree (e.g., a velocity vector `(u, v)`) and splits
  each array leaf along the given `axis`. For example, splitting a 2D velocity
  field along `axis=0` would result in a tuple of PyTrees, where each PyTree
  represents one row of the original velocity field.

  Args:
    inputs: The PyTree to be split.
    axis: The axis along which to split the arrays.
    keep_dims: If `False` (default), the split axis is removed (squeezed) from
      the output arrays. If `True`, the split axis is kept with size 1.

  Returns:
    A tuple of new PyTrees, one for each slice along the specified axis.

  Raises:
    ValueError: If the arrays in the input PyTree have different sizes along
      the specified `axis`.
  """
  # Deconstruct the PyTree into a flat list of arrays and a tree definition.
  arrays, tree_def = jax.tree_flatten(inputs)
  
  # Check for consistency: all arrays must have the same size along the split axis.
  axis_shapes = set(a.shape[axis] for a in arrays)
  if len(axis_shapes) != 1:
    raise ValueError(f'Arrays must have equal sized axis but got {axis_shapes}')
  axis_shape, = axis_shapes
  
  # Use `jnp.split` to split each array in the flattened list into `axis_shape` pieces.
  # `splits` becomes a list of lists: [[u_slice1, u_slice2,...], [v_slice1, v_slice2,...]]
  splits = [jnp.split(a, axis_shape, axis=axis) for a in arrays]
  
  # Optionally, remove the split dimension from each slice.
  if not keep_dims:
    # `jax.tree_map` applies the squeeze function to every slice in the nested list `splits`.
    splits = jax.tree_map(lambda a: jnp.squeeze(a, axis), splits)
    
  # Transpose the list of lists using `zip(*...)`.
  # This regroups the slices: [(u_slice1, v_slice1), (u_slice2, v_slice2), ...]
  splits = zip(*splits)
  
  # Reconstruct the original PyTree structure for each group of slices.
  return tuple(jax.tree_unflatten(tree_def, leaves) for leaves in splits)


def concat_along_axis(pytrees: Tuple[PyTree, ...], axis: int) -> PyTree:
  """
  Concatenates a sequence of PyTrees along a specified axis.

  This is the inverse operation of `split_axis`. It takes a tuple of PyTrees
  (which must all have the same structure) and concatenates their corresponding
  array leaves along the given axis.

  Args:
    pytrees: A tuple of PyTrees to concatenate.
    axis: The axis along which to join the arrays.

  Returns:
    A single new PyTree with the concatenated arrays.
  """
  # Define a lambda function that takes multiple array arguments and concatenates them.
  concat_leaves_fn = lambda *args: jnp.concatenate(args, axis)
  # `jax.tree_map` applies this function to the corresponding leaves of each PyTree
  # in the input tuple. For example, it will call `concat_leaves_fn(u1, u2, ...)`
  # for the `u` component and `concat_leaves_fn(v1, v2, ...)` for the `v` component.
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
  2x2 blocks of an image, effectively reducing its resolution by a factor of 2.

  This function is equivalent to `skimage.measure.block_reduce`.

  Args:
    array: The input array.
    block_size: A tuple specifying the size of the blocks in each dimension.
      The block size must evenly divide the shape of the `array`.
    reduction_fn: A function to apply to each block (e.g., `jnp.mean`, `jnp.sum`).

  Returns:
    The reduced array.
  """
  new_shape = []
  # Reshape the array to have new axes corresponding to the blocks.
  # e.g., an array of shape (100, 100) with block_size=(2, 2) is reshaped to (50, 2, 50, 2).
  for b, s in zip(block_size, array.shape):
    multiple, residual = divmod(s, b)
    if residual != 0:
      raise ValueError('`block_size` must divide `array.shape`;'
                       f'got {block_size}, {array.shape}.')
    new_shape += [multiple, b]
    
  # Apply the reduction function over the newly created block axes.
  # This nested vmap construction is a functional way to iterate over the blocks.
  multiple_axis_reduction_fn = reduction_fn
  # The reduction is applied to the axes corresponding to the block dimensions
  # (in the reshaped array, these are axes 1, 3, 5, ...).
  for j in reversed(range(array.ndim)):
    multiple_axis_reduction_fn = jax.vmap(multiple_axis_reduction_fn, j)
  return multiple_axis_reduction_fn(array.reshape(new_shape))


def laplacian_matrix(size: int, step: float) -> np.ndarray:
  """
  Creates a 1D finite difference Laplacian operator matrix with periodic BCs.

  This matrix represents the second derivative `d²f/dx²` on a uniform grid,
  discretized using a second-order central difference stencil `[1, -2, 1]/step²`.
  The use of `scipy.linalg.circulant` naturally enforces periodic boundary
  conditions by wrapping the stencil around the edges. This is used by
  matrix-based (fast diagonalization) Poisson solvers.

  Args:
    size: The number of grid points in the dimension.
    step: The grid spacing, `dx`.

  Returns:
    A `(size, size)` NumPy array representing the periodic Laplacian operator.
  """
  # The first column of a circulant matrix defines the entire matrix.
  column = np.zeros(size)
  # Central element of the stencil: -2/dx²
  column[0] = -2 / step**2
  # Neighbor elements of the stencil: 1/dx²
  column[1] = column[-1] = 1 / step**2 # `column[-1]` creates the wrap-around.
  return scipy.linalg.circulant(column)

def laplacian_matrix_neumann(size: int, step: float) -> np.ndarray:
  """
  Creates a 1D finite difference Laplacian matrix with homogeneous Neumann BCs.

  For a Neumann condition (zero-flux), the stencil at the boundary is modified.
  This matrix is constructed using `toeplitz` for the main stencil and then the
  corner elements are adjusted to reflect the Neumann condition `u_ghost = u_interior`.

  Args:
    size: The number of grid points in the dimension.
    step: The grid spacing, `dx`.

  Returns:
    A `(size, size)` NumPy array representing the Neumann Laplacian operator.
  """
  # The first column defines the stencil for a Toeplitz matrix.
  column = np.zeros(size)
  column[0] = -2 / step ** 2
  column[1] = 1 / step ** 2
  matrix = scipy.linalg.toeplitz(column)
  # Adjust the boundary stencil for the zero-gradient condition. This changes
  # the diagonal element from -2/dx² to -1/dx².
  matrix[0, 0] = matrix[-1, -1] = -1 / step**2
  return matrix


def _laplacian_boundary_dirichlet_cell_centered(
    laplacians: List[Array],
    grid: grids.Grid,
    axis: int,
    side: str
) -> None:
  """
  Modifies a 1D periodic Laplacian matrix IN-PLACE to enforce a homogeneous Dirichlet BC.
  This function assumes the variable being operated on is located at cell centers.

  For a homogeneous Dirichlet condition (`u=0` at the wall) and a cell-centered
  variable, the ghost cell value is the negative of the first interior cell
  value (`u_ghost = -u_interior`). This changes the standard stencil `[1, -2, 1]`
  at the boundary. For the first interior point `u_1`, the Laplacian becomes:
  `(u_2 - 2*u_1 + u_ghost)/dx² = (u_2 - 2*u_1 - u_1)/dx² = (u_2 - 3*u_1)/dx²`.
  This function modifies the matrix row corresponding to `u_1` to reflect this
  change, specifically by changing the diagonal element `L[0,0]` from -2 to -3.

  Args:
    laplacians: A list of 1D Laplacian matrices (one for each dimension). This
      list is modified in-place.
    grid: The `Grid` object.
    axis: The axis along which to impose the Dirichlet BC.
    side: A string, either 'lower' or 'upper', specifying which boundary to modify.
  """
  # This function assumes a homogeneous boundary condition (value = 0).
  # The comment explains the math: for a cell-centered variable `u`, the ghost
  # cell `u[0]` is `-u[1]`. Thus, the standard stencil `[1, -2, 1]` applied to
  # `[u[0], u[1], u[2]]` becomes `[-u[1] - 2*u[1] + u[2]]`, which is `-3*u[1] + u[2]`.
  # This is equivalent to changing the diagonal element of the matrix from -2 to -3.
  if side == 'lower':
    # Modify the diagonal element for the first interior point.
    laplacians[axis][0, 0] = laplacians[axis][0, 0] - 1 / grid.step[axis]**2
  else: # side == 'upper'
    # Modify the diagonal element for the last interior point.
    laplacians[axis][-1, -1] = laplacians[axis][-1, -1] - 1 / grid.step[axis]**2
    
  # Since the boundary is no longer periodic, we must remove the "wrap-around"
  # connections in the matrix, which were created by `scipy.linalg.circulant`.
  # This is done by zeroing out the top-right and bottom-left corner elements.
  laplacians[axis][0, -1] = 0.0
  laplacians[axis][-1, 0] = 0.0
  # The function modifies the list in-place, so it doesn't need to return anything.
  return


def _laplacian_boundary_neumann_cell_centered(
    laplacians: List[Array],
    grid: grids.Grid,
    axis: int,
    side: str
) -> None:
  """
  Modifies a 1D periodic Laplacian matrix IN-PLACE to satisfy a homogeneous Neumann BC.
  This function assumes the variable is located at cell centers.

  For a homogeneous Neumann condition (`du/dx=0`) and a cell-centered variable,
  the ghost cell value is equal to the first interior cell (`u_ghost = u_interior`).
  This changes the stencil `[1, -2, 1]` at the boundary. For the first interior
  point `u_1`, the Laplacian becomes:
  `(u_2 - 2*u_1 + u_ghost)/dx² = (u_2 - 2*u_1 + u_1)/dx² = (u_2 - u_1)/dx²`.
  This function modifies the matrix row to reflect this, changing `L[0,0]` from -2 to -1.
  """
  # The logic is analogous to the Dirichlet case, but the stencil modification is different.
  if side == 'lower':
    # Modify the diagonal element for the first interior point.
    laplacians[axis][0, 0] = laplacians[axis][0, 0] + 1 / grid.step[axis]**2
  else: # side == 'upper'
    # Modify the diagonal element for the last interior point.
    laplacians[axis][-1, -1] = laplacians[axis][-1, -1] + 1 / grid.step[axis]**2
    
  # Remove the periodic wrap-around connections from the matrix.
  laplacians[axis][0, -1] = 0.0
  laplacians[axis][-1, 0] = 0.0
  # The function modifies the list in-place.
  return


def laplacian_matrix_w_boundaries(
    grid: grids.Grid,
    offset: Tuple[float, ...],
    bc: boundaries.BoundaryConditions,
) -> List[Array]:
  """
  A factory function that constructs 1D Laplacian matrices that correctly
  incorporate the specified boundary conditions.

  This is a key utility for matrix-based Poisson solvers. It starts with periodic
  Laplacian matrices for all dimensions and then calls the appropriate helper
  methods to modify them for Dirichlet or Neumann conditions where specified.

  Currently, only homogeneous (zero-value) or periodic boundary conditions are supported.

  Args:
    grid: The `Grid` object used to construct the Laplacian.
    offset: The offset of the variable on which the Laplacian will act.
    bc: The boundary conditions of the variable.

  Returns:
    A list of 1D NumPy arrays, where each array is the Laplacian matrix for
    one dimension, correctly modified for the given BCs.
  """
  if not isinstance(bc, boundaries.ConstantBoundaryConditions):
    raise NotImplementedError(
        f'Explicit laplacians are not implemented for {bc}.')
        
  # Step 1: Start with periodic Laplacian matrices for all dimensions.
  laplacians = list(map(laplacian_matrix, grid.shape, grid.step))
  
  # Step 2: Iterate through each axis and apply modifications based on the BC type.
  for axis in range(grid.ndim):
    # Case A: The variable is cell-centered (offset ~ 0.5).
    if np.isclose(offset[axis], 0.5):
      # Check both the lower and upper boundaries of this axis.
      for i, side in enumerate(['lower', 'upper']):
        if bc.types[axis][i] == boundaries.BCTType.NEUMANN:
          _laplacian_boundary_neumann_cell_centered(laplacians, grid, axis, side)
        elif bc.types[axis][i] == boundaries.BCTType.DIRICHLET:
          _laplacian_boundary_dirichlet_cell_centered(laplacians, grid, axis, side)
          
    # Case B: The variable is on a cell face/edge (offset is an integer).
    elif np.isclose(offset[axis] % 1, 0.):
      # This logic handles a staggered grid variable with Dirichlet BCs on both ends.
      if bc.types[axis][0] == boundaries.BCType.DIRICHLET and bc.types[
          axis][1] == boundaries.BCType.DIRICHLET:
        # For a variable defined on the interior faces (e.g., u-velocity), the
        # effective grid size for the solver is smaller than the number of cells.
        # We simply truncate the periodic matrix to the correct interior size.
        laplacians[axis] = laplacians[axis][:-1, :-1]
      elif boundaries.BCType.NEUMANN in bc.types[axis]:
        # Neumann conditions for face-aligned variables are not implemented in this matrix form.
        raise NotImplementedError(
            'edge-aligned Neumann boundaries are not implemented.')
            
  # Return the final list of modified 1D Laplacian matrices.
  return laplacians


def unstack(array: Array, axis: int) -> Tuple[Array, ...]:
  """
  Splits an array into a tuple of slices along a given axis, removing the split dimension.
  
  This is a convenience wrapper around `jnp.split` and `jnp.squeeze`. For example,
  `unstack` on a `(N, M, K)` array along `axis=1` would return a tuple of `M`
  arrays, each of shape `(N, K)`.

  Args:
    array: The input JAX or NumPy array.
    axis: The axis along which to split the array.

  Returns:
    A tuple of array slices.
  """
  # A helper lambda function to remove a dimension of size 1 from an array.
  squeeze_fn = lambda x: jnp.squeeze(x, axis=axis)
  
  # `jnp.split` divides the array into a list of subarrays along the specified axis.
  # The number of subarrays is determined by `array.shape[axis]`.
  # The `squeeze_fn` is then applied to each subarray to remove the now-singleton axis.
  return tuple(squeeze_fn(x) for x in jnp.split(array, array.shape[axis], axis))


def gram_schmidt_qr(
    matrix: Array,
    precision: jax.lax.Precision = jax.lax.Precision.HIGHEST
) -> Tuple[Array, Array]:
  """
  Computes the QR decomposition of a matrix using the Gram-Schmidt orthogonalization process.

  The QR decomposition factorizes a matrix `A` into an orthonormal matrix `Q`
  (meaning `Q^T * Q = I`) and an upper triangular matrix `R`, such that `A = Q * R`.

  This specific implementation uses the classical Gram-Schmidt algorithm. While
  conceptually simple, it is known to be less numerically stable than other
  methods like Householder reflections, especially for matrices that are nearly
  singular or have many columns. However, it can be suitable for tall, thin
  matrices (more rows than columns) with few columns.

  Args:
    matrix: The 2D input array `A` to be decomposed.
    precision: A JAX enum specifying the numerical precision for matrix
      multiplication. This is primarily relevant for performance on TPUs.

  Returns:
    A tuple `(Q, R)` where `Q` is an orthonormal matrix and `R` is an upper
    triangular matrix.
  """

  def orthogonalize(vector: Array, others: list[Array]) -> Array:
    """
    Orthogonalizes a `vector` with respect to a list of already orthogonal `others`.
    This is the core step of the Gram-Schmidt process.
    """
    # Base case: If there are no other vectors, just normalize the current one.
    if not others:
      return vector / jnp.linalg.norm(vector)
      
    # The projection of vector `c` onto vector `x` is `dot(c, x) * x`.
    # This lambda function subtracts this projection from `c`.
    orthogonalize_step = lambda c, x: tuple([c - jnp.dot(c, x) * x, None])
    
    # `jax.lax.scan` iteratively applies the `orthogonalize_step` function,
    # subtracting the projection onto each of the `others` vectors sequentially.
    vector, _ = jax.lax.scan(orthogonalize_step, vector, jnp.stack(others))
    
    # After removing all components parallel to the `others`, normalize the result.
    return vector / jnp.linalg.norm(vector)

  # Get the number of columns in the input matrix.
  num_columns = matrix.shape[1]
  # Unstack the matrix into a list of its column vectors.
  columns = unstack(matrix, axis=1)
  
  # Initialize lists to store the computed columns of Q and rows of R.
  q_columns = []
  r_rows = []
  
  # Iterate through each column of the original matrix `A`.
  for vec_index, column in enumerate(columns):
    # Create the next orthonormal column for `Q` by orthogonalizing the current
    # column of `A` with respect to all previously computed columns of `Q`.
    next_q_column = orthogonalize(column, q_columns)
    
    # The elements of the `R` matrix are the dot products `R_ij = dot(A_j, Q_i)`.
    # Since Q is orthonormal and R is upper triangular, `dot(A_j, Q_i)` is zero for `j < i`.
    r_rows.append(jnp.asarray([
        jnp.dot(columns[i], next_q_column) if i >= vec_index else 0.
        for i in range(num_columns)])) # This logic seems reversed; should be jnp.dot(column, q_columns[i])
        
    # Append the new orthonormal vector to the list of Q's columns.
    q_columns.append(next_q_column)
    
  # Stack the computed columns and rows to form the final Q and R matrices.
  q = jnp.stack(q_columns, axis=1)
  r = jnp.stack(r_rows)
  
  # The standard QR decomposition requires the diagonal elements of R to be positive.
  # This section corrects the signs if needed.
  # Create a diagonal matrix `D` with +1 or -1 on the diagonal.
  d = jnp.diag(jnp.sign(jnp.diagonal(r)))
  # `A = Q*R = (Q*D)*(D*R)`. We post-multiply Q by D and pre-multiply R by D.
  # This flips the sign of the corresponding columns in Q and rows in R,
  # making the diagonal of the new R positive without changing the product.
  q = jnp.matmul(q, d, precision=precision)
  r = jnp.matmul(d, r, precision=precision)
  
  return q, r


def interp1d(
    x: Array,
    y: Array,
    axis: int = -1,
    fill_value: Union[str, Array] = jnp.nan,
    assume_sorted: bool = True,
) -> Callable[[Array], jax.Array]:
  """
  A factory that builds a JAX-compatible 1D linear interpolation function.

  Given a set of data points `(x, y)` that approximate some function `f` (so `y = f(x)`),
  this function returns a new function. This new function can be called with new
  coordinate points `x_new` to find the corresponding `y_new` values via linear
  interpolation. This is a JAX reimplementation of a common pattern found in
  libraries like `scipy.interpolate.interp1d`.

  Example:
    # Create some sample data
    x = jnp.linspace(0, 10)
    y = jnp.sin(x)
    
    # Create the interpolation function
    f = interp1d(x, y)

    # Evaluate the function at a new point
    y_new = f(1.23)  # Approximates jnp.sin(1.23)

  Args:
    x: A 1D JAX array of data point coordinates (the "known" x-values).
    y: A JAX array of data point values corresponding to `f(x)`. It can be
      multi-dimensional, with one axis corresponding to the `x` dimension.
    axis: The axis of `y` that corresponds to the `x` dimension. Defaults to the last axis.
    fill_value: Value to use for points outside the interpolation range `[min(x), max(x)]`.
      - A scalar: Fills with this value.
      - 'extrapolate': Uses linear extrapolation.
      - 'constant_extrapolate': Uses the value of the nearest endpoint.
    assume_sorted: If `True`, `x` must be monotonically increasing. If `False`,
      the data will be sorted internally first.

  Returns:
    A callable function that takes an array of new x-coordinates `x_new` and
    returns an array of interpolated y-coordinates `y_new`.
  """
  # --- Input Validation and Setup ---
  
  # Validate the `fill_value` string argument.
  allowed_fill_value_strs = {'constant_extrapolate', 'extrapolate'}
  if isinstance(fill_value, str):
    if fill_value not in allowed_fill_value_strs:
      raise ValueError(
          f'`fill_value` "{fill_value}" not in {allowed_fill_value_strs}')
  else:
    # Ensure a numerical `fill_value` is a scalar.
    fill_value = np.asarray(fill_value)
    if fill_value.ndim > 0:
      raise ValueError(f'Only scalar `fill_value` supported. Found shape: {fill_value.shape}')

  # Ensure `x` is a 1D array with at least two points.
  x = jnp.asarray(x)
  if x.ndim != 1:
    raise ValueError(f'Expected `x` to be 1D. Found shape {x.shape}')
  if x.size < 2:
    raise ValueError(f'`x` must have at least 2 entries to define a line segment.')
  n_x = x.shape[0]
  
  # If `x` is not sorted, sort it and reorder `y` to match.
  if not assume_sorted:
    ind = jnp.argsort(x)
    x = x[ind]
    y = jnp.take(y, ind, axis=axis)

  # Ensure `y` is an array and its shape is compatible with `x`.
  y = jnp.asarray(y)
  if y.ndim < 1:
    raise ValueError(f'Expected `y` to have rank >= 1. Found shape {y.shape}')
  if x.size != y.shape[axis]:
    raise ValueError(
        f'x and y arrays must be equal in length along interpolation axis. '
        f'Found x.shape={x.shape} and y.shape={y.shape} and axis={axis}')

  # Normalize the axis index to be positive.
  axis = _normalize_axis(axis, ndim=y.ndim)

  def interp_func(x_new: jax.Array) -> jax.Array:
    """
    The actual interpolation function that is returned by the factory.
    It closes over the prepared `x` and `y` data.
    """
    x_new = jnp.asarray(x_new)

    # To handle arbitrary shapes of `x_new`, we flatten it for the core
    # calculation and then reshape the output to match the original `x_new` shape.
    x_new_shape = x_new.shape
    x_new = jnp.ravel(x_new)

    # --- Find Bracketing Indices ---
    # For each point in `x_new`, find the indices of the known `x` data points
    # that bracket it.
    
    # Clip `x_new` to the bounds of `x` to prevent `searchsorted` from going out of bounds.
    x_new_clipped = jnp.clip(x_new, np.min(x), np.max(x))
    # `jnp.searchsorted` efficiently finds the insertion index for each `x_new` point
    # in the sorted `x` array. This gives us the index of the point *above* `x_new`.
    above_idx = jnp.minimum(n_x - 1,
                            jnp.searchsorted(x, x_new_clipped, side='right'))
    # The index of the point *below* `x_new` is simply `above_idx - 1`.
    below_idx = jnp.maximum(0, above_idx - 1)

    # A helper to expand the shape of 1D arrays for correct broadcasting with the potentially N-D `y` array.
    def expand(array):
      array = jnp.asarray(array)
      return jnp.reshape(array, array.shape + (1,) * (y.ndim - axis - 1))

    # --- Perform Linear Interpolation ---
    # Get the x and y values of the bracketing points.
    x_above = jnp.take(x, above_idx)
    x_below = jnp.take(x, below_idx)
    y_above = jnp.take(y, above_idx, axis=axis)
    y_below = jnp.take(y, below_idx, axis=axis)
    
    # Calculate the slope of the line segment between the bracketing points.
    # Add a small epsilon to the denominator for numerical stability.
    slope = (y_above - y_below) / expand(jnp.maximum(x_above - x_below, 1e-9))

    # Apply the linear interpolation formula: y_new = y_below + slope * (x_new - x_below)
    # The exact logic depends on the fill_value behavior for out-of-bounds points.
    if isinstance(fill_value, str) and fill_value == 'extrapolate':
      # Use the original `x_new` to allow extrapolation.
      delta_x = expand(x_new - x_below)
      y_new = y_below + delta_x * slope
    elif isinstance(fill_value, str) and fill_value == 'constant_extrapolate':
      # Use the clipped `x_new` to prevent extrapolation.
      delta_x = expand(x_new_clipped - x_below)
      y_new = y_below + delta_x * slope
    else:  # The default behavior is to use the provided scalar fill_value.
      delta_x = expand(x_new - x_below)
      fill_value_ = expand(fill_value)
      y_new = y_below + delta_x * slope
      # Use `jnp.where` to replace any out-of-bounds results with the fill value.
      y_new = jnp.where(
          (delta_x < 0) | (delta_x > expand(x_above - x_below)),
          fill_value_, y_new)
    # Reshape the result to match the expected output shape, which is a combination
    # of y's shape and x_new's shape.
    return jnp.reshape(
        y_new, y_new.shape[:axis] + x_new_shape + y_new.shape[axis + 1:])

  # Return the callable interpolation function.
  return interp_func
