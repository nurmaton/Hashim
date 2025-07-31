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
Functions for pressure projection and solving the Poisson equation for pressure.

In an incompressible fluid, the velocity field `v` must be "divergence-free"
(i.e., `∇ ⋅ v = 0`). This mathematical constraint ensures that mass is conserved;
the net flow of fluid into any small volume must equal the net flow out.

This module implements the **pressure projection** method, a standard algorithm
to enforce this constraint at each time step. The method consists of two main stages:

1.  **Solve the Pressure Poisson Equation**: After an initial "predictor" step,
    the velocity field `v*` is generally not divergence-free. A Poisson equation,
    `∇²q = ∇ ⋅ v*`, is solved for a pressure-like correction field `q`. The
    right-hand-side (RHS), `∇ ⋅ v*`, represents the amount of "compressibility"
    that needs to be projected out.

2.  **Correct the Velocity**: The velocity field is then corrected by subtracting
    the gradient of `q`, `v_new = v* - ∇q`. This correction ensures that the
    final velocity field `v_new` is divergence-free.

This module provides `projection_and_update_pressure` as the main high-level
function to perform this entire process. It also offers several specialized
low-level functions to solve the Poisson equation itself, each optimized for
different types of boundary conditions:
-   `solve_fast_diag`: For fully periodic domains, using a highly efficient FFT-based method.
-   `solve_fast_diag_moving_wall`: For channel flows (periodic in one direction, Neumann in another).
-   `solve_fast_diag_Far_Field`: The most general solver, handling arbitrary
    combinations of Dirichlet and Neumann boundary conditions.
"""

from typing import Callable, Optional
import scipy.linalg
import numpy as np
from jax_ib.base import array_utils
from jax_cfd.base import fast_diagonalization
import jax.numpy as jnp
from jax_cfd.base import pressure # Import from jax_cfd for potential reuse.
from jax_ib.base import grids
from jax_ib.base import boundaries
from jax_ib.base import finite_differences as fd
from jax_ib.base import particle_class

# --- Type Aliases ---
Array = grids.Array
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
BoundaryConditions = grids.BoundaryConditions


def laplacian_matrix_neumann(size: int, step: float) -> np.ndarray:
  """
  Creates a 1D finite difference Laplacian operator matrix with homogeneous Neumann BC.

  This is a helper function used to construct the Poisson solver for non-periodic
  domains. The standard central difference stencil is `[1, -2, 1]/step^2`. For a
  Neumann condition (zero-flux), the stencil at the boundary is modified to be
  `[-1, 1]` effectively, which changes the diagonal `[-2]` term to `[-1]`.

  Args:
    size: The number of grid points in the dimension.
    step: The grid spacing `dx`.

  Returns:
    A NumPy array representing the 1D Neumann Laplacian operator.
  """
  # Start with the standard central difference stencil for the first column.
  column = np.zeros(size)
  column[0] = -2 / step ** 2
  column[1] = 1 / step ** 2
  # `scipy.linalg.toeplitz` creates the full matrix from this first column,
  # resulting in a symmetric matrix with the desired stencil on the inner diagonals.
  matrix = scipy.linalg.toeplitz(column)
  # Modify the diagonal elements at the two boundaries to enforce the Neumann condition.
  matrix[0, 0] = matrix[-1, -1] = -1 / step**2
  return matrix


def _rhs_transform(
    u: grids.GridArray,
    bc: boundaries.BoundaryConditions,
) -> Array:
  """
  Transforms the RHS of the Poisson equation to ensure a solution exists.

  When a Poisson equation `∇²x = u` has Neumann boundary conditions on all sides,
  a solution `x` only exists if the integral (or mean) of the right-hand-side `u`
  is zero. This is a mathematical solvability condition. This function enforces
  this condition by subtracting the mean from `u`. The solution `x` is then
  unique up to an arbitrary additive constant.

  Args:
    u: A `GridArray` for the RHS of the Poisson equation (i.e., the divergence).
    bc: The boundary conditions for the pressure field.

  Returns:
    A new data array for the RHS with its mean subtracted, if necessary.
  """
  u_data = u.data
  # Check if all boundaries for every axis are Neumann.
  is_all_neumann = all(
      bc_type == boundaries.BCType.NEUMANN
      for axis_bcs in bc.types for bc_type in axis_bcs
  )
  if is_all_neumann:
    # If so, subtract the mean from the data to satisfy the solvability condition.
    u_data = u_data - jnp.mean(u_data)
  return u_data
  

def projection_and_update_pressure(
    All_variables: particle_class.All_Variables,
    solve: Callable = pressure.solve_fast_diag,
) -> particle_class.All_Variables:
  """
  Applies pressure projection to make a velocity field divergence-free and updates the state.

  This is the main high-level function for the pressure step in the solver. It
  orchestrates the entire projection process:
  1. Solves the Poisson equation for the pressure correction `q`.
  2. Updates the total pressure field `p_new = p_old + q`.
  3. Corrects the velocity field by subtracting the gradient of `q`.
  4. Returns the new, complete simulation state with the corrected velocity.

  Args:
    All_variables: The entire current state of the simulation.
    solve: The specific function to use for solving the Poisson equation.

  Returns:
    A new `All_Variables` object with a divergence-free velocity field and
    updated pressure.
  """
  # Unpack the current state variables.
  v = All_variables.velocity
  old_pressure = All_variables.pressure
  particles = All_variables.particles
  Drag =  All_variables.Drag 
  Step_count = All_variables.Step_count
  MD_var = All_variables.MD_var
  
  grid = grids.consistent_grid(*v)
  # Determine the appropriate pressure boundary conditions from the velocity BCs.
  pressure_bc = boundaries.get_pressure_bc_from_velocity(v)

  # `q` represents the pressure *correction* for this time step, not the total pressure.
  # An initial guess `q0` (zero) is created for solvers that might need it.
  q0 = grids.GridArray(jnp.zeros(grid.shape), grid.cell_center, grid)
  q0 = grids.GridVariable(q0, pressure_bc)

  # Step 1: Solve the Poisson equation `∇²q = ∇ ⋅ v`. The `solve` function
  # computes `∇ ⋅ v` internally and returns the data array for `q`.
  qsol_data = solve(v, q0)
  # Wrap the result in a GridVariable.
  q = grids.GridVariable(grids.GridArray(qsol_data, q0.offset, q0.grid), pressure_bc)
    
  # Step 2: Update the total pressure field by adding the correction.
  New_pressure_Array =  grids.GridArray(q.data + old_pressure.data, q.offset, q.grid)  
  New_pressure = grids.GridVariable(New_pressure_Array, pressure_bc) 

  # Step 3: Correct the velocity by subtracting the pressure gradient: v_new = v - ∇q.
  q_grad = fd.forward_difference(q)
  
  # Create the new, divergence-free velocity field.
  v_projected = tuple(
      grids.GridVariable(u.array - q_g, u.bc) for u, q_g in zip(v, q_grad))
      
  # For non-periodic domains, it's good practice to re-impose the BCs on the
  # corrected velocity field to ensure consistency at the boundaries.
  if not boundaries.has_all_periodic_boundary_conditions(*v):
    v_projected = tuple(u.impose_bc() for u in v_projected)
  
  # Step 4: Return the new, updated state container.
  new_variable = particle_class.All_Variables(particles,v_projected,New_pressure,Drag,Step_count,MD_var)
  return new_variable


def solve_fast_diag(
    v: GridVariableVector,
    q0: Optional[GridVariable] = None,
    implementation: Optional[str] = None
) -> GridArray:
  """
  Solves the Poisson equation for fully periodic domains using a Fast Diagonalization solver.

  This is a highly efficient direct (non-iterative) solver that works by
  transforming the problem into Fourier space using the Fast Fourier Transform (FFT).
  In Fourier space, the Laplacian operator `∇²` becomes a simple diagonal matrix of
  its eigenvalues, which makes solving the system `Â * p̂ = b̂` a trivial
  element-wise division: `p̂ = b̂ / Â`.

  Args:
    v: The velocity vector field. The RHS of the Poisson equation, `∇ ⋅ v`, is
      computed internally.
    q0: An initial guess for pressure (unused, kept for API compatibility with
      iterative solvers).
    implementation: A string to select a specific backend implementation for the
      solver if available (unused here).

  Returns:
    A `GridArray` containing the solution for the pressure correction `q`.
  """
  # Mark unused arguments to clarify they are not used in this specific solver.
  del q0
  
  # This solver is only mathematically valid for fully periodic domains.
  if not boundaries.has_all_periodic_boundary_conditions(*v):
    raise ValueError('solve_fast_diag() expects periodic velocity BCs')
  
  # Ensure all velocity components are on the same grid.
  grid = grids.consistent_grid(*v)
  # The Right-Hand Side (RHS) of the Poisson equation is the divergence of the velocity field.
  rhs = fd.divergence(v)
  
  # Get the 1D periodic Laplacian matrices for each dimension of the grid. These
  # matrices represent the `∇²` operator in real space.
  laplacians = list(map(array_utils.laplacian_matrix, grid.shape, grid.step))
  
  # Create the pseudoinverse solver. This pre-computes the FFT plans and the
  # eigenvalues of the Laplacian, which are needed to perform the division in Fourier space.
  pinv = fast_diagonalization.pseudoinverse(
      laplacians, rhs.dtype,
      hermitian=True, circulant=True, implementation=implementation)
      
  # Apply the solver (which performs FFT -> divide -> IFFT) to the RHS.
  return grids.applied(pinv)(rhs)


def solve_fast_diag_moving_wall(
    v: GridVariableVector,
    q0: Optional[GridVariable] = None,
    implementation: Optional[str] = 'matmul'
) -> GridArray:
  """
  Solves the Poisson equation for a channel flow setup (periodic in x, Neumann in y)
  using a specialized Fast Diagonalization solver.

  This solver is a hybrid. It uses the efficient FFT for the periodic dimension(s)
  but must use other fast transforms (like the Discrete Sine/Cosine Transform, which
  are implicitly handled by the diagonalization of the Neumann matrix) or direct
  matrix multiplication for the non-periodic Neumann dimension(s).
  """
  del q0 # Mark unused argument.
  ndim = len(v)

  grid = grids.consistent_grid(*v)
  rhs = fd.divergence(v)
  
  # Create the 1D Laplacian matrices, using the special Neumann matrix for the y-axis (axis 1).
  laplacians = [
      array_utils.laplacian_matrix(grid.shape[0], grid.step[0]),           # Periodic in x
      array_utils.laplacian_matrix_neumann(grid.shape[1], grid.step[1]), # Neumann in y
  ]
  # Add periodic matrices for any higher dimensions (e.g., for a 3D channel flow).
  for d in range(2, ndim):
    laplacians += [array_utils.laplacian_matrix(grid.shape[d], grid.step[d])]
    
  # Create and apply the pseudoinverse solver. `circulant=False` indicates that
  # at least one of the matrices is not circulant (i.e., not periodic), so a
  # pure FFT-based method cannot be used for all dimensions.
  pinv = fast_diagonalization.pseudoinverse(
      laplacians, rhs.dtype,
      hermitian=True, circulant=False, implementation=implementation)
  return grids.applied(pinv)(rhs)
  
  
def solve_fast_diag_Far_Field(
    v: GridVariableVector,
    q0: Optional[GridVariable] = None,
    implementation: Optional[str] = None
) -> GridArray:
  """
  Solves the Poisson equation for domains with arbitrary combinations of
  Neumann and Dirichlet boundary conditions.

  This is the most general of the fast diagonalization solvers. It constructs the
  appropriate 1D Laplacian matrices for each boundary condition type along each axis.
  It also ensures the RHS is transformed correctly for the special case of all-Neumann
  domains to guarantee a solution exists.
  """
  del q0 # Mark unused argument.

  grid = grids.consistent_grid(*v)
  rhs = fd.divergence(v)
  # Infer the correct pressure BCs from the velocity BCs.
  pressure_bc = boundaries.get_pressure_bc_from_velocity(v)
  # Subtract the mean from the RHS if the domain is all-Neumann.
  rhs_transformed_data = _rhs_transform(rhs, pressure_bc)
  rhs_transformed = grids.GridArray(rhs_transformed_data, rhs.offset, rhs.grid)
  
  # The commented out code shows a manual construction of the Laplacian matrices.
  # laplacians = [
  #           laplacian_matrix_neumann(grid.shape[0], grid.step[0]),
  #           laplacian_matrix_neumann(grid.shape[1], grid.step[1]),
  # ]
  
  # This helper function automatically builds the list of 1D Laplacian matrices
  # based on the specific boundary conditions for each axis.
  laplacians = array_utils.laplacian_matrix_w_boundaries(
      rhs.grid, rhs.offset, pressure_bc)
      
  # Create and apply the general-purpose pseudoinverse solver.
  pinv = fast_diagonalization.pseudoinverse(
      laplacians, rhs_transformed.dtype,
      hermitian=True, circulant=False, implementation='matmul')
  return grids.applied(pinv)(rhs_transformed)

def calc_P(
    v: GridVariableVector,
    solve: Callable = solve_fast_diag,
) -> GridVariable:
  """
  Calculates the pressure correction field `q` for a given velocity `v`.

  This is a simplified helper function that just solves the Poisson equation
  without performing the subsequent velocity correction or state update. It can
  be useful for diagnostics or in solver architectures where the pressure
  calculation is separated from the velocity update.

  Args:
    v: The velocity vector field.
    solve: The solver function to use (defaults to the periodic solver).

  Returns:
    A `GridVariable` containing the pressure correction field `q`.
  """
  grid = grids.consistent_grid(*v)
  pressure_bc = boundaries.get_pressure_bc_from_velocity(v)

  # Create a zero initial guess for the pressure correction `q`.
  q0 = grids.GridArray(jnp.zeros(grid.shape), grid.cell_center, grid)
  q0 = grids.GridVariable(q0, pressure_bc)

  # Solve for the pressure correction data using the provided `solve` function.
  q_data = solve(v, q0)
  # Wrap the data in a GridVariable.
  q = grids.GridVariable(grids.GridArray(q_data, q0.offset, q0.grid), pressure_bc)

  return q
