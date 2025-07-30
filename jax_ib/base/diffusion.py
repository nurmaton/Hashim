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
Module for functionality related to the diffusion term in the Navier-Stokes eq.

Diffusion describes the process by which momentum is transported due to random
molecular motion. In fluid dynamics, this is manifested as viscosity, which acts
as an internal friction that smooths out velocity gradients.

This module implements the diffusion term, `ν ∇²v`, and provides multiple methods
for solving the diffusion equation, which is required for implicit time-stepping
schemes. These methods include iterative solvers (`solve_cg`) and direct solvers
based on matrix diagonalization (`solve_fast_diag`).
"""
from typing import Optional

import jax.scipy.sparse.linalg
from jax_ib.base import array_utils
from jax_ib.base import boundaries
from jax_ib.base import fast_diagonalization
from jax_ib.base import finite_differences as fd
from jax_ib.base import grids

# Type aliases for clarity
Array = grids.Array
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector


def diffuse(c: GridVariable, nu: float) -> GridArray:
  """
  Computes the rate of change of a quantity `c` due to explicit diffusion.

  This function calculates the diffusion term `ν * ∇²c` for use in an explicit
  time-stepping scheme.

  Args:
    c: The `GridVariable` representing the quantity to be diffused (e.g., a
      velocity component).
    nu: A float representing the kinematic viscosity.

  Returns:
    A `GridArray` containing the result of the diffusion operation.
  """
  # The diffusion term is simply the Laplacian of the quantity `c` scaled by
  # the kinematic viscosity `nu`.
  return nu * fd.laplacian(c)


def stable_time_step(viscosity: float, grid: grids.Grid) -> float:
  """
  Calculates a stable time step size for an explicit diffusion scheme.

  For explicit time-stepping methods, the diffusion term is subject to a strict
  stability constraint: `dt <= dx² / (2 * D * nu)`, where D is the number of
  dimensions. If the time step is too large, the numerical solution can become
  unstable. This function calculates the maximum `dt` that satisfies this.

  Args:
    viscosity: The kinematic viscosity `nu`.
    grid: The `Grid` object for the simulation domain.

  Returns:
    The calculated maximum stable time step `dt`.
  """
  if viscosity == 0:
    return float('inf') # If there is no diffusion, there is no stability limit from it.
  # Find the smallest grid spacing in any dimension.
  dx = min(grid.step)
  ndim = grid.ndim
  # The denominator includes a factor of 2 for each dimension.
  return dx ** 2 / (viscosity * 2 ** ndim)


def solve_cg(
    v: GridVariableVector,
    nu: float,
    dt: float,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    maxiter: Optional[int] = None
) -> GridVariableVector:
  """
  Solves the implicit diffusion equation using the Conjugate Gradient (CG) method.

  For an implicit scheme (like Backward Euler), we must solve the linear system:
  `(I - ν*dt*∇²) v_new = v_old`
  
  The CG method is an iterative algorithm that is well-suited for solving large,
  sparse, symmetric positive-definite linear systems, which is the case here for
  periodic domains.

  NOTE: This implementation currently requires periodic boundary conditions.

  Args:
    v: The `GridVariableVector` of velocity at the current time step.
    nu: The kinematic viscosity.
    dt: The time step.
    rtol: Relative tolerance for the iterative solver to determine convergence.
    atol: Absolute tolerance for the iterative solver.
    maxiter: Maximum number of iterations for the solver.

  Returns:
    A `GridVariableVector` of the velocity at the next time step.
  """
  if not boundaries.has_all_periodic_boundary_conditions(*v):
    raise ValueError('solve_cg() expects periodic boundary conditions')

  def solve_component(u: GridVariable) -> GridArray:
    """Solves the implicit diffusion equation for a single velocity component."""

    # This function defines the linear operator A in the system Ax=b,
    # where A = (I - ν*dt*∇²).
    def linear_op(u_new_data: GridArray) -> GridArray:
      """The linear operator A = (I - ν*dt*∇²) applied to a vector."""
      # Wrap the raw array data in a GridVariable to apply boundary conditions.
      u_new_var = grids.GridVariable(u_new_data, u.bc)
      # Calculate A*u_new = u_new - dt*ν*laplacian(u_new).
      return u_new_var.array - dt * nu * fd.laplacian(u_new_var)

    # Use JAX's built-in conjugate gradient solver.
    # It solves `linear_op(x) = u.array` for `x`, with `u.array` as the initial guess.
    x, _ = jax.scipy.sparse.linalg.cg(
        linear_op, u.array, x0=u.array, tol=rtol, atol=atol, maxiter=maxiter)
    return x

  # Apply the solver to each component of the velocity vector `v`.
  return tuple(grids.GridVariable(solve_component(u), u.bc) for u in v)


def solve_fast_diag(
    v: GridVariableVector,
    nu: float,
    dt: float,
    implementation: Optional[str] = None
) -> GridVariableVector:
  """
  Solves the implicit diffusion equation using a fast diagonalization method.

  This method is a direct (non-iterative) solver that is highly efficient for
  problems with periodic boundary conditions. It works because the Laplacian
  operator on a periodic domain can be diagonalized by the Fourier transform.
  Solving the system in Fourier space becomes a simple element-wise division.

  The actual implementation uses pre-computed eigenvectors of the Laplacian to
  transform the problem, solve it, and transform back.

  Args:
    v: The `GridVariableVector` of velocity at the current time step.
    nu: The kinematic viscosity.
    dt: The time step.
    implementation: Specific backend for the diagonalization solver if needed.

  Returns:
    A `GridVariableVector` of the velocity at the next time step.
  """
  if not boundaries.has_all_periodic_boundary_conditions(*v):
    raise ValueError('solve_fast_diag() expects periodic boundary conditions')
  grid = grids.consistent_grid(*v)

  # Get the 1D Laplacian matrices for each dimension of the grid.
  laplacians = list(map(array_utils.laplacian_matrix, grid.shape, grid.step))

  # In Fourier space, the operator `(I - ν*dt*∇²)^-1` corresponds to division by
  # `(1 - ν*dt*λ)`, where λ are the eigenvalues of the Laplacian.
  # This function transforms the eigenvalues of `∇²` to the eigenvalues of
  # the full solution operator `(I - ν*dt*∇²)^-1 - I`.
  def func(eigenvalues_of_laplacian):
    x = eigenvalues_of_laplacian
    dt_nu_x = dt * nu * x
    # This is the transformation of eigenvalues. It computes `λ' = (ν*dt*λ) / (1 - ν*dt*λ)`.
    return dt_nu_x / (1 - dt_nu_x)

  # Create the fast diagonalization solver object. This pre-computes the
  # necessary transforms.
  op = fast_diagonalization.transform(
      func, laplacians, v[0].dtype,
      hermitian=True, circulant=True, implementation=implementation)

  # The solution v_new is given by `(I - ν*dt*∇²)^-1 * v_old`.
  # This can be rewritten as `v_old + ((I - ν*dt*∇²)^-1 - I) * v_old`.
  # The `op` we created calculates the `((...)^-1 - I)` part.
  # This formulation is more numerically stable when `ν*dt` is small.
  return tuple(grids.GridVariable(u.array + grids.applied(op)(u.array), u.bc)
               for u in v)
