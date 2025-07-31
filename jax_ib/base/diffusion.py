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

# TODO(pnorgaard): The suggestion to implement the Biconjugate Gradient Stabilized (BiCGSTAB)
# method is a good one. BiCGSTAB is an iterative solver for linear systems that, unlike
# the Conjugate Gradient (CG) method, does not require the system matrix to be symmetric.
# This would allow for solving diffusion problems with more complex, non-symmetric
# discretizations or boundary conditions.

"""
Module for computing the diffusion term in the Navier-Stokes equations.

Diffusion describes the transport of momentum due to random molecular motion,
which manifests as viscosity in a fluid. It acts as an internal friction that
smooths out velocity gradients over time. This module implements the diffusion
term, `ν ∇²v`, and provides methods for its numerical solution.

The functions provided can be categorized as follows:

1.  **Explicit Diffusion (`diffuse`)**: This function calculates the diffusion
    term `ν * ∇²v` directly using a finite difference approximation of the
    Laplacian. It is intended for use in explicit time-stepping schemes (e.g.,
    Forward Euler). While simple to implement, these schemes are subject to a
    strict time step constraint (`stable_time_step`) for numerical stability.

2.  **Implicit Diffusion Solvers**: These functions solve the diffusion equation
    for the velocity at the *next* time step, `v_new`. This is required for
    implicit schemes (e.g., Backward Euler), which are unconditionally stable
    and allow for much larger time steps. Solving the implicit equation requires
    solving a large linear system of equations, `(I - ν*dt*∇²) v_new = v_old`.
    Two methods are provided:
    -   `solve_cg`: An iterative solver using the Conjugate Gradient method.
        It is memory-efficient and can handle complex boundary conditions, but
        convergence can be slow.
    -   `solve_fast_diag`: A direct solver that uses the Fast Fourier Transform
        (FFT) to diagonalize the Laplacian operator. It is extremely fast and
        accurate but is restricted to periodic domains.
"""

from typing import Optional

import jax.scipy.sparse.linalg

# Import necessary components from other modules in the library.
from jax_cfd.base import array_utils
from jax_ib.base import boundaries
from jax_cfd.base import fast_diagonalization
from jax_ib.base import finite_differences as fd
from jax_ib.base import grids

# --- Type Aliases ---
Array = grids.Array
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector


def diffuse(c: GridVariable, nu: float) -> GridArray:
  """
  Computes the rate of change of a quantity `c` due to explicit diffusion.

  This function calculates the diffusion term `ν * ∇²c` for use in an explicit
  time-stepping scheme (like Forward Euler). It represents the tendency of a
  concentrated quantity to spread out over time.

  Args:
    c: The `GridVariable` representing the quantity to be diffused (e.g., a
      velocity component).
    nu: A float representing the kinematic viscosity (or diffusivity coefficient).

  Returns:
    A `GridArray` containing the result of the diffusion operation.
  """
  # The diffusion term is simply the Laplacian of the quantity `c` (computed
  # using finite differences) scaled by the kinematic viscosity `nu`.
  return nu * fd.laplacian(c)


def stable_time_step(viscosity: float, grid: grids.Grid) -> float:
  """
  Calculates a stable time step size for an explicit diffusion scheme.

  For explicit time-stepping methods (like Forward Euler or Runge-Kutta), the
  diffusion term is subject to a strict stability constraint. If the time step
  `dt` is too large, the numerical solution can become unstable and "blow up."
  This constraint, derived from von Neumann stability analysis of the Forward-Time
  Central-Space (FTCS) scheme, is typically given by:
  `dt <= dx² / (2 * D * nu)`, where D is the number of dimensions.

  This function calculates the maximum `dt` that satisfies this condition. Using
  a smaller `dt` is always safe.

  Args:
    viscosity: The kinematic viscosity `nu`.
    grid: The `Grid` object for the simulation domain.

  Returns:
    A float representing the largest stable time step.
  """
  # If there is no viscosity, the diffusion term is zero, and it imposes no stability limit.
  if viscosity == 0:
    return float('inf')
    
  # The stability constraint is most restrictive for the smallest grid spacing.
  dx = min(grid.step)
  ndim = grid.ndim
  
  # The formula for the time step limit. The denominator includes a factor of 2
  # for each spatial dimension of the problem.
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

  For an implicit time-stepping scheme (like Backward Euler), the diffusion
  update requires solving a linear system of equations of the form `A * x = b`,
  where `x` is the velocity at the new time step `v_new`, `b` is the velocity
  at the current time step `v_old`, and `A` is the linear operator `(I - ν*dt*∇²)`.

  The CG method is an iterative algorithm that is well-suited for solving large,
  sparse, symmetric positive-definite linear systems, which is the case for the
  discretized Laplacian with periodic boundary conditions.

  NOTE: This implementation is restricted to fully periodic boundary conditions.

  Args:
    v: The `GridVariableVector` of velocity at the current time step (`v_old`).
    nu: The kinematic viscosity.
    dt: The time step.
    rtol: Relative tolerance for the iterative solver to determine convergence.
    atol: Absolute tolerance for the iterative solver.
    maxiter: Maximum number of iterations for the solver.

  Returns:
    A `GridVariableVector` of the velocity at the next time step (`v_new`).
  """
  # The CG method requires a symmetric positive-definite matrix, which is only
  # guaranteed here for fully periodic boundary conditions.
  if not boundaries.has_all_periodic_boundary_conditions(*v):
    raise ValueError('solve_cg() expects periodic BC')

  def solve_component(u: GridVariable) -> GridArray:
    """
    Solves the implicit diffusion equation `(I - ν*dt*∇²) u_new = u_old`
    for a single velocity component `u`.
    """

    # This function defines the linear operator A = (I - ν*dt*∇²).
    # It takes a potential solution `u_new` and returns `A * u_new`.
    def linear_op(u_new_data: GridArray) -> GridArray:
      """The linear operator A applied to a candidate solution vector."""
      # Wrap the raw array data in a GridVariable to correctly apply boundary
      # conditions during the Laplacian calculation.
      u_new_var = grids.GridVariable(u_new_data, u.bc)
      # Calculate A*u_new = u_new - dt*ν*laplacian(u_new).
      return u_new_var.array.data - dt * nu * fd.laplacian(u_new_var).data

    # Use JAX's built-in conjugate gradient solver.
    # It solves `linear_op(x) = b` for `x`.
    # `b` is the RHS, which is the velocity at the current time step (`u.array`).
    # `x0` is the initial guess, which is also set to the current velocity.
    x, _ = jax.scipy.sparse.linalg.cg(
        linear_op, u.array.data, x0=u.array.data, tol=rtol, atol=atol, maxiter=maxiter)
    return grids.GridArray(x, u.offset, u.grid)

  # Apply the solver independently to each component of the velocity vector `v`.
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
  operator `∇²` on a periodic domain can be diagonalized by the Fourier transform.
  Solving the system `(I - ν*dt*∇²) v_new = v_old` in Fourier space becomes a
  simple element-wise division: `v̂_new = v̂_old / (1 - ν*dt*λ)`, where `λ` are
  the eigenvalues of the Laplacian.

  Args:
    v: The `GridVariableVector` of velocity at the current time step (`v_old`).
    nu: The kinematic viscosity.
    dt: The time step.
    implementation: Specific backend for the diagonalization solver if needed.

  Returns:
    A `GridVariableVector` of the velocity at the next time step (`v_new`).
  """
  # This solver is only mathematically valid for fully periodic domains
  # where the FFT is applicable.
  if not boundaries.has_all_periodic_boundary_conditions(*v):
    raise ValueError('solve_fast_diag() expects periodic BC')
    
  grid = grids.consistent_grid(*v)
  # Get the 1D Laplacian matrices for each dimension of the grid. Their
  # eigenvalues are used to construct the solver.
  laplacians = list(map(array_utils.laplacian_matrix, grid.shape, grid.step))

  # In Fourier space, the eigenvalues of the solution operator `(I - ν*dt*∇²)^-1`
  # are `1 / (1 - ν*dt*λ)`.
  # This function transforms the eigenvalues of `∇²` (represented by `x`) to the
  # eigenvalues of the operator `(I - ν*dt*∇²)^-1 - I`.
  def func(eigenvalues_of_laplacian):
    x = eigenvalues_of_laplacian
    dt_nu_x = dt * nu * x
    # This is the transformation of eigenvalues: `λ' = (ν*dt*λ) / (1 - ν*dt*λ)`.
    return dt_nu_x / (1 - dt_nu_x)

  # Create the fast diagonalization solver object. This pre-computes the
  # FFT plans and the transformed eigenvalues.
  op = fast_diagonalization.transform(
      func, laplacians, v[0].dtype,
      hermitian=True, circulant=True, implementation=implementation)

  # The solution `v_new` is given by `(I - ν*dt*∇²)^-1 * v_old`.
  # For better numerical stability when `ν*dt` is small, this is rewritten as:
  # `v_new = v_old + ((I - ν*dt*∇²)^-1 - I) * v_old`.
  # The `op` we created calculates the `((...)^-1 - I)` part.
  return tuple(grids.GridVariable(u.array + grids.applied(op)(u.array), u.bc)
               for u in v)
