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
Module for calculating the advection term of the Navier-Stokes equations.

Advection (or convection) describes the transport of a quantity (like momentum
or a scalar concentration) by the bulk motion of the fluid. This module provides
several numerical methods to compute this transport, primarily implementing the
term `-(v ⋅ ∇)c`, where `v` is the velocity and `c` is the advected quantity.

The methods implemented fall into two main categories:

1.  **Eulerian Finite Volume Methods**: These methods compute the advection term
    as the negative divergence of a flux (`-∇ ⋅ (vc)`). They are implemented
    in a modular framework (`advect_general`) where the numerical properties of
    the scheme are determined by the choice of interpolation function used to
    estimate values at control volume faces. The available schemes include:
    -   `advect_linear`: A standard 2nd-order central-difference scheme.
    -   `advect_upwind`: A robust but diffusive 1st-order upwind scheme.
    -   `advect_van_leer`: A high-resolution, non-oscillatory (TVD) scheme that
        uses a flux limiter to achieve 2nd-order accuracy in smooth regions while
        maintaining stability at sharp gradients.

2.  **Semi-Lagrangian Method (`advect_step_semilagrangian`)**: This is a distinct
    approach that calculates the advected field at the next time step, `c(t+dt)`.
    It works by tracing grid points backward in time along velocity streamlines
    to a "departure point" and interpolating the value from that point. This
    method is unconditionally stable but can be more diffusive.

The module also provides an optimized `convect_linear` function specifically for
the self-advection of the velocity field, `-(v ⋅ ∇)v`, which is the nonlinear
convection term in the momentum equation.
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from jax_ib.base import boundaries
from jax_ib.base import finite_differences as fd
from jax_ib.base import grids
from jax_ib.base import interpolation

# --- Type Aliases ---
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
InterpolationFn = interpolation.InterpolationFn
# TODO(dkochkov): This comment suggests considering operator splitting methods
# (like Strang splitting) as an alternative way to solve the advection-diffusion
# equation, which can sometimes be more stable or efficient.


def _advect_aligned(cs: GridVariableVector, v: GridVariableVector) -> GridArray:
  """
  Computes advection from pre-aligned velocity and scalar quantities.

  This is a low-level helper function that performs the final step of a finite
  volume calculation. It assumes the scalar `cs` and velocity `v` have already
  been interpolated to the faces of a control volume. It then computes the flux
  `flux = cs * v` and returns its negative divergence. This represents the net
  outflow of the quantity `c` from the control volume.

  The boundary condition on the intermediate flux variable is inherited from the
  scalar quantity `c`, which is a common and physically reasonable choice.

  Args:
    cs: A `GridVariableVector` where `cs[i]` is the scalar quantity `c`
      interpolated to the same location as the i-th velocity component `v[i]`.
    v: A `GridVariableVector` for the velocity field, with components located
      at the control volume faces.

  Returns:
    A `GridArray` containing the time derivative of `c` due to advection by `v`.
    The result is located at the center of the control volume.
  """
  # TODO(jamieas): This comment suggests that the alignment checks could be made
  # more rigorous to ensure the inputs truly represent values on a valid control volume.
  
  # Basic validation: the scalar and velocity must have the same number of components.
  if len(cs) != len(v):
    raise ValueError('`cs` and `v` must have the same length;'
                     f'got {len(cs)} vs. {len(v)}.')
                     
  # Calculate the flux across each face by multiplying the scalar value by the velocity.
  # This creates a tuple of `GridArray`s, one for each flux component (e.g., F_x, F_y).
  flux_arrays = tuple(c.array * u.array for c, u in zip(cs, v))
  
  # The flux is a physical quantity that also needs boundary conditions to be well-defined.
  # We create a `GridVariableVector` for the flux, inheriting the BC from the scalar `c`.
  flux = tuple(grids.GridVariable(f, c.bc) for f, c in zip(flux_arrays, cs))
  
  # The rate of change due to advection is the negative divergence of the flux.
  # A net outflow of flux (positive divergence) means the quantity `c` inside
  # the control volume must decrease (hence the negative sign).
  return -fd.divergence(flux)


def advect_general(
    c: GridVariable,
    v: GridVariableVector,
    u_interpolation_fn: InterpolationFn,
    c_interpolation_fn: InterpolationFn,
    dt: Optional[float] = None
) -> GridArray:
  """
  A general framework for computing advection using specified interpolation methods.

  This function implements a finite volume advection scheme by orchestrating the
  necessary interpolation and divergence steps. The specific nature of the
  scheme (e.g., linear central difference, upwind, etc.) is determined by the
  interpolation functions that are provided as arguments. This modular design
  makes it easy to experiment with different numerical schemes.

  The procedure is:
    1. Determine the locations of the control volume faces around the scalar `c`.
    2. Interpolate the velocity `v` to these face locations using `u_interpolation_fn`.
    3. Interpolate the scalar `c` to the same face locations using `c_interpolation_fn`.
    4. Pass these aligned quantities to `_advect_aligned` to compute the flux
       and its divergence.

  Args:
    c: The scalar `GridVariable` to be transported.
    v: The velocity field `GridVariableVector`.
    u_interpolation_fn: The function used to interpolate the velocity `v`.
    c_interpolation_fn: The function used to interpolate the scalar `c`.
    dt: The time step, which is required by some advanced interpolation schemes.

  Returns:
    A `GridArray` representing the time derivative of `c` due to advection by `v`.
  """
  # Get the grid offsets corresponding to the faces of the control volume centered on c.
  # For a 2D cell-centered `c`, this would be ((1.0, 0.5), (0.5, 1.0)).
  target_offsets = grids.control_volume_offsets(c)
  
  # Interpolate each velocity component to the corresponding control volume face.
  # On a standard staggered grid, the velocity is often already at the correct
  # face location, in which case this interpolation does nothing.
  aligned_v = tuple(u_interpolation_fn(u, target_offset, v, dt)
                    for u, target_offset in zip(v, target_offsets))
                    
  # Interpolate the scalar quantity `c` from its original location (e.g., cell center)
  # to the control volume faces. This is the key step where the choice of scheme matters.
  aligned_c = tuple(c_interpolation_fn(c, target_offset, aligned_v, dt)
                    for target_offset in target_offsets)
                    
  # Pass the aligned velocity and scalar to the helper function to compute the
  # divergence of the flux, which is the final advection term.
  return _advect_aligned(aligned_c, aligned_v)


def advect_linear(
    c: GridVariable,
    v: GridVariableVector,
    dt: Optional[float] = None
) -> GridArray:
  """
  Computes advection using linear interpolation for both velocity and the scalar.

  This corresponds to a standard second-order central difference scheme for the
  advection term. It is accurate in smooth flows but can be prone to numerical
  oscillations (wiggles) in regions with sharp gradients.
  """
  # This is a convenience wrapper around `advect_general` that pre-fills the
  # interpolation functions with `interpolation.linear`.
  return advect_general(c, v, interpolation.linear, interpolation.linear, dt)


def advect_upwind(
    c: GridVariable,
    v: GridVariableVector,
    dt: Optional[float] = None
) -> GridArray:
  """
  Computes advection using first-order upwind interpolation for the scalar `c`.

  This scheme is very stable and non-oscillatory (monotonic), making it robust.
  However, it is only first-order accurate and introduces significant numerical
  diffusion, which can smear out sharp features in the flow.
  """
  # This wrapper uses linear interpolation for the velocity (a standard choice)
  # and the specialized `interpolation.upwind` for the scalar `c`.
  return advect_general(c, v, interpolation.linear, interpolation.upwind, dt)


def _align_velocities(v: GridVariableVector) -> Tuple[GridVariableVector]:
  """
  Pre-interpolates all velocity components needed for the convection term `(v ⋅ ∇)v`.
  
  Calculating the advection of each velocity component `v_j` by the full velocity
  field `v` requires interpolating all other velocity components `v_i` to the
  faces of `v_j`'s control volume. This function performs all of these
  interpolations upfront to avoid redundant calculations within the main
  convection function.

  Args:
    v: The velocity field `GridVariableVector`.

  Returns:
    A nested tuple `aligned_v` where `aligned_v[i][j]` is the `GridVariable` for
    component `v[i]` interpolated to the `j`-th face of the control volume
    centered around the location of `v[i]`.
  """
  grid = grids.consistent_grid(*v)
  # The commented out line is a remnant of a previous implementation.
  #grid = v[0].grid
  
  # Get the control volume face offsets for each velocity component.
  offsets = tuple(grids.control_volume_offsets(u) for u in v)
  
  # Perform the nested interpolations.
  # The outer loop iterates through the velocity component `v[i]` being interpolated.
  # The inner loop iterates through the target faces `j`.
  aligned_v = tuple(
      tuple(interpolation.linear(v[i], offsets[i][j])
            for j in range(grid.ndim))
      for i in range(grid.ndim))
  return aligned_v


def _velocities_to_flux(
    aligned_v: Tuple[GridVariableVector]
) -> Tuple[GridVariableVector]:
  """
  Computes the momentum flux tensor `v_i * v_j` from aligned velocity components.

  This function calculates the flux associated with the nonlinear convection
  term `(v ⋅ ∇)v`. The `i,j`-th component of this flux tensor represents the
  flux of the `i`-th component of momentum across a face oriented in the `j`-th
  direction. This is calculated as `v_i * v_j`.

  Args:
    aligned_v: A nested tuple of `GridVariable`s produced by `_align_velocities`.
      `aligned_v[i][j]` is the velocity component `v[i]` interpolated to the
      location of the `j`-th flux component.

  Returns:
    A nested tuple `flux` of `GridVariable`s representing the momentum flux tensor.
    The entry `flux[i][j]` is `v_i * v_j`.
  """
  # Get the number of spatial dimensions.
  ndim = len(aligned_v)
  # Initialize a list of empty tuples to hold the rows of the flux tensor.
  flux = [tuple() for _ in range(ndim)]
  
  # Iterate through the components of the flux tensor `F_ij = v_i * v_j`.
  for i in range(ndim):
    for j in range(ndim):
      # The flux tensor is symmetric: `F_ij = F_ji`. To avoid redundant
      # computations, we only calculate the components for `i <= j`.
      if i <= j:
        # The boundary condition for a product of variables should be consistent
        # between the two variables. This function checks for consistency.
        # The commented out line is a remnant of a simpler, less safe assumption.
        bc = grids.consistent_boundary_conditions(
            aligned_v[i][j], aligned_v[j][i])
        
        # Calculate the flux component `v_i * v_j` and wrap it in a GridVariable.
        flux_component = GridVariable(
            aligned_v[i][j].array * aligned_v[j][i].array, bc
        )
        # Add the computed component to the `i`-th row of the flux tensor.
        flux[i] += (flux_component,)
      else:
        # For the lower triangle of the tensor (`i > j`), we reuse the
        # already computed symmetric component: `flux[i][j] = flux[j][i]`.
        flux[i] += (flux[j][i],)
        
  return tuple(flux)


def convect_linear(v: GridVariableVector) -> GridArrayVector:
  """
  Computes convection (self-advection) of the velocity field `v`.

  This function calculates the convection term `-(v ⋅ ∇)v` for the momentum
  equation. It is conceptually equivalent to calling `advect_linear` for each
  component of `v` being advected by the full velocity field `v`.

  However, this implementation is optimized. It first pre-calculates all the
  necessary interpolated velocity components using `_align_velocities` and then
  computes the full momentum flux tensor, avoiding redundant calculations that
  would occur if `advect_linear` were called in a simple loop.

  Args:
    v: The `GridVariableVector` representing the velocity field.

  Returns:
    A `GridArrayVector` containing the time derivative of each component of `v`
    due to convection.
  """
  # TODO(jamieas): These comments suggest potential future improvements, such as
  # further vectorization (perhaps using `jax.vmap`) or extending the physics
  # to handle variable fluid density.
  
  # Step 1: Interpolate all velocity components to all necessary face locations.
  aligned_v = _align_velocities(v)
  
  # Step 2: Compute the full momentum flux tensor `v_i * v_j` from the aligned velocities.
  fluxes = _velocities_to_flux(aligned_v)
  
  # Step 3: The convection term for each component `v_i` is the negative
  # divergence of the corresponding row of the flux tensor, `∇ ⋅ (v_i * v)`.
  return tuple(-fd.divergence(flux_row) for flux_row in fluxes)


def advect_van_leer(
    c: GridVariable,
    v: GridVariableVector,
    dt: float
) -> GridArray:
  """
  Computes advection using the Van-Leer flux-limiter scheme.

  This is a high-resolution scheme that is second-order accurate in smooth
  regions of the flow but reverts to a first-order upwind scheme near sharp
  gradients or shocks to prevent spurious oscillations. This property makes it
  Total Variation Diminishing (TVD), meaning it is robust and accurate.

  The method works by calculating a stable, first-order (upwind) flux and then
  adding a limited "anti-diffusive" flux correction to restore second-order
  accuracy where the solution is smooth.

  Args:
    c: The `GridVariable` representing the quantity to be transported.
    v: A `GridVariableVector` for the velocity field.
    dt: The time step, which is needed to compute the Courant number for the limiter.

  Returns:
    A `GridArray` containing the time derivative of `c` due to advection.
  """
  # TODO(dkochkov): This comment suggests a good refactoring: this direct
  # implementation could be replaced by the more modular `apply_tvd_limiter`
  # approach for better code reuse and clarity.
  
  # Step 1: Interpolate velocity `v` to the control volume faces of `c`.
  offsets = grids.control_volume_offsets(c)
  aligned_v = tuple(interpolation.linear(u, offset)
                    for u, offset in zip(v, offsets))
                    
  flux = []
  # Iterate through each spatial dimension to compute the flux component for that axis.
  for axis, (u, h) in enumerate(zip(aligned_v, c.grid.step)):
    # Get the stencil of values for `c` needed to compute the flux at the faces.
    c_center = c.data                 # c at grid point i
    c_left = c.shift(-1, axis=axis).data  # c at grid point i-1
    c_right = c.shift(+1, axis=axis).data # c at grid point i+1
    
    # Step 2: Compute the basic, stable first-order upwind flux.
    # If velocity `u` is positive, the flux is `u * c_center`; if negative, it's `u * c_right`.
    upwind_flux = grids.applied(jnp.where)(
        u.array > 0, u.array * c_center, u.array * c_right)

    # Step 3: Compute the higher-order flux correction term.
    # The comments below explain the formula for the flux correction `df`.
    # It depends on the local Courant number `gamma` and a limited estimate
    # of the second derivative `dc`.
    
    # First, calculate the unlimited second-derivative term, `dc`.
    # This term `2 * (c_i+1 - c_i) * (c_i - c_{i-1}) / (c_{i+1} - c_{i-1})` can be
    # unstable, so it needs to be limited.
    diffs_prod = 2 * (c_right - c_center) * (c_center - c_left)
    neighbor_diff = c_right - c_left
    
    # The correction is only applied where the gradients have the same sign
    # (`diffs_prod > 0`), which indicates a smooth region. Where gradients
    # change sign (an extremum), the correction is zeroed out to prevent oscillations.
    safe = diffs_prod > 0
    # Use `safe_div` (via `jnp.where`) to avoid division by zero.
    forward_correction = jnp.where(
        safe, diffs_prod / jnp.where(safe, neighbor_diff, 1), 0
    )
    
    # The correction for negative velocity uses a stencil shifted one cell to the right.
    forward_correction_array = grids.GridVariable(
        grids.GridArray(forward_correction, u.offset, u.grid), u.bc)
    backward_correction_array = forward_correction_array.shift(+1, axis)
    backward_correction = backward_correction_array.data
    
    # Calculate the pre-factor for the correction, which includes the Courant number.
    abs_velocity = abs(u.array)
    courant_numbers = (dt / h) * abs_velocity # `gamma = |u| * dt / h`
    pre_factor = 0.5 * (1 - courant_numbers) * abs_velocity
    
    # Select the correct correction term based on the velocity direction.
    flux_correction = pre_factor * grids.applied(jnp.where)(
        u.array > 0, forward_correction, backward_correction)
        
    # Step 4: The total flux is the sum of the upwind flux and the limited correction.
    flux.append(upwind_flux + flux_correction)
    
  # Step 5: Assign boundary conditions to the final flux vector and compute its divergence.
  flux_gv = tuple(GridVariable(f, c.bc) for f in flux)
  advection = -fd.divergence(flux_gv)
  
  return advection


def advect_step_semilagrangian(
    c: GridVariable,
    v: GridVariableVector,
    dt: float
) -> GridVariable:
  """
  Computes one time step of advection using a semi-Lagrangian method.

  Note: Unlike the other advection functions in this module, this function returns
  the advected quantity at the *next time step* (`c(t + dt)`), NOT the time derivative (`dc/dt`).

  The method works by tracing the velocity field backwards in time from each
  grid point `x` to find its "departure point," `x_d = x - v*dt`. The new value at
  the grid point, `c(x, t + dt)`, is then set to the interpolated value of the
  field at that departure point, `c(x_d, t)`.

  This method is unconditionally stable with respect to the CFL condition related
  to advection, allowing for potentially larger time steps than Eulerian methods.
  However, it can be more numerically diffusive (less accurate) and does not
  inherently conserve the advected quantity.

  Args:
    c: The `GridVariable` to be transported.
    v: The `GridVariableVector` representing the velocity field.
    dt: The time step to advect over.

  Returns:
    A `GridVariable` containing the advected quantity at time `t + dt`.
  """
  # Ensure all variables are on the same grid.
  grid = grids.consistent_grid(c, *v)

  # TODO(shoyer): This comment indicates a current limitation and provides a
  # hint for how to generalize the coordinate-to-index mapping for domains
  # that do not start at zero.
  if not all(d[0] == 0 for d in grid.domain):
    raise ValueError(
        f'Grid domains currently must start at zero. Found {grid.domain}')
        
  # Step 1: For each grid point `x` where `c` is defined, calculate the
  # departure point `x_d = x - v*dt`.
  # First, interpolate the velocity `v` to the location of `c`.
  v_at_c = [interpolation.linear(u, c.offset).data for u in v]
  # Then, trace back from each grid point `x` in the mesh.
  coords = [x - dt * u for x, u in zip(grid.mesh(c.offset), v_at_c)]
            
  # Step 2: Convert the physical coordinates of the departure points into
  # fractional grid indices, which is the format required by `map_coordinates`.
  indices = [x / s - o for s, o, x in zip(grid.step, c.offset, coords)]
  
  # This implementation currently only supports periodic boundary conditions.
  if not boundaries.has_all_periodic_boundary_conditions(c):
    raise NotImplementedError('non-periodic BCs not yet supported for semi-lagrangian')
    
  # Step 3: Interpolate the original data `c` at the fractional `indices`.
  # `jax.scipy.ndimage.map_coordinates` performs multi-dimensional spline interpolation.
  # `order=1` specifies linear interpolation. `mode='wrap'` handles the periodic BCs.
  c_advected_data = grids.applied(jax.scipy.ndimage.map_coordinates)(
      c.array, indices, order=1, mode='wrap')
      
  # Wrap the new data in a GridVariable and return it.
  return GridVariable(c_advected, c.bc)


# TODO(dkochkov): These comments are notes for future development.
# `advect_with_flux_limiter` would be a more general version of the function below.
# Moving `advect_van_leer` to a test file suggests it might be superseded by
# this more modular `_using_limiters` version.
def advect_van_leer_using_limiters(
    c: GridVariable,
    v: GridVariableVector,
    dt: float
) -> GridArray:
  """
  Implements Van-Leer advection in a modular way by applying a TVD limiter.

  This function achieves the same physical result as `advect_van_leer`, but in a
  more composable and extensible way. It uses the `apply_tvd_limiter` factory
  from the `interpolation` module to create a new, limited interpolation scheme
  on the fly. This new scheme is then passed to the `advect_general` framework.

  This approach is powerful because you could easily swap `van_leer_limiter` for
  another limiter (e.g., minmod, superbee) to create a different TVD scheme
  without rewriting the advection logic.
  """
  # Create a new, composite interpolation function by applying the Van Leer limiter
  # to the high-order Lax-Wendroff scheme.
  c_interpolation_fn = interpolation.apply_tvd_limiter(
      interpolation.lax_wendroff, limiter=interpolation.van_leer_limiter)
      
  # Use this new composite interpolation function in the general advection framework.
  return advect_general(c, v, interpolation.linear, c_interpolation_fn, dt)


def stable_time_step(
    max_velocity: float,
    max_courant_number: float,
    grid: grids.Grid
) -> float:
  """
  Calculates a stable time step size for explicit advection schemes.

  This is based on the Courant-Friedrichs-Lewy (CFL) condition, which is a
  necessary condition for the stability of many explicit time-stepping schemes
  for hyperbolic PDEs like the advection equation. It requires that the numerical
  domain of dependence contains the true physical domain of dependence.
  
  In practice, it means that information (like a fluid particle) should not
  travel more than one grid cell in a single time step. The Courant number is
  defined as `C = u * dt / dx`. To ensure stability, we require `C <= C_max`.

  Args:
    max_velocity: The maximum expected velocity magnitude (`u_max`) in the simulation.
    max_courant_number: The maximum allowable Courant number (`C_max`), which acts
      as a safety factor. It is typically in the range `[0.5, 1.0)`.
    grid: The `Grid` object for the simulation domain.

  Returns:
    The calculated stable time step `dt`, derived from `dt = C_max * dx / u_max`.
  """
  # The stability constraint is most restrictive for the smallest grid spacing.
  dx = min(grid.step)
  # Avoid division by zero if velocity is zero.
  if max_velocity == 0:
      return float('inf')
  # Calculate dt based on the CFL condition.
  return max_courant_number * dx / max_velocity
