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

Advection (or convection) describes the transport of a quantity, such as heat
or momentum, by the bulk motion of the fluid. This module implements the term
`-(v ⋅ ∇)c`, where `v` is the velocity field and `c` is the quantity being
transported.

The primary approach used is a finite volume method, where the advection is
computed as the negative divergence of a flux (`-∇ ⋅ (vc)`). Different functions
in this module use different interpolation schemes (e.g., linear, upwind,
Van-Leer) to estimate the value of `c` at the control volume faces, leading to
schemes with varying accuracy and stability properties. A semi-Lagrangian
method is also provided as an alternative approach.
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from jax_ib.base import boundaries
from jax_ib.base import finite_differences as fd
from jax_ib.base import grids
from jax_ib.base import interpolation

# Type aliases for clarity
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
InterpolationFn = interpolation.InterpolationFn


def _advect_aligned(cs: GridVariableVector, v: GridVariableVector) -> GridArray:
  """
  Computes advection from pre-aligned velocity and scalar quantities.

  This is a low-level helper function that performs the final step of a finite
  volume calculation. It assumes the scalar `cs` and velocity `v` have already
  been interpolated to the faces of a control volume. It then computes the flux
  `flux = cs * v` and returns its negative divergence.

  Args:
    cs: A sequence of `GridVariable`s representing a scalar `c` that has been
      interpolated to be aligned with each component of `v` at the control
      volume faces.
    v: A sequence of `GridVariable`s for the velocity field, also located at the
      control volume faces.

  Returns:
    A `GridArray` containing the time derivative of `c` due to advection by `v`.
  """
  if len(cs) != len(v):
    raise ValueError('`cs` and `v` must have the same length;'
                     f'got {len(cs)} vs. {len(v)}.')
  
  # Calculate the flux across each face by multiplying the scalar value by the velocity.
  flux_data = tuple(c.array * u.array for c, u in zip(cs, v))

  # The boundary condition for the flux is inherited from the scalar quantity `c`.
  flux = tuple(grids.GridVariable(f, c.bc) for f, c in zip(flux_data, cs))

  # The rate of change due to advection is the negative divergence of the flux.
  # A net outflow of flux (positive divergence) means the quantity inside decreases.
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

  This function implements a finite volume advection scheme. The specific nature
  of the scheme (e.g., linear, upwind) is determined by the interpolation
  functions provided as arguments.

  The procedure is:
    1. Determine the locations of the control volume faces around the scalar `c`.
    2. Interpolate the velocity `v` to these face locations using `u_interpolation_fn`.
    3. Interpolate the scalar `c` to the same face locations using `c_interpolation_fn`.
    4. Compute the final advection term from these aligned quantities.

  Args:
    c: The scalar `GridVariable` to be transported.
    v: The velocity field `GridVariableVector`.
    u_interpolation_fn: The function used to interpolate the velocity `v`.
    c_interpolation_fn: The function used to interpolate the scalar `c`.
    dt: Time step, required by some advanced interpolation schemes. Unused here.

  Returns:
    The time derivative of `c` due to advection by `v`.
  """
  # Get the grid offsets corresponding to the faces of the control volume centered on c.
  target_offsets = grids.control_volume_offsets(c)
  
  # Interpolate each velocity component to the corresponding control volume face.
  aligned_v = tuple(u_interpolation_fn(u, target_offset, v, dt)
                    for u, target_offset in zip(v, target_offsets))
                    
  # Interpolate the scalar quantity `c` to the same control volume faces.
  aligned_c = tuple(c_interpolation_fn(c, target_offset, aligned_v, dt)
                    for target_offset in target_offsets)
                    
  # Pass the aligned velocity and scalar to the helper function to compute the divergence of the flux.
  return _advect_aligned(aligned_c, aligned_v)


def advect_linear(
    c: GridVariable,
    v: GridVariableVector,
    dt: Optional[float] = None
) -> GridArray:
  """Computes advection using linear interpolation for both velocity and scalar.
  This corresponds to a standard second-order central difference scheme. It is
  accurate but can be prone to numerical oscillations.
  """
  return advect_general(c, v, interpolation.linear, interpolation.linear, dt)


def advect_upwind(
    c: GridVariable,
    v: GridVariableVector,
    dt: Optional[float] = None
) -> GridArray:
  """Computes advection using first-order upwind interpolation for the scalar.
  This scheme is very stable and non-oscillatory but is only first-order
  accurate and introduces numerical diffusion (smearing out sharp features).
  """
  return advect_general(c, v, interpolation.linear, interpolation.upwind, dt)


def _align_velocities(v: GridVariableVector) -> Tuple[GridVariableVector]:
  """Pre-interpolates all velocity components needed for the convection term.
  
  For the convection term `(v ⋅ ∇)v`, calculating the advection of each velocity
  component `v_j` requires interpolating all other velocity components `v_i` to
  the faces of `v_j`'s control volume. This function performs all of these
  interpolations upfront to avoid redundant calculations.

  Args:
    v: The velocity field vector.

  Returns:
    A d-tuple of d-tuples of `GridVariable`s `aligned_v`, where `d = len(v)`.
    `aligned_v[i][j]` is component `v[i]` interpolated to the control volume
    faces of component `v[j]`.
  """
  grid = grids.consistent_grid(*v)
  # Get the control volume face offsets for each velocity component.
  offsets = tuple(grids.control_volume_offsets(u) for u in v)
  # Perform the interpolations.
  aligned_v = tuple(
      tuple(interpolation.linear(v[i], offsets[j][i]) # Note the swapped indices i and j
            for i in range(grid.ndim))
      for j in range(grid.ndim))
  return aligned_v


def _velocities_to_flux(
    aligned_v: Tuple[GridVariableVector]
) -> Tuple[GridVariableVector]:
  """Computes the momentum flux tensor `v_i * v_j` from aligned velocities."""
  ndim = len(aligned_v)
  # Initialize a list of empty tuples to hold the rows of the flux tensor.
  flux = [tuple() for _ in range(ndim)]
  for i in range(ndim):
    for j in range(ndim):
      # To avoid duplicate work, only compute for i <= j and reuse for j < i.
      if i <= j:
        # The flux boundary condition is determined by the constituent velocities.
        bc = boundaries.get_pressure_bc_from_velocity(
            (aligned_v[i][j], aligned_v[j][i]))
        flux_component = GridVariable(aligned_v[i][j].array * aligned_v[j][i].array, bc)
        flux[i] += (flux_component,)
      else:
        # Reuse the already computed symmetric component: flux[i][j] = flux[j][i].
        flux[i] += (flux[j][i],)
  return tuple(flux)


def convect_linear(v: GridVariableVector) -> GridArrayVector:
  """
  Computes convection (self-advection) of the velocity field `v`.

  This calculates the term `-(v ⋅ ∇)v`. It is conceptually equivalent to calling
  `advect_linear` for each component of `v`, but it is optimized to avoid
  re-computing shared interpolation terms.

  Args:
    v: The velocity field to be convected.

  Returns:
    A `GridArrayVector` containing the time derivative of each component of `v`
    due to convection.
  """
  # Step 1: Interpolate all velocity components to all necessary face locations.
  aligned_v = _align_velocities(v)
  # Step 2: Compute the momentum flux tensor from the aligned velocities.
  fluxes = _velocities_to_flux(aligned_v)
  # Step 3: The convection term for each component is the negative divergence
  # of the corresponding row of the flux tensor.
  return tuple(-fd.divergence(flux) for flux in fluxes)


def advect_van_leer(
    c: GridVariable,
    v: GridVariableVector,
    dt: float
) -> GridArray:
  """
  Computes advection using the Van-Leer flux-limiter scheme.

  This is a high-resolution scheme that is second-order accurate in smooth
  regions of the flow but reverts to a first-order scheme near sharp gradients
  or shocks. This property makes it Total Variation Diminishing (TVD), meaning
  it does not create new spurious oscillations, making it robust and accurate.

  Args:
    c: The quantity to be transported.
    v: The velocity field.
    dt: The time step, which is needed to compute the Courant number for the limiter.

  Returns:
    The time derivative of `c` due to advection by `v`.
  """
  offsets = grids.control_volume_offsets(c)
  aligned_v = tuple(interpolation.linear(u, offset)
                    for u, offset in zip(v, offsets))
  flux = []
  for axis, (u, h) in enumerate(zip(aligned_v, c.grid.step)):
    # Get values at the center, left, and right of the control volume face.
    c_center = c.data
    c_left = c.shift(-1, axis=axis).data
    c_right = c.shift(+1, axis=axis).data

    # Start with the basic, stable first-order upwind flux.
    upwind_flux = grids.applied(jnp.where)(
        u.array > 0, u.array * c_center, u.array * c_right)

    # --- Compute the Van-Leer high-order flux correction ---
    # The correction term depends on the ratio of successive gradients.
    diffs_prod = 2 * (c_right - c_center) * (c_center - c_left)
    neighbor_diff = c_right - c_left
    
    # Use a safe division to avoid NaNs when the denominator is zero.
    safe = diffs_prod > 0
    forward_correction = jnp.where(
        safe, diffs_prod / jnp.where(safe, neighbor_diff, 1), 0
    )

    # The correction for negative velocity is a shifted version of the positive one.
    forward_correction_array = grids.GridVariable(
        grids.GridArray(forward_correction, u.offset, u.grid), u.bc)
    backward_correction = forward_correction_array.shift(+1, axis).data
    
    abs_velocity = abs(u.array)
    # The Courant number `gamma = |u| * dt / h` determines the weight of the correction.
    courant_numbers = (dt / h) * abs_velocity
    
    # Combine the pieces to get the final flux correction.
    pre_factor = 0.5 * (1 - courant_numbers) * abs_velocity
    flux_correction = pre_factor * grids.applied(jnp.where)(
        u.array > 0, forward_correction, backward_correction)
    
    # The total flux is the upwind flux plus the high-order correction.
    flux.append(upwind_flux + flux_correction)
    
  # Assign boundary conditions and compute the divergence.
  flux = tuple(GridVariable(f, c.bc) for f in flux)
  advection = -fd.divergence(flux)
  return advection


def advect_step_semilagrangian(
    c: GridVariable,
    v: GridVariableVector,
    dt: float
) -> GridVariable:
  """
  Computes one time step of advection using a semi-Lagrangian method.

  Note: Unlike other advection functions in this module, this function returns
  the advected quantity at the *next time step*, NOT the time derivative.

  The method works by tracing the velocity field backwards in time by `dt` from
  each grid point to find its "departure point." The new value at the grid
  point is then set to the interpolated value of `c` at that departure point.
  This method is unconditionally stable with respect to the CFL condition,
  allowing for potentially larger time steps, but it can be more diffusive.

  Args:
    c: The quantity to be transported.
    v: The velocity field.
    dt: The time step to advect over.

  Returns:
    A `GridVariable` containing the advected quantity at time `t + dt`.
  """
  grid = grids.consistent_grid(c, *v)

  if not all(d[0] == 0 for d in grid.domain):
    raise ValueError(
        f'Grid domains currently must start at zero. Found {grid.domain}')
        
  # For each grid point `x`, calculate the departure point `x - v*dt`.
  coords = [x - dt * interpolation.linear(u, c.offset).data
            for x, u in zip(grid.mesh(c.offset), v)]
            
  # Convert the physical coordinates of departure points to fractional grid indices.
  indices = [x / s - o for s, o, x in zip(grid.step, c.offset, coords)]
  
  if not boundaries.has_all_periodic_boundary_conditions(c):
    raise NotImplementedError('non-periodic BCs not yet supported')
    
  # `map_coordinates` interpolates the array `c` at the fractional `indices`.
  # `order=1` specifies linear interpolation. `mode='wrap'` handles periodic BCs.
  c_advected_data = grids.applied(jax.scipy.ndimage.map_coordinates)(
      c.array, indices, order=1, mode='wrap')
      
  return GridVariable(c_advected_data, c.bc)


def advect_van_leer_using_limiters(
    c: GridVariable,
    v: GridVariableVector,
    dt: float
) -> GridArray:
  """
  Implements Van-Leer advection in a modular way by applying a TVD limiter
  to the Lax-Wendroff interpolation scheme. This is a cleaner, more composable
  way to formulate the scheme compared to the direct implementation above.
  """
  # Create a new interpolation function by composing the base scheme with a limiter.
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

  This is based on the Courant-Friedrichs-Lewy (CFL) condition, which requires
  that the fluid does not travel more than one grid cell in a single time step.
  The Courant number is defined as `C = u * dt / dx`. To ensure stability, we
  require `C < max_courant_number`.

  Args:
    max_velocity: The maximum expected velocity in the simulation.
    max_courant_number: The maximum allowable Courant number (a safety factor,
      typically in the range [0.5, 1.0)).
    grid: The `Grid` object for the simulation domain.

  Returns:
    The calculated stable time step `dt`.
  """
  # Find the smallest grid spacing in any dimension.
  dx = min(grid.step)
  # Calculate dt based on the CFL condition: dt = C_max * dx / u_max.
  return max_courant_number * dx / max_velocity
