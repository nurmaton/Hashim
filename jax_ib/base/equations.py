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
Assembles the components of the Navier-Stokes equations into a complete
time-stepping function for Fluid-Structure Interaction (FSI) simulations.

This module provides high-level "factory" functions that construct a complete
stepper for solving the incompressible Navier-Stokes equations coupled with an
immersed, deformable body. It combines various physical terms (advection,
diffusion), numerical procedures (pressure projection), and specialized FSI
components (IBM forcing, particle motion) into a single, callable function
that advances the entire simulation state by one time step.

The primary function, `semi_implicit_navier_stokes_timeBC`, builds a solver
that uses a semi-implicit projection method, a robust and standard algorithm
for FSI, which proceeds in the following sequence:

1.  **Explicit Step**: An intermediate fluid velocity is computed by advancing
    the standard advection and diffusion terms explicitly in time.
2.  **IBM Forcing**: The physical forces from the immersed boundary (e.g.,
    elasticity, surface tension) are calculated and "spread" to the fluid grid.
3.  **Pressure Projection**: A Poisson equation is solved for the pressure field
    required to enforce the incompressibility constraint on the velocity field.
4.  **Velocity Correction**: The intermediate velocity is corrected with the
    pressure gradient to yield a final, divergence-free velocity field.
5.  **Particle Motion**: The final fluid velocity is interpolated back to the
    immersed boundary to update its position for the next time step.

The use of a factory pattern allows for great flexibility, enabling users to
easily swap out different numerical methods (e.g., different advection schemes,
pressure solvers, or time integrators) to configure the solver.
"""

import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp

# Import all the necessary building blocks from other modules in the library.
from jax_ib.base import advection
from jax_ib.base import diffusion
from jax_ib.base import grids
from jax_ib.base import pressure
from jax_cfd.base import pressure as pressureCFD # For potential re-use from jax_cfd.
from jax_ib.base import time_stepping
from jax_ib.base import boundaries
from jax_ib.base import finite_differences
import tree_math # A utility for working with PyTrees.
from jax_ib.base import particle_class
from jax_cfd.base import equations as equationsCFD # For potential re-use from jax_cfd.

# --- Type Aliases ---
# Using type hints for function types makes the code's intent clearer.
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
# A function that computes the convection term: `(v) -> dv/dt`.
ConvectFn = Callable[[GridVariableVector], GridArrayVector]
# A function that computes the diffusion term: `(c, nu) -> dc/dt`.
DiffuseFn = Callable[[GridVariable, float], GridArray]
# A function for an arbitrary external forcing term.
ForcingFn = Callable[[GridVariableVector], GridArrayVector]
# A function to update boundary conditions, taking the full state and a time step.
BCFn =  Callable[[particle_class.All_Variables, float], particle_class.All_Variables]
# A function to update boundary conditions, taking just velocity and a time step.
BCFn_new =  Callable[[GridVariableVector, float], GridVariableVector]
# A function to calculate the Immersed Boundary Method force.
IBMFn =  Callable[[particle_class.All_Variables, float], GridVariableVector]
# A function to calculate the pressure gradient.
GradPFn = Callable[[GridVariable], GridArrayVector]
# A function to update the particle's position.
PosFn =  Callable[[particle_class.All_Variables, float], particle_class.All_Variables]
# A function to calculate drag or other diagnostic quantities.
DragFn =  Callable[[particle_class.All_Variables], particle_class.All_Variables]


def _wrap_term_as_vector(fun, *, name):
  """
  A wrapper to apply a scalar function to each component of a vector.
  Also wraps the function call with `jax.named_call` for easier profiling and
  debugging using tools like the JAX profiler. `vector_argnums=0` tells
  `tree_math` that the first argument to the function is the vector to be unzipped.
  """
  return tree_math.unwrap(jax.named_call(fun, name=name), vector_argnums=0)


def navier_stokes_explicit_terms(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffuse: DiffuseFn = diffusion.diffuse,
    forcing: Optional[ForcingFn] = None,
    
) -> Callable[[GridVariableVector], GridVariableVector]:
  """
  Factory that returns a function for the explicit terms of the Navier-Stokes equation.
  
  This combines advection, diffusion, and any external forcing into a single
  function that calculates the rate of change of velocity, *excluding* the
  pressure gradient term. This represents the Right-Hand-Side (RHS) of the
  momentum equation before the pressure projection step:
  `dv/dt = -(v ⋅ ∇)v + ν∇²v + f`.

  Args:
    density: Fluid density (ρ).
    viscosity: Fluid dynamic viscosity (μ). Note: the diffuse function expects
      kinematic viscosity (ν = μ/ρ), and this conversion is handled internally.
    dt: Time step, needed for some advection schemes like Van Leer.
    grid: The simulation grid.
    convect: A function to compute the convection term. Defaults to Van Leer.
    diffuse: A function to compute the diffusion term.
    forcing: An optional function for external forces (e.g., gravity).

  Returns:
    A function that takes a velocity vector and returns its time derivative
    due to the explicit terms.
  """
  # The grid argument is currently unused in this function's logic but is
  # kept for API consistency with other potential implementations.
  del grid

  # If no specific convection function is provided, set a robust default.
  if convect is None:
    # `advect_van_leer_using_limiters` is a good, general-purpose choice that
    # balances second-order accuracy in smooth regions with stability near shocks.
    def convect(v):  # pylint: disable=function-redefined
      # This applies the advection function to each component of the velocity vector.
      return tuple(
          advection.advect_van_leer_using_limiters(u, v, dt) for u in v)

  # Adapt the scalar diffusion function (which acts on one GridVariable) to
  # work on a vector of velocities by applying it to each component.
  def diffuse_velocity(v, *args):
    return tuple(diffuse(u, *args) for u in v)

  # Use the wrapper to prepare the component functions for vector inputs and profiling.
  convection = _wrap_term_as_vector(convect, name='convection')
  diffusion_ = _wrap_term_as_vector(diffuse_velocity, name='diffusion')
  if forcing is not None:
    forcing = _wrap_term_as_vector(forcing, name='forcing')

  # This is the core function that will be returned by the factory.
  @tree_math.wrap # Allows the function to work seamlessly with PyTrees (like GridVariableVector).
  @functools.partial(jax.named_call, name='navier_stokes_momentum')
  def _explicit_terms(v):
    """Computes the sum of all explicit momentum terms."""
    # Start with the convection term -(v ⋅ ∇)v.
    dv_dt = convection(v)
    # Add diffusion term ν∇²v, where ν = viscosity / density.
    if viscosity is not None:
      dv_dt += diffusion_(v, viscosity / density)
    # Add external forcing term f/ρ.
    if forcing is not None:
      dv_dt += forcing(v) / density
    
    return dv_dt

  def explicit_terms_with_same_bcs(v):
    """
    A final wrapper to ensure the output GridVariables have the same BCs as the input.
    The result of calculus operations is a GridArray; this function puts the
    boundary conditions back on to create a valid GridVariableVector.
    """
    dv_dt_arrays = _explicit_terms(v)
    # Re-wrap each resulting GridArray with the boundary condition from the
    # corresponding input velocity component.
    return tuple(grids.GridVariable(a, u.bc) for a, u in zip(dv_dt_arrays, v))

  # The factory returns the fully constructed function.
  return explicit_terms_with_same_bcs


def explicit_Reserve_BC(
    ReserveBC: BCFn ,
    step_time: float,
) -> Callable[[particle_class.All_Variables], particle_class.All_Variables]:
  """
  A factory function that creates a standardized wrapper for a boundary condition reservation step.

  This pattern is used to make various simulation steps conform to a uniform
  interface that the main time-stepper expects. This function takes a specific
  BC reservation function and returns a new function that takes the full
  simulation state (`All_Variables`) as input.

  Args:
    ReserveBC: The specific boundary condition reservation function to be wrapped
      (e.g., `boundaries.Reserve_BC`).
    step_time: The time step `dt`, which is passed to the wrapped function.

  Returns:
    A new function that applies the BC reservation step to the simulation state.
  """
  # Define a simple inner function that just calls the provided `ReserveBC` function.
  def Reserve_boundary(v, *args):
    return ReserveBC(v, *args)
  # Use the `_wrap_term_as_vector` helper to give the step a name for profiling
  # and adapt it to the PyTree structure.
  Reserve_bc_ = _wrap_term_as_vector(Reserve_boundary, name='Reserve_BC')
   
  @tree_math.wrap # Allows the function to operate on the `All_Variables` PyTree.
  # The commented out line is a remnant of a potential alternative implementation.
  # @functools.partial(jax.named_call, name='master_BC_fn')
  def _Reserve_bc(v):
    """The final wrapped function that will be returned."""
    # Call the adapted function with the state `v` and the pre-configured `step_time`.
    return Reserve_bc_(v, step_time)

  return _Reserve_bc

def explicit_update_BC(
    updateBC: BCFn,
    step_time: float,
) -> Callable[[particle_class.All_Variables], particle_class.All_Variables]:
  """
  A factory function that creates a standardized wrapper for a boundary condition update step.

  This follows the same pattern as `explicit_Reserve_BC`. It takes a specific
  BC update function and makes it compatible with the main solver's API.

  Args:
    updateBC: The specific boundary condition update function to be wrapped
      (e.g., `boundaries.update_BC`).
    step_time: The time step `dt`.

  Returns:
    A new function that applies the BC update step to the simulation state.
  """
  # Define the inner function.
  def Update_boundary(v, *args):
    return updateBC(v, *args)
  # Adapt it for PyTrees and profiling.
  Update_bc_ = _wrap_term_as_vector(Update_boundary, name='Update_BC')
   
  @tree_math.wrap
  def _Update_bc(v):
    """The final wrapped function."""
    return Update_bc_(v, step_time)

  return _Update_bc


def explicit_IBM_Force(
    cal_IBM_force: IBMFn,
    step_time: float,
) -> Callable[[particle_class.All_Variables], GridVariableVector]:
  """
  A factory function that creates a standardized wrapper for the IBM force calculation step.

  Args:
    cal_IBM_force: The specific IBM force calculation function to be wrapped
      (e.g., `IBM_Force.calc_IBM_force_NEW_MULTIPLE`).
    step_time: The time step `dt`.

  Returns:
    A new function that calculates the IBM force from the simulation state.
  """
  # Define the inner function.
  def IBM_FORCE(v, *args):
    return cal_IBM_force(v, *args)
  # Adapt it for PyTrees and profiling.
  IBM_FORCE_ = _wrap_term_as_vector(IBM_FORCE, name='IBM_FORCE')
   
  @tree_math.wrap
  def _IBM_FORCE(v):
    """The final wrapped function."""
    return IBM_FORCE_(v, step_time)

  return _IBM_FORCE


def explicit_Update_position(
    cal_Update_Position: PosFn,
    step_time: float,
) -> Callable[[particle_class.All_Variables], particle_class.All_Variables]:
  """
  A factory function that creates a standardized wrapper for the particle position update step.

  Args:
    cal_Update_Position: The specific particle motion function to be wrapped
      (e.g., `particle_motion.update_massive_deformable_particle`).
    step_time: The time step `dt`.

  Returns:
    A new function that updates the particle's position within the simulation state.
  """
  # Define the inner function.
  def Update_Position(v, *args):
    return cal_Update_Position(v, *args)
  # Adapt it for PyTrees and profiling.
  Update_Position_ = _wrap_term_as_vector(Update_Position, name='Update_Position')
   
  @tree_math.wrap
  def _Update_Position(v):
    """The final wrapped function."""
    return Update_Position_(v, step_time)

  return _Update_Position


def explicit_Calc_Drag(
    cal_Drag: DragFn,
    step_time: float,
) -> Callable[[particle_class.All_Variables], particle_class.All_Variables]:
  """
  A factory function that creates a standardized wrapper for a drag calculation/diagnostic step.

  Args:
    cal_Drag: The specific diagnostic function to be wrapped.
    step_time: The time step `dt`.

  Returns:
    A new function that performs the diagnostic calculation on the simulation state.
  """
  # Define the inner function.
  def Calculate_Drag(v, *args):
    return cal_Drag(v, *args)
  # Adapt it for PyTrees and profiling.
  Calculate_Drag_ = _wrap_term_as_vector(Calculate_Drag, name='Calculate_Drag')
   
  @tree_math.wrap
  def _Calculate_Drag(v):
    """The final wrapped function."""
    return Calculate_Drag_(v, step_time)

  return _Calculate_Drag


def explicit_Pressure_Gradient(
    cal_Pressure_Grad: GradPFn,
) -> Callable[[GridVariable], GridArrayVector]:
  """
  A factory function that creates a standardized wrapper for a pressure gradient calculation.

  This follows the same wrapper pattern as the previous functions. It takes a
  specific gradient function and makes it compatible with the main solver's API.
  This step is often used for calculating pressure drag on an immersed body.

  Args:
    cal_Pressure_Grad: The specific gradient function to be wrapped
      (e.g., `finite_differences.forward_difference`).

  Returns:
    A new function that calculates the pressure gradient from the pressure field.
  """
  # Define the inner function. Note that `v` here would be the pressure `GridVariable`.
  def Pressure_Grad(v):
    return cal_Pressure_Grad(v)
  # Adapt it for PyTrees and profiling.
  Pressure_Grad_ = _wrap_term_as_vector(Pressure_Grad, name='Pressure_Grad')
   
  @tree_math.wrap
  def _Pressure_Grad(v):
    """The final wrapped function."""
    return Pressure_Grad_(v)

  return _Pressure_Grad


def semi_implicit_navier_stokes_timeBC(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffuse: DiffuseFn = diffusion.diffuse,
    pressure_solve: Callable = pressure.solve_fast_diag,
    forcing: Optional[ForcingFn] = None,
    time_stepper: Callable = time_stepping.forward_euler_updated,
    IBM_forcing: IBMFn=None,
    Updating_Position:PosFn=None ,
    Pressure_Grad:GradPFn=finite_differences.forward_difference,
    Drag_fn:DragFn=None,
    
) -> Callable[[particle_class.All_Variables], particle_class.All_Variables]:
  """
  The main factory function for creating a complete FSI (Fluid-Structure Interaction) solver.

  This function assembles all the necessary physical and numerical components into a
  single, high-level time-stepping function. It uses a semi-implicit
  projection method, which is a standard algorithm for incompressible flow:
  1. Advance velocity with explicit terms (advection, diffusion, etc.) to get an intermediate velocity `v*`.
  2. Add the IBM forcing term.
  3. Solve a Poisson equation for pressure `p` to make the new velocity field divergence-free.
  4. Correct the velocity with the pressure gradient: `v_new = v* - dt/rho * ∇p`.
  5. Update the particle position based on the new, corrected velocity field.

  Args:
    All arguments are functions or parameters that define the specific numerical
    methods and physical properties for the simulation.

  Returns:
    A single function that takes the current simulation state (`All_Variables`)
    and returns the state at the next time step.
  """

  # 1. Create the function for the explicit advection-diffusion-forcing terms
  #    by calling the factory defined previously in this file.
  explicit_terms = navier_stokes_explicit_terms(
      density=density,
      viscosity=viscosity,
      dt=dt,
      grid=grid,
      convect=convect,
      diffuse=diffuse,
      forcing=forcing)

  # 2. Define the pressure projection function, which both solves for pressure
  #    and applies the correction to the velocity. We also name this step for profiling.
  pressure_projection = jax.named_call(pressure.projection_and_update_pressure, name='pressure')
  
  # 3. Use the wrapper factories to prepare all the other simulation steps,
  #    ensuring they have a consistent API.
  Reserve_BC = explicit_Reserve_BC(ReserveBC = boundaries.Reserve_BC,step_time = dt)
  update_BC = explicit_update_BC(updateBC = boundaries.update_BC,step_time = dt)
  IBM_force = explicit_IBM_Force(cal_IBM_force = IBM_forcing,step_time = dt)
  Update_Position =  explicit_Update_position(cal_Update_Position = Updating_Position,step_time = dt)
  Pressure_Grad =  explicit_Pressure_Gradient(cal_Pressure_Grad = Pressure_Grad)
  Calculate_Drag =  explicit_Calc_Drag(cal_Drag = Drag_fn,step_time = dt)
  
  # The commented out line is a remnant of a previous implementation or for debugging.
  #jax.named_call(boundaries.update_BC, name='Update_BC')
  
  # TODO(jamieas): This comment suggests a potential improvement: staggering the
  # pressure and advection/diffusion steps in time (e.g., a leapfrog scheme)
  # can sometimes improve stability or accuracy.
  
  # 4. Assemble all these component functions into a single ODE definition object.
  # This object encapsulates the entire logic of a single time derivative evaluation.
  ode = time_stepping.ExplicitNavierStokesODE_BCtime(
      explicit_terms,
      # The pressure projection part is defined as a lambda function to include the `pressure_solve` argument.
      lambda v: pressure_projection(v, solve=pressure_solve),
      update_BC,
      Reserve_BC,
      IBM_force,
      Update_Position,
      Pressure_Grad,
      Calculate_Drag,
  )
  
  # 5. Pass the complete ODE definition to the chosen time-stepper factory.
  # This creates the final, concrete function that advances the simulation.
  step_fn = time_stepper(ode, dt)
  return step_fn


def semi_implicit_navier_stokes_penalty(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffuse: DiffuseFn = diffusion.diffuse,
    pressure_solve: Callable = pressure.solve_fast_diag,
    forcing: Optional[ForcingFn] = None,
    time_stepper: Callable = time_stepping.forward_euler_penalty,
) -> Callable[[particle_class.All_Variables], particle_class.All_Variables]:
  """
  Returns a function that performs a time step of Navier-Stokes, simplified for a penalty method.
  
  This is a slightly simplified version of the main solver factory. It omits the
  explicit IBM force, particle update, and diagnostic steps from the `ODE`
  definition. This implies that for this workflow, the IBM force is likely
  included within the `explicit_terms` function, and the particle motion is
  updated in a separate step outside this fluid solver step.

  Args:
    (Same as the function above, but with fewer high-level function arguments).

  Returns:
    A single function that advances the fluid state by one time step.
  """
  # Create the explicit terms function as before.
  explicit_terms = navier_stokes_explicit_terms(
      density=density,
      viscosity=viscosity,
      dt=dt,
      grid=grid,
      convect=convect,
      diffuse=diffuse,
      forcing=forcing)

  # Define the pressure projection step.
  pressure_projection = jax.named_call(pressure.projection_and_update_pressure, name='pressure')
  
  # Prepare the boundary condition update steps.
  Reserve_BC = explicit_Reserve_BC(ReserveBC = boundaries.Reserve_BC,step_time = dt)
  update_BC = explicit_update_BC(updateBC = boundaries.update_BC,step_time = dt)
  
  # Assemble the simplified ODE object.
  ode = time_stepping.ExplicitNavierStokesODE_Penalty(
      explicit_terms,
      lambda v: pressure_projection(v, solve=pressure_solve),
      update_BC,
      Reserve_BC,
  )
  
  # Create the final time-stepping function using the chosen integrator.
  step_fn = time_stepper(ode, dt)
  return step_fn
