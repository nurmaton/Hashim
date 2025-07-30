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
Assembles the components of the Navier-Stokes equations into a time-stepping function.

This module provides high-level "factory" functions that construct a complete
stepper for the incompressible Navier-Stokes equations with an immersed boundary.
It combines various physical terms (advection, diffusion, forcing) and numerical
procedures (pressure projection, IBM force calculation, time integration) into a
single callable function that advances the simulation state by one time step.

The primary function, `semi_implicit_navier_stokes_timeBC`, builds a solver
that uses a semi-implicit time-stepping scheme. This means that some terms
(like advection) are treated explicitly, while others (like pressure) are
treated implicitly, which is a common and robust approach for fluid simulation.
"""

import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp

# Import all the necessary building blocks from other modules.
from jax_ib.base import advection
from jax_ib.base import diffusion
from jax_ib.base import grids
from jax_ib.base import pressure
from jax_cfd.base import pressure as pressureCFD # Potentially for a different pressure solver
from jax_ib.base import time_stepping
from jax_ib.base import boundaries
from jax_ib.base import finite_differences
import tree_math # A utility for working with PyTrees
from jax_ib.base import particle_class
from jax_cfd.base import equations as equationsCFD # Potentially for other equation setups

# --- Type Aliases ---
# Using type hints for function types makes the code easier to read and understand.
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
  debugging using tools like the JAX profiler.
  """
  # `tree_math.unwrap` adapts a function designed for a single PyTree (like one
  # velocity component) to work on a tuple of PyTrees (the full velocity vector).
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
  Factory that returns a function for the explicit terms of the Navier-Stokes eq.
  
  This combines advection, diffusion, and any external forcing into a single
  function that calculates the rate of change of velocity, *excluding* the
  pressure gradient term. `dv/dt = -(v ⋅ ∇)v + ν∇²v + f`.

  Args:
    density: Fluid density.
    viscosity: Fluid kinematic viscosity.
    dt: Time step, needed for some advection schemes like Van Leer.
    grid: The simulation grid.
    convect: A function to compute the convection term. Defaults to Van Leer.
    diffuse: A function to compute the diffusion term.
    forcing: An optional function for external forces (e.g., gravity).

  Returns:
    A function that takes a velocity vector and returns its time derivative
    due to the explicit terms.
  """
  del grid  # The grid argument is currently unused but kept for API consistency.

  # Set a default convection scheme if none is provided.
  if convect is None:
    # `advect_van_leer_using_limiters` is a good, robust default choice that
    # balances accuracy and stability.
    def convect(v):  # pylint: disable=function-redefined
      return tuple(
          advection.advect_van_leer_using_limiters(u, v, dt) for u in v)

  # Adapt the scalar diffusion function to work on a vector of velocities
  # by applying it to each component.
  def diffuse_velocity(v, *args):
    return tuple(diffuse(u, *args) for u in v)

  # Use the wrapper to prepare the component functions for vector inputs.
  convection = _wrap_term_as_vector(convect, name='convection')
  diffusion_ = _wrap_term_as_vector(diffuse_velocity, name='diffusion')
  if forcing is not None:
    forcing = _wrap_term_as_vector(forcing, name='forcing')

  @tree_math.wrap # Allows the function to work seamlessly with PyTrees.
  @functools.partial(jax.named_call, name='navier_stokes_momentum')
  def _explicit_terms(v):
    """Computes the sum of all explicit momentum terms."""
    # Start with the convection term.
    dv_dt = convection(v)
    # Add diffusion if viscosity is non-zero.
    if viscosity is not None:
      dv_dt += diffusion_(v, viscosity / density)
    # Add external forcing if provided.
    if forcing is not None:
      dv_dt += forcing(v) / density
    
    return dv_dt

  def explicit_terms_with_same_bcs(v):
    """Ensures the output GridVariables have the same BCs as the input."""
    dv_dt = _explicit_terms(v)
    # The result of the calculus operations (a GridArray) is wrapped back into a
    # GridVariable, preserving the original boundary condition information.
    return tuple(grids.GridVariable(a, u.bc) for a, u in zip(dv_dt, v))

  return explicit_terms_with_same_bcs


# --- A series of factory functions to wrap specific simulation steps ---
# These wrappers make the various steps (updating BCs, calculating forces, etc.)
# conform to the standard interface expected by the main time-stepping ODE.

def explicit_Reserve_BC(
    ReserveBC: BCFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Wraps the boundary condition reservation step."""
  # Inner function to match expected signature.
  def Reserve_boundary(v, *args):
    return ReserveBC(v, *args)
  # Wrap it for vector inputs and profiling.
  Reserve_bc_ = _wrap_term_as_vector(Reserve_boundary, name='Reserve_BC')
   
  @tree_math.wrap
  def _Reserve_bc(v):
    return Reserve_bc_(v,step_time)

  return _Reserve_bc

def explicit_update_BC(
    updateBC: BCFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Wraps the boundary condition update step."""
  def Update_boundary(v, *args):
    return updateBC(v, *args)
  Update_bc_ = _wrap_term_as_vector(Update_boundary, name='Update_BC')
   
  @tree_math.wrap
  def _Update_bc(v):
    return Update_bc_(v,step_time)

  return _Update_bc


def explicit_IBM_Force(
    cal_IBM_force: IBMFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Wraps the IBM force calculation step."""
  def IBM_FORCE(v, *args):
    return cal_IBM_force(v, *args)
  IBM_FORCE_ = _wrap_term_as_vector(IBM_FORCE, name='IBM_FORCE')
   
  @tree_math.wrap
  def _IBM_FORCE(v):
    return IBM_FORCE_(v,step_time)

  return _IBM_FORCE


def explicit_Update_position(
    cal_Update_Position: PosFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Wraps the particle position update step."""
  def Update_Position(v, *args):
    return cal_Update_Position(v, *args)
  Update_Position_ = _wrap_term_as_vector(Update_Position, name='Update_Position')
   
  @tree_math.wrap
  def _Update_Position(v):
    return Update_Position_(v,step_time)

  return _Update_Position


def explicit_Calc_Drag(
    cal_Drag: DragFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Wraps the drag calculation step."""
  def Calculate_Drag(v, *args):
    return cal_Drag(v, *args)
  Calculate_Drag_ = _wrap_term_as_vector(Calculate_Drag, name='Calculate_Drag')
   
  @tree_math.wrap
  def _Calculate_Drag(v):
       # The step_time argument might be unused here, depending on `cal_Drag`.
       return Calculate_Drag_(v,step_time)

  return _Calculate_Drag

def explicit_Pressure_Gradient(
    cal_Pressure_Grad: GradPFn,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Wraps the pressure gradient calculation step."""
  def Pressure_Grad(v):
    return cal_Pressure_Grad(v)
  Pressure_Grad_ = _wrap_term_as_vector(Pressure_Grad, name='Pressure_Grad')
   
  @tree_math.wrap
  def _Pressure_Grad(v):
    return Pressure_Grad_(v)

  return _Pressure_Grad

# --- The Main High-Level Solver Factories ---

def semi_implicit_navier_stokes_timeBC(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffuse: DiffuseFn = diffusion.diffuse,
    pressure_solve: Callable = pressureCFD.solve_fast_diag,
    forcing: Optional[ForcingFn] = None,
    time_stepper: Callable = time_stepping.forward_euler_updated,
    IBM_forcing: IBMFn=None,
    Updating_Position:PosFn=None ,
    Pressure_Grad:GradPFn=finite_differences.forward_difference,
    Drag_fn:DragFn=None,
    
) -> Callable[[GridVariableVector], GridVariableVector]:
  """
  Returns a function that performs a full time step of the Navier-Stokes equations
  with immersed boundaries and time-dependent boundary conditions.

  This is the main entry point for creating a solver. It uses a semi-implicit
  projection method, which is a standard algorithm for incompressible flow:
  1. Advance velocity with explicit terms (advection, diffusion, etc.) to get an intermediate velocity `v*`.
  2. Add the IBM forcing term.
  3. Update boundary conditions if they are time-dependent.
  4. Solve a Poisson equation for pressure `p` that ensures the final velocity field is divergence-free.
  5. Correct the velocity with the pressure gradient: `v_new = v* - dt/rho * ∇p`.
  6. Update the particle position based on the new, corrected velocity field.
  
  Args:
    All arguments are functions or parameters that define the specific numerical
    methods and physical properties for the simulation.

  Returns:
    A single function that takes the current simulation state (`All_Variables`)
    and returns the state at the next time step.
  """
  # 1. Create the function for the explicit advection-diffusion-forcing terms.
  explicit_terms = navier_stokes_explicit_terms(
      density=density,
      viscosity=viscosity,
      dt=dt,
      grid=grid,
      convect=convect,
      diffuse=diffuse,
      forcing=forcing)

  # 2. Define the pressure projection function, which both solves for pressure
  #    and applies the correction to the velocity.
  pressure_projection = jax.named_call(pressure.projection_and_update_pressure, name='pressure')
  
  # 3. Prepare all the other steps using the explicit wrappers defined above.
  Reserve_BC = explicit_Reserve_BC(ReserveBC = boundaries.Reserve_BC,step_time = dt)
  update_BC = explicit_update_BC(updateBC = boundaries.update_BC,step_time = dt)
  IBM_force = explicit_IBM_Force(cal_IBM_force = IBM_forcing,step_time = dt)
  Update_Position =  explicit_Update_position(cal_Update_Position = Updating_Position,step_time = dt)
  Pressure_Grad =  explicit_Pressure_Gradient(cal_Pressure_Grad = Pressure_Grad)
  Calculate_Drag =  explicit_Calc_Drag(cal_Drag = Drag_fn,step_time = dt)
  
  # 4. Assemble all these pieces into a single ODE definition object.
  # This ODE object describes to the time-stepper how to perform all the steps
  # within a single time integration step.
  ode = time_stepping.ExplicitNavierStokesODE_BCtime(
      explicit_terms,
      lambda v: pressure_projection(v, pressure_solve), # Pressure is the implicit part.
      update_BC,
      Reserve_BC,
      IBM_force,
      Update_Position,
      Pressure_Grad,
      Calculate_Drag,
  )
  
  # 5. Create the final time-stepping function using the chosen integrator.
  step_fn = time_stepper(ode, dt)
  return step_fn


def semi_implicit_navier_stokes_penalty(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffuse: DiffuseFn = diffusion.diffuse,
    pressure_solve: Callable = pressureCFD.solve_fast_diag,
    forcing: Optional[ForcingFn] = None,
    time_stepper: Callable = time_stepping.forward_euler_penalty,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """
  Returns a function that performs a time step of Navier Stokes for the penalty method.
  
  This is a slightly simplified version of the main solver factory, likely tailored
  for the specific needs of the penalty method where some steps might be different
  or combined.
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

  # Assemble the ODE object, which has fewer components than the BCtime version.
  ode = time_stepping.ExplicitNavierStokesODE_Penalty(
      explicit_terms,
      lambda v: pressure_projection(v, pressure_solve),
      update_BC,
      Reserve_BC,
  )
  
  # Create the final time-stepping function using the chosen integrator.
  step_fn = time_stepper(ode, dt)
  return step_fn
