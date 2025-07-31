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
Functions for advancing the simulation state forward in time.

This module provides the core machinery for time integration. It defines a
powerful and modular architecture for constructing time-stepping functions
that solve the spatially discretized Navier-Stokes equations coupled with an
immersed boundary.

The main architectural pattern is:

1.  **ODE Definition Classes**: A class (e.g., `ExplicitNavierStokesODE_BCtime`)
    is used to encapsulate the physics of the problem. It acts as a container
    for all the functions that constitute the right-hand-side of the system of
    ordinary differential equations (ODEs) that result from spatial discretization.
    This includes functions for explicit fluid terms (advection, diffusion),
    pressure projection, IBM forces, and particle motion updates. Two main
    versions exist:
    - `ExplicitNavierStokesODE_Penalty`: A simpler definition for penalty-based
      methods where the IBM forcing is part of the explicit fluid terms.
    - `ExplicitNavierStokesODE_BCtime`: A comprehensive definition for a fully
      coupled Fluid-Structure Interaction problem, with separate steps for
      forcing and particle motion.

2.  **Time-Stepper Factory**: A high-level function (e.g., `navier_stokes_rk_updated`)
    acts as a "factory." It takes an instance of an ODE class, a time step `dt`,
    and a `ButcherTableau` (which defines a specific Runge-Kutta method) and
    returns a new function.

3.  **Step Function**: The returned function (`step_fn`) is the final, concrete
    time-stepper. It takes the current simulation state (`All_Variables` PyTree)
    as input and returns the state at the next time step, `t + dt`. This is the
    function that is typically used within a `jax.lax.scan` loop to run the
-   full simulation.

This design separates the physical model from the numerical integration scheme,
making it easy to experiment with different time-stepping methods (e.g., Forward
Euler, higher-order Runge-Kutta) without changing the underlying physics code.
"""

import dataclasses
from typing import Callable, Sequence, TypeVar
import jax
import tree_math # A utility library for PyTree arithmetic.
from jax_ib.base import boundaries
from jax_ib.base import grids
from jax_cfd.base import time_stepping # For potential re-use from the jax_cfd library.
from jax_ib.base import particle_class


# A generic type variable that can represent any JAX PyTree. This is used to
# type hint the simulation state, which is a complex PyTree.
PyTreeState = TypeVar("PyTreeState")

# A type hint for a "time step function". This is a function that takes a state
# (any PyTree) and returns a new state of the same type.
TimeStepFn = Callable[[PyTreeState], PyTreeState]


class ExplicitNavierStokesODE_Penalty:
  """
  A container class for the functions defining a spatially discretized
  Navier-Stokes equation, tailored for the penalty method.

  This class acts as a "struct" or "record" to bundle together all the component
  functions required to evaluate the time evolution of the system. An instance
  of this class is passed to a time integrator (like Forward Euler) to create a
  full step function.

  The spatially discretized Navier-Stokes equations can be thought of as a system
  of differential-algebraic equations (DAEs). This class splits the system into:
  1. An explicit ODE part: `∂u/∂t = explicit_terms(u)`
  2. An algebraic constraint: `0 = incompressibility_constraint(u)`
  """

  def __init__(self, explicit_terms, pressure_projection,update_BC,Reserve_BC):
    """
    Initializes the ODE definition object.

    Args:
      explicit_terms: A function that calculates the explicit parts of the
        Navier-Stokes equations (advection, diffusion, forcing).
      pressure_projection: A function that enforces the incompressibility
        constraint, typically by solving a Poisson equation for pressure.
      update_BC: A function to update time-dependent boundary conditions.
      Reserve_BC: A function to reserve or reset boundary conditions, likely
        a legacy or special-purpose utility.
    """
    self.explicit_terms = explicit_terms
    self.pressure_projection = pressure_projection
    self.update_BC = update_BC
    self.Reserve_BC = Reserve_BC


  def explicit_terms(self, state):
    """
    Abstract method placeholder for explicitly evaluating the ODE.
    The actual implementation is provided via the `explicit_terms` attribute
    during initialization. This method is here to define the interface.
    """
    raise NotImplementedError

  def pressure_projection(self, state):
    """
    Abstract method placeholder for enforcing the incompressibility constraint.
    The actual implementation is provided via the `pressure_projection` attribute.
    """
    raise NotImplementedError

  def update_BC(self, state):
    """
    Abstract method placeholder for updating Wall Boundary Conditions.
    The actual implementation is provided via the `update_BC` attribute.
    """
    raise NotImplementedError

  def Reserve_BC(self, state):
    """
    Abstract method placeholder for reverting spurious updates of Wall BC.
    The actual implementation is provided via the `Reserve_BC` attribute.
    """
    raise NotImplementedError

class ExplicitNavierStokesODE_BCtime:
  """
  A container for the functions defining the spatially discretized Navier-Stokes
  equations, specifically including steps for Immersed Boundary Method (IBM)
  and time-dependent boundary conditions.

  This class acts as a comprehensive "struct" that bundles all the individual
  steps of a complex fluid-structure interaction time step into a single object.
  This object is then passed to a time integrator (like a Runge-Kutta method),
  which uses the provided functions to advance the simulation state.
  
  This is the more advanced ODE definition, including all components for a full
  simulation with moving particles and diagnostics.
  """

  def __init__(
      self,
      explicit_terms: Callable,
      pressure_projection: Callable,
      update_BC: Callable,
      Reserve_BC: Callable,
      IBM_force: Callable,
      Update_Position: Callable,
      Pressure_Grad: Callable,
      Calculate_Drag: Callable
  ):
    """
    Initializes the comprehensive ODE definition object.

    Args:
      explicit_terms: A function that calculates the explicit fluid dynamics
        terms (advection, diffusion, etc.).
      pressure_projection: A function that enforces the fluid incompressibility
        constraint.
      update_BC: A function to update time-dependent boundary conditions.
      Reserve_BC: A function to reserve or reset boundary conditions.
      IBM_force: A function to calculate the force exerted by the immersed
        boundary on the fluid.
      Update_Position: A function to update the position of the immersed
        boundary based on fluid interaction.
      Pressure_Grad: A function to calculate the pressure gradient, often used
        for diagnostic purposes like calculating pressure drag.
      Calculate_Drag: A function to calculate and record diagnostic quantities
        like the total drag force on the body.
    """
    # Stores the function for explicit fluid terms (advection, diffusion, etc.).
    self.explicit_terms = explicit_terms
    # Stores the function for the pressure projection step.
    self.pressure_projection = pressure_projection
    # Stores the function to update time-dependent boundary conditions.
    self.update_BC = update_BC
    # Stores the function to reserve/reset boundary conditions.
    self.Reserve_BC = Reserve_BC
    # Stores the function to calculate the IBM force.
    self.IBM_force = IBM_force
    # Stores the function to update the particle's position.
    self.Update_Position = Update_Position
    # Stores the function to calculate the pressure gradient.
    self.Pressure_Grad = Pressure_Grad
    # Stores the function to calculate drag force.
    self.Calculate_Drag = Calculate_Drag

  def explicit_terms(self, state):
    """
    Abstract method placeholder for explicitly evaluating the ODE.
    The actual implementation is provided via the `explicit_terms` attribute
    during initialization. This is here to define the class interface.
    """
    raise NotImplementedError

  def pressure_projection(self, state):
    """
    Abstract method placeholder for enforcing the incompressibility constraint.
    The actual implementation is provided via the `pressure_projection` attribute.
    """
    raise NotImplementedError

  def update_BC(self, state):
    """
    Abstract method placeholder for updating Wall Boundary Conditions.
    The actual implementation is provided via the `update_BC` attribute.
    """
    raise NotImplementedError

  def Reserve_BC(self, state):
    """
    Abstract method placeholder for reverting spurious updates of Wall BC.
    The actual implementation is provided via the `Reserve_BC` attribute.
    """
    raise NotImplementedError
    
  def IBM_force(self, state):
    """
    Abstract method placeholder for calculating the IBM force.
    The docstring "Revert spurious updates..." is likely a copy-paste error
    and should describe the IBM force calculation.
    """
    raise NotImplementedError

  def Update_Position(self, state):
    """
    Abstract method placeholder for updating the particle position.
    The docstring "Revert spurious updates..." is likely a copy-paste error.
    """
    raise NotImplementedError

  def Pressure_Grad(self, state):
    """
    Abstract method placeholder for calculating the pressure gradient.
    The docstring "Revert spurious updates..." is likely a copy-paste error.
    """
    raise NotImplementedError
    
  def Calculate_Drag(self, state):
    """
    Abstract method placeholder for calculating drag.
    The docstring "Revert spurious updates..." is likely a copy-paste error.
    """
    raise NotImplementedError


@dataclasses.dataclass
class ButcherTableau_updated:
  """
  A data class to hold the coefficients of a Runge-Kutta time integration scheme.
  The tableau defines the weights and stages of the RK method. For an explicit
  method, the `a` matrix is lower triangular.
  See: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

  Attributes:
    a: A matrix of coefficients `a_ij` defining the weights for intermediate stages.
    b: A vector of coefficients `b_j` defining the weights for the final result.
    c: A vector of coefficients `c_i` defining the time points of intermediate stages.
  """
  a: Sequence[Sequence[float]]
  b: Sequence[float]
  c: Sequence[float]

  def __post_init__(self):
    """A validation check that runs after the dataclass is initialized."""
    # This check ensures the dimensions of the tableau are consistent.
    if len(self.a) + 1 != len(self.b):
      raise ValueError("inconsistent Butcher tableau")
      
      
def navier_stokes_rk_updated(
    tableau: ButcherTableau_updated,
    equation: ExplicitNavierStokesODE_BCtime,
    time_step: float,
) -> TimeStepFn:
  """
  Creates a Runge-Kutta time-stepper for the incompressible Navier-Stokes
  equations with immersed boundaries.

  This function is a "factory": it takes the definition of an ODE and an RK method
  and returns a concrete function that performs one time step. The algorithm
  combines the RK stages for explicit terms with pressure projection steps to
  maintain incompressibility, and it incorporates the IBM forcing and particle
  updates.

  Args:
    tableau: A `ButcherTableau_updated` object defining the specific RK method
      (e.g., Forward Euler, RK4).
    equation: An `ExplicitNavierStokesODE_BCtime` object containing all the
      functions that define the system's evolution.
    time_step: The overall time-step size, `dt`.

  Returns:
    A `TimeStepFn` that takes the current `All_Variables` state and returns the
    state advanced by one time step.
  """
  # pylint: disable=invalid-name
  # Unpack equation components and tableau coefficients for convenience.
  # The `tree_math.unwrap` utility adapts the functions to work on the PyTree
  # structure of the velocity vector.
  dt = time_step
  F = tree_math.unwrap(equation.explicit_terms)
  P = tree_math.unwrap(equation.pressure_projection)
  M = tree_math.unwrap(equation.update_BC)
  R = tree_math.unwrap(equation.Reserve_BC)
  IBM = tree_math.unwrap(equation.IBM_force)
  Update_Pos = tree_math.unwrap(equation.Update_Position) 
  Grad_Pressure = tree_math.unwrap(equation.Pressure_Grad) 
  Drag_Calculation = tree_math.unwrap(equation.Calculate_Drag)  

  a = tableau.a
  b = tableau.b
  num_steps = len(b)
  
  @tree_math.wrap # Allows the step function to operate on the `All_Variables` PyTree.
  def step_fn(u0_all_vars):
    """The returned function that performs one complete time step."""
    # Lists to store the state `u` and derivative `k` at each RK stage.
    u = [None] * num_steps
    k = [None] * num_steps

    # --- A series of nested helper functions for data conversion ---
    # These functions are used to switch between the full `All_Variables` state,
    # a `GridVariableVector` for velocity, and a `Vector` of raw JAX arrays.
    # This is complex and suggests the data structures could potentially be simplified.

    def convert_to_velocity_vecot(u_grid_vars):
        """Extracts raw data arrays from a GridVariableVector."""
        u = u_grid_vars.tree
        return tree_math.Vector(tuple(u[i].array for i in range(len(u))))
        
    def convert_to_velocity_tree(vel_vector_raw, bcs):
        """Re-wraps raw data arrays into a GridVariableVector."""
        # return tree_math.Vector(tuple(grids.GridVariable(grids.GridArray(data, gv.offset, gv.grid), bc)
        #                         for data, gv, bc in zip(vel_vector_raw.tree, u0.tree, bcs)))
        return tree_math.Vector(tuple(grids.GridVariable(v,bc) for v,bc in zip(vel_vector_raw.tree,bcs)))
    
    def convert_all_variabl_to_velocity_vecot(u_all_vars):
        """Extracts the velocity GridVariableVector from the full state."""
        u = u_all_vars.tree.velocity
        return  tree_math.Vector(u)
        
    def covert_veloicty_to_All_variable_vecot(particles, vel_tree, pressure, Drag, Step_count, MD_var):
        """Rebuilds the full All_Variables state from its components."""
        u = vel_tree.tree
        return tree_math.Vector(particle_class.All_Variables(particles, u, pressure, Drag, Step_count, MD_var))
    
    # --- Helper functions to extract non-velocity parts of the state. ---
    def velocity_bc(u_all_vars): return tuple(uv.bc for uv in u_all_vars.tree.velocity)
    def the_particles(u_all_vars): return u_all_vars.tree.particles
    def the_pressure(u_all_vars): return u_all_vars.tree.pressure
    def the_Drag(u_all_vars): return u_all_vars.tree.Drag
    
    # --- Main Time-Stepping Logic ---
    # 1. Unpack the initial state `u0_all_vars`.
    particles = the_particles(u0_all_vars)
    ubc = velocity_bc(u0_all_vars)  
    pressure = the_pressure(u0_all_vars)
    Drag = the_Drag(u0_all_vars)
    Step_count = u0_all_vars.tree.Step_count
    MD_var = u0_all_vars.tree.MD_var
    
    # Isolate the initial velocity.
    u0 = convert_all_variabl_to_velocity_vecot(u0_all_vars)

    # 2. Perform the Runge-Kutta stages.
    u0_raw = convert_to_velocity_vecot(u0)
    u[0] = u0_raw
    k[0] = convert_to_velocity_vecot(F(u0)) # k0 = F(u(t))
    dP = Grad_Pressure(tree_math.Vector(pressure)) # Pre-calculate pressure gradient?

    for i in range(1, num_steps):
      # Calculate the intermediate velocity `u_star` for the current stage.
      u_star_raw = u0_raw + dt * sum(a[i-1][j] * k[j] for j in range(i) if a[i-1][j])
      # Project the intermediate velocity to enforce incompressibility.
      u_star_tree = convert_to_velocity_tree(u_star_raw, ubc)
      u[i] = convert_to_velocity_vecot(P(u_star_tree))
      # Calculate the derivative at the intermediate stage: ki = F(u(t + c_i*dt)).
      k[i] = convert_to_velocity_vecot(F(convert_to_velocity_tree(u[i], ubc)))

    # 3. Combine stages to get the final explicit velocity update.
    # The pressure gradient `dP` is subtracted here, an unconventional step.
    u_star_raw = u0_raw + dt * sum(b[j] * k[j] for j in range(num_steps) if b[j]) - dP
    
    # 4. Apply the Immersed Boundary Method forcing.
    u_star_tree = convert_to_velocity_tree(u_star_raw, ubc)
    u_star_all_vars = covert_veloicty_to_All_variable_vecot(particles, u_star_tree, pressure, Drag, Step_count, MD_var)
    Force = IBM(u_star_all_vars)
    
    # 5. Calculate diagnostic quantities.
    # This uses the Force as input, which is unusual. It might be calculating the
    # fluid's reaction force on the body.
    Drag_variable = Drag_Calculation(covert_veloicty_to_All_variable_vecot(particles, Force, pressure, Drag, Step_count, MD_var))
    Drag = the_Drag(Drag_variable)
    
    Force_raw = convert_to_velocity_vecot(Force)
    
    # Add the IBM force to the intermediate velocity.
    u_star_star_raw = u_star_raw + dt * Force_raw
    
    # The commented out code block appears to be a previous iteration of the logic.
    
    # 6. Final pressure projection and state updates.
    u_final_tree = convert_to_velocity_tree(u_star_star_raw, ubc)
    u_final_all_vars = covert_veloicty_to_All_variable_vecot(particles, u_final_tree, pressure, Drag, Step_count, MD_var)
    
    # Final projection to ensure the velocity field is divergence-free after forcing.
    u_final_all_vars = P(u_final_all_vars)
    # Update time-dependent boundary conditions for the next step.
    u_final_all_vars = M(u_final_all_vars)
    # Update the particle position based on the final, corrected fluid velocity.
    u_final_all_vars = Update_Pos(u_final_all_vars)
    
    return u_final_all_vars

  return step_fn

def navier_stokes_rk_penalty(
    tableau: ButcherTableau_updated,
    equation: ExplicitNavierStokesODE_Penalty, # Note: Takes the simpler ODE class
    time_step: float,
) -> TimeStepFn:
  """
  Creates a forward Runge-Kutta time-stepper for incompressible Navier-Stokes,
  tailored for a penalty method approach.

  This function implements a standard RK scheme combined with pressure projection.
  It is a simplified version of `navier_stokes_rk_updated` that omits the explicit
  IBM force calculation and particle position updates from within the time-stepping
  loop. This suggests that in the penalty method workflow, the IBM forces are
  likely included within the `explicit_terms` (F) function, and the particle
  motion is updated in a separate step outside this fluid solver step.

  Args:
    tableau: A `ButcherTableau_updated` object defining the specific RK method.
    equation: An `ExplicitNavierStokesODE_Penalty` object containing the system's
      governing functions.
    time_step: The overall time-step size, `dt`.

  Returns:
    A `TimeStepFn` that advances the simulation state by one time step.
  """
  # pylint: disable=invalid-name
  # Unpack the component functions from the equation object for easier access.
  dt = time_step
  F = tree_math.unwrap(equation.explicit_terms)
  P = tree_math.unwrap(equation.pressure_projection)
  M = tree_math.unwrap(equation.update_BC)
  R = tree_math.unwrap(equation.Reserve_BC)

  # Unpack the RK coefficients from the tableau.
  a = tableau.a
  b = tableau.b
  num_steps = len(b)
  
  @tree_math.wrap # Decorator to make this function work with PyTrees.
  def step_fn(u0_all_vars):
    """The returned function that performs one complete time step."""
    # Lists to store the state `u` and derivative `k` at each RK stage.
    u = [None] * num_steps
    k = [None] * num_steps

    # --- A series of nested helper functions for data conversion ---
    # These are identical to the helpers in the previous function and are used
    # to switch between different data structure representations.

    def convert_to_velocity_vecot(u_grid_vars):
        """Extracts raw data arrays from a GridVariableVector."""
        u = u_grid_vars.tree
        # Note: The original code had `u[i].array` which might be incorrect if `u` is
        # already the raw array. Assuming `u` is a GridVariableVector.
        return tree_math.Vector(tuple(uv.array.data for uv in u))
        
    def convert_to_velocity_tree(vel_vector_raw, bcs):
        """Re-wraps raw data arrays into a GridVariableVector."""
        # This reconstruction is complex and assumes knowledge of the original GridArrays.
        return tree_math.Vector(tuple(grids.GridVariable(grids.GridArray(data, gv.offset, gv.grid), bc)
                                for data, gv, bc in zip(vel_vector_raw.tree, u0.tree, bcs)))
    
    def convert_all_variabl_to_velocity_vecot(u_all_vars):
        """Extracts the velocity GridVariableVector from the full state."""
        u = u_all_vars.tree.velocity
        return  tree_math.Vector(u)
        
    def covert_veloicty_to_All_variable_vecot(particles, vel_tree, pressure, Drag, Step_count, MD_var):
        """Rebuilds the full All_Variables state from its components."""
        u = vel_tree.tree
        return tree_math.Vector(particle_class.All_Variables(particles, u, pressure, Drag, Step_count, MD_var))
    
    # --- Helper functions to extract non-velocity parts of the state. ---
    def velocity_bc(u_all_vars): return tuple(uv.bc for uv in u_all_vars.tree.velocity)
    def the_particles(u_all_vars): return u_all_vars.tree.particles
    def the_pressure(u_all_vars): return u_all_vars.tree.pressure
    def the_Drag(u_all_vars): return u_all_vars.tree.Drag

    # --- Main Time-Stepping Logic ---
    # 1. Unpack the initial state `u0_all_vars`.
    particles = the_particles(u0_all_vars)
    ubc = velocity_bc(u0_all_vars)  
    pressure = the_pressure(u0_all_vars)
    Drag = the_Drag(u0_all_vars)
    Step_count = u0_all_vars.tree.Step_count
    MD_var = u0_all_vars.tree.MD_var

    # Isolate the initial velocity GridVariableVector.
    u0 = convert_all_variabl_to_velocity_vecot(u0_all_vars)
    
    # Get the raw JAX array data for the initial velocity.
    u0_raw = convert_to_velocity_vecot(u0)
    
    # 2. Perform the Runge-Kutta stages.
    u[0] = u0_raw
    k[0] = convert_to_velocity_vecot(F(u0)) # k0 = F(u(t))
    
    for i in range(1, num_steps):
      # Calculate the intermediate velocity `u_star` for the current stage.
      u_star_raw = u0_raw + dt * sum(a[i-1][j] * k[j] for j in range(i) if a[i-1][j])
      # Project the intermediate velocity to enforce incompressibility.
      u_star_tree = convert_to_velocity_tree(u_star_raw, ubc)
      u_star_all_vars = covert_veloicty_to_All_variable_vecot(particles, u_star_tree, pressure, Drag, Step_count, MD_var)
      u[i] = convert_to_velocity_vecot(P(u_star_all_vars))
      # Calculate the derivative at the intermediate stage: ki = F(u(t + c_i*dt)).
      k[i] = convert_to_velocity_vecot(F(convert_to_velocity_tree(u[i], ubc)))

    # 3. Combine stages to get the final explicit velocity update.
    u_star_raw = u0_raw + dt * sum(b[j] * k[j] for j in range(num_steps) if b[j])

    # 4. Rebuild the full state before the final projection and BC update.
    u_final_tree = convert_to_velocity_tree(u_star_raw, ubc)
    u_final_all_vars = covert_veloicty_to_All_variable_vecot(particles, u_final_tree, pressure, Drag, Step_count, MD_var)
    
    # Final pressure projection.
    u_final_all_vars = P(u_final_all_vars)
    
    # Final boundary condition update.
    u_final_all_vars = M(u_final_all_vars)
    
    # The final state is returned. Note the absence of IBM force and position update.
    return u_final_all_vars

  return step_fn

def forward_euler_penalty(
    equation: ExplicitNavierStokesODE_Penalty,
    time_step: float,
) -> TimeStepFn:
  """
  A convenience function to create a Forward Euler time-stepper for the penalty method.
  
  Forward Euler is the simplest explicit Runge-Kutta method (RK1). This function
  acts as a shortcut by pre-populating the Butcher Tableau with the correct
  coefficients for Forward Euler and passing them to the general RK stepper factory
  (`navier_stokes_rk_penalty`).

  Args:
    equation: An `ExplicitNavierStokesODE_Penalty` object containing the system's
      governing functions.
    time_step: The time step size, `dt`.

  Returns:
    A `TimeStepFn` that advances the simulation state by one time step using
    the Forward Euler method.
  """
  # `jax.named_call` gives this step a specific name ("forward_euler") which will
  # appear in JAX's profiler output, making it easier to debug and analyze performance.
  return jax.named_call(
      # Call the general RK stepper factory.
      navier_stokes_rk_penalty(
          # Provide the Butcher Tableau for the Forward Euler method.
          # It has one stage (len(b)=1), with no intermediate steps (a=[]),
          # starting at time c=0, and the final result is just the derivative
          # from the first stage multiplied by dt (b=[1]).
          ButcherTableau_updated(a=[], b=[1], c=[0]),
          equation,
          time_step),
      name="forward_euler",
  )

def forward_euler_updated(
    equation: ExplicitNavierStokesODE_BCtime,
    time_step: float,
) -> TimeStepFn:
  """
  A convenience function to create a Forward Euler time-stepper for the full IBM problem.

  This function is the counterpart to `forward_euler_penalty`. It creates a
  Forward Euler stepper by calling the more comprehensive RK factory,
  `navier_stokes_rk_updated`, which handles the full set of IBM and diagnostic steps.

  Args:
    equation: An `ExplicitNavierStokesODE_BCtime` object containing the full
      set of governing functions for the fluid-structure interaction problem.
    time_step: The time step size, `dt`.

  Returns:
    A `TimeStepFn` that advances the complete simulation state by one time step
    using the Forward Euler method.
  """
  # `jax.named_call` gives this step a name for profiling.
  return jax.named_call(
      # Call the comprehensive RK stepper factory.
      navier_stokes_rk_updated(
          # Use the same simple Butcher Tableau for the Forward Euler method.
          ButcherTableau_updated(a=[], b=[1], c=[0]),
          equation,
          time_step),
      name="forward_euler",
  )
