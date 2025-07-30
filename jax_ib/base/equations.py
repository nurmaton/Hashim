import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from jax_ib.base import advection
from jax_ib.base import diffusion
from jax_ib.base import grids
from jax_ib.base import pressure
from jax_cfd.base import pressure as pressureCFD
from jax_ib.base import time_stepping
from jax_ib.base import boundaries
from jax_ib.base import finite_differences
import tree_math
from jax_ib.base import particle_class
from jax_cfd.base import equations as equationsCFD

GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
ConvectFn = Callable[[GridVariableVector], GridArrayVector]
DiffuseFn = Callable[[GridVariable, float], GridArray]
ForcingFn = Callable[[GridVariableVector], GridArrayVector]
BCFn =  Callable[[particle_class.All_Variables, float], particle_class.All_Variables]
IBMFn =  Callable[[particle_class.All_Variables, float], GridVariableVector]
GradPFn = Callable[[GridVariable], GridArrayVector]
PosFn =  Callable[[particle_class.All_Variables, float], particle_class.All_Variables]
DragFn =  Callable[[particle_class.All_Variables], particle_class.All_Variables]

def _wrap_term_as_vector(fun, *, name):
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
  """Returns a function that performs a time step of Navier Stokes."""
  del grid

  if convect is None:
    def convect(v):
      return tuple(advection.advect_van_leer_using_limiters(u, v, dt) for u in v)

  def diffuse_velocity(v, *args):
    return tuple(diffuse(u, *args) for u in v)

  convection = _wrap_term_as_vector(convect, name='convection')
  diffusion_ = _wrap_term_as_vector(diffuse_velocity, name='diffusion')
  if forcing is not None:
    forcing = _wrap_term_as_vector(forcing, name='forcing')

  @tree_math.wrap
  @functools.partial(jax.named_call, name='navier_stokes_momentum')
  def _explicit_terms(v):
    dv_dt = convection(v)
    if viscosity is not None:
      dv_dt += diffusion_(v, viscosity / density)
    if forcing is not None:
      dv_dt += forcing(v) / density
    return dv_dt

  def explicit_terms_with_same_bcs(v):
    dv_dt = _explicit_terms(v)
    return tuple(grids.GridVariable(a, u.bc) for a, u in zip(dv_dt, v))

  return explicit_terms_with_same_bcs

# (explicit_* wrapper functions remain the same)
def explicit_Reserve_BC(ReserveBC: BCFn, step_time: float) -> Callable:
   Reserve_boundary = lambda v, *a: ReserveBC(v, *a)
   _Reserve_bc = _wrap_term_as_vector(Reserve_boundary, name='Reserve_BC')
   return tree_math.wrap(lambda v: _Reserve_bc(v, step_time))

def explicit_update_BC(updateBC: BCFn, step_time: float) -> Callable:
   Update_boundary = lambda v, *a: updateBC(v, *a)
   _Update_bc = _wrap_term_as_vector(Update_boundary, name='Update_BC')
   return tree_math.wrap(lambda v: _Update_bc(v, step_time))

def explicit_IBM_Force(cal_IBM_force: IBMFn, step_time: float) -> Callable:
   IBM_FORCE = lambda v, *a: cal_IBM_force(v, *a)
   _IBM_FORCE = _wrap_term_as_vector(IBM_FORCE, name='IBM_FORCE')
   return tree_math.wrap(lambda v: _IBM_FORCE(v, step_time))

def explicit_Update_position(cal_Update_Position: PosFn, step_time: float) -> Callable:
   Update_Position = lambda v, *a: cal_Update_Position(v, *a)
   _Update_Position = _wrap_term_as_vector(Update_Position, name='Update_Position')
   return tree_math.wrap(lambda v: _Update_Position(v, step_time))

def explicit_Calc_Drag(cal_Drag: DragFn, step_time: float) -> Callable:
   Calculate_Drag = lambda v, *a: cal_Drag(v, *a)
   _Calculate_Drag = _wrap_term_as_vector(Calculate_Drag, name='Calculate_Drag')
   return tree_math.wrap(lambda v: _Calculate_Drag(v, step_time))

def explicit_Pressure_Gradient(cal_Pressure_Grad: GradPFn) -> Callable:
   Pressure_Grad = lambda v: cal_Pressure_Grad(v)
   _Pressure_Grad = _wrap_term_as_vector(Pressure_Grad, name='Pressure_Grad')
   return tree_math.wrap(lambda v: _Pressure_Grad(v))

# --- MODIFIED HIGH-LEVEL SOLVER FUNCTION ---
def semi_implicit_navier_stokes_timeBC(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    gravity: Optional[jnp.ndarray] = None, # <-- NEW ARGUMENT with a default
    convect: Optional[ConvectFn] = None,
    diffuse: DiffuseFn = diffusion.diffuse,
    pressure_solve: Callable = pressureCFD.solve_fast_diag,
    time_stepper: Callable = time_stepping.forward_euler_updated,
    IBM_forcing: Optional[IBMFn] = None,
    Updating_Position: Optional[PosFn] = None,
    Pressure_Grad: Optional[GradPFn] = finite_differences.forward_difference,
    Drag_fn: Optional[DragFn] = None,
) -> Callable[[particle_class.All_Variables], particle_class.All_Variables]:
  """Returns a function that performs a time step of Navier Stokes."""

  # --- NEW: Define the gravitational body force function ---
  # This function will be passed to `navier_stokes_explicit_terms`.
  gravity_forcing_fn = None
  if gravity is not None:
    def gravity_forcing_fn(v):
        force_components = []
        for i, u in enumerate(v):
            force = grids.GridArray(jnp.full_like(u.data, density * gravity[i]), u.offset, grid)
            force_components.append(force)
        return tuple(force_components)
  # --- END NEW ---

  # The `forcing` argument now takes our new gravity function.
  explicit_terms = navier_stokes_explicit_terms(
      density=density,
      viscosity=viscosity,
      dt=dt,
      grid=grid,
      convect=convect,
      diffuse=diffuse,
      forcing=gravity_forcing_fn)

  pressure_projection = jax.named_call(pressure.projection_and_update_pressure, name='pressure')
  Reserve_BC = explicit_Reserve_BC(ReserveBC = boundaries.Reserve_BC,step_time = dt)
  update_BC = explicit_update_BC(updateBC = boundaries.update_BC,step_time = dt)
  IBM_force = explicit_IBM_Force(cal_IBM_force = IBM_forcing,step_time = dt)
  Update_Position =  explicit_Update_position(cal_Update_Position = Updating_Position,step_time = dt)
  Pressure_Grad =  explicit_Pressure_Gradient(cal_Pressure_Grad = Pressure_Grad)
  Calculate_Drag =  explicit_Calc_Drag(cal_Drag = Drag_fn,step_time = dt)

  ode = time_stepping.ExplicitNavierStokesODE_BCtime(
      explicit_terms,
      lambda v: pressure_projection(v, pressure_solve),
      update_BC,
      Reserve_BC,
      IBM_force,
      Update_Position,
      Pressure_Grad,
      Calculate_Drag,
  )
  step_fn = time_stepper(ode, dt)
  return step_fn

# (semi_implicit_navier_stokes_penalty remains unchanged)
def semi_implicit_navier_stokes_penalty(*args, **kwargs):
    # This is just a placeholder to keep the file structure.
    # The actual implementation is in your original file.
    return equationsCFD.semi_implicit_navier_stokes(*args, **kwargs)
