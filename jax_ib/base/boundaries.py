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
"""Classes that specify how boundary conditions are applied to arrays."""

import dataclasses
from typing import Any, Callable, Iterable, Sequence, Tuple, Optional, Union
from jax import lax
import jax
import jax.numpy as jnp
from jax_ib.base import grids
import numpy as np
import scipy
from jax.tree_util import register_pytree_node_class
from jax_ib.base import particle_class

BoundaryConditions = grids.BoundaryConditions
GridArray = grids.GridArray
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
Array = Union[np.ndarray, jax.Array]
BCArray = grids.BCArray


class BCType:
  PERIODIC = 'periodic'
  DIRICHLET = 'dirichlet'
  NEUMANN = 'neumann'

@register_pytree_node_class
@dataclasses.dataclass(init=False, frozen=False)
class ConstantBoundaryConditions(BoundaryConditions):
  """Boundary conditions for a PDE variable that are constant in space and time."""
  types: Tuple[Tuple[str, str], ...]
  bc_values: Tuple[Tuple[Optional[float], Optional[float]], ...]
  boundary_fn: Callable[...,Optional[float]]
  time_stamp: Optional[float]
  def __init__(self,
               time_stamp: Optional[float],values: Sequence[Tuple[Optional[float], Optional[float]]],types: Sequence[Tuple[str, str]],boundary_fn:Callable[...,Optional[float]]):
    types = tuple(types)
    values = tuple(values)
    boundary_fn = boundary_fn
    time_stamp = time_stamp

    object.__setattr__(self, 'bc_values', values)
    object.__setattr__(self, 'boundary_fn', boundary_fn)
    object.__setattr__(self, 'time_stamp', time_stamp if time_stamp is not None else [])
    object.__setattr__(self, 'types', types)


  def tree_flatten(self):
    """Returns flattening recipe for GridVariable JAX pytree."""
    children = (self.time_stamp,self.bc_values,)
    aux_data = (self.types,self.boundary_fn)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Returns unflattening recipe for GridVariable JAX pytree."""
    return cls(*children, *aux_data)


  def update_bc_(self,time_stamp: float, dt: float):
    return time_stamp + dt


  def shift(
      self,
      u: GridArray,
      offset: int,
      axis: int,
  ) -> GridArray:
    """Shift an GridArray by `offset`."""
    padded = self._pad(u, offset, axis)
    trimmed = self._trim(padded, -offset, axis)
    return trimmed

  def _pad(
      self,
      u: GridArray,
      width: int,
      axis: int,
  ) -> GridArray:
    """Pad a GridArray."""

    def make_padding(width):
      if width < 0:  # pad lower boundary
        bc_type = self.types[axis][0]
        padding = (-width, 0)
      else:  # pad upper boundary
        bc_type = self.types[axis][1]
        padding = (0, width)

      full_padding = [(0, 0)] * u.grid.ndim
      full_padding[axis] = padding
      return full_padding, padding, bc_type

    full_padding, padding, bc_type = make_padding(width)
    offset = list(u.offset)
    offset[axis] -= padding[0]
    if not (bc_type == BCType.PERIODIC or
            bc_type == BCType.DIRICHLET) and abs(width) > 1:
      raise ValueError(
          f'Padding past 1 ghost cell is not defined in {bc_type} case.')

    u, trimmed_padding = self._trim_padding(u)
    data = u.data
    full_padding[axis] = tuple(
        pad + trimmed_pad
        for pad, trimmed_pad in zip(full_padding[axis], trimmed_padding))

    if bc_type == BCType.PERIODIC:
      if u.grid.shape[axis] > u.shape[axis]:
        raise ValueError('the GridArray shape does not match the grid.')
      pad_kwargs = dict(mode='wrap')
      data = jnp.pad(data, full_padding, **pad_kwargs)

    elif bc_type == BCType.DIRICHLET:
      if np.isclose(u.offset[axis] % 1, 0.5):  # cell center
        if u.grid.shape[axis] > u.shape[axis]:
          raise ValueError('the GridArray shape does not match the grid.')
        data = (2 * jnp.pad(
            data, full_padding, mode='constant', constant_values=self.bc_values)
                - jnp.pad(data, full_padding, mode='symmetric'))
      elif np.isclose(u.offset[axis] % 1, 0):  # cell edge
        if u.grid.shape[axis] > u.shape[axis] + 1:
          raise ValueError('For a dirichlet cell-face boundary condition, ' +
                           'the GridArray has more than 1 grid point missing.')
        elif u.grid.shape[axis] == u.shape[axis] + 1 and not np.isclose(
            u.offset[axis], 1):
          raise ValueError('For a dirichlet cell-face boundary condition, ' +
                           'the GridArray has more than 1 grid point missing.')

        def _needs_pad_with_boundary_value():
          if (np.isclose(u.offset[axis], 0) and
              width > 0) or (np.isclose(u.offset[axis], 1) and width < 0):
            return True
          elif u.grid.shape[axis] == u.shape[axis] + 1:
            return True
          else:
            return False

        if _needs_pad_with_boundary_value():
          if np.isclose(abs(width), 1):
            data = jnp.pad(
                data,
                full_padding,
                mode='constant',
                constant_values=self.bc_values)
          elif abs(width) > 1:
            bc_padding, _, _ = make_padding(int(width /
                                                abs(width)))
            full_padding_past_bc, _, _ = make_padding(
                (abs(width) - 1) * int(width / abs(width)))
            expanded_data = jnp.pad(
                data, bc_padding, mode='constant', constant_values=(0, 0))
            padding_values = list(self.bc_values)
            padding_values[axis] = [pad / 2 for pad in padding_values[axis]]
            data = 2 * jnp.pad(
                data,
                full_padding,
                mode='constant',
                constant_values=tuple(padding_values)) - jnp.pad(
                    expanded_data, full_padding_past_bc, mode='reflect')
        else:
          padding_values = list(self.bc_values)
          padding_values[axis] = [pad / 2 for pad in padding_values[axis]]
          data = 2 * jnp.pad(
              data,
              full_padding,
              mode='constant',
              constant_values=tuple(padding_values)) - jnp.pad(
                  data, full_padding, mode='reflect')
      else:
        raise ValueError('expected offset to be an edge or cell center, got '
                         f'offset[axis]={u.offset[axis]}')
    elif bc_type == BCType.NEUMANN:
      if u.grid.shape[axis] > u.shape[axis]:
        raise ValueError('the GridArray shape does not match the grid.')
      if not (np.isclose(u.offset[axis] % 1, 0) or
              np.isclose(u.offset[axis] % 1, 0.5)):
        raise ValueError('expected offset to be an edge or cell center, got '
                         f'offset[axis]={u.offset[axis]}')
      else:
        data = (
            jnp.pad(data, full_padding, mode='edge') + u.grid.step[axis] *
            (jnp.pad(data, full_padding, mode='constant') - jnp.pad(
                data,
                full_padding,
                mode='constant',
                constant_values=self.bc_values)))
    else:
      raise ValueError('invalid boundary type')

    return GridArray(data, tuple(offset), u.grid)

  def _trim(
      self,
      u: GridArray,
      width: int,
      axis: int,
  ) -> GridArray:
    """Trim padding from a GridArray."""
    if width < 0:
      padding = (-width, 0)
    else:
      padding = (0, width)

    limit_index = u.data.shape[axis] - padding[1]
    data = lax.slice_in_dim(u.data, padding[0], limit_index, axis=axis)
    offset = list(u.offset)
    offset[axis] += padding[0]
    return GridArray(data, tuple(offset), u.grid)

  def _trim_padding(self, u: grids.GridArray, axis=0):
    """Trim all padding from a GridArray."""
    padding = (0, 0)
    if u.shape[axis] > u.grid.shape[axis]:
      negative_trim = 0
      if u.offset[axis] < 0:
        negative_trim = -round(-u.offset[axis])
        u = self._trim(u, negative_trim, axis)
      positive_trim = u.shape[axis] - u.grid.shape[axis]
      if positive_trim > 0:
        u = self._trim(u, positive_trim, axis)
      padding = (negative_trim, positive_trim)
    return u, padding

  def values(
      self, axis: int,
      grid: grids.Grid) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """Returns boundary values on the grid along axis."""
    if None in self.bc_values[axis]:
      return (None, None)
    bc_values = tuple(
        jnp.full(grid.shape[:axis] +
                 grid.shape[axis + 1:], self.bc_values[axis][-i])
        for i in [0, 1])
    return bc_values

  def trim_boundary(self, u: grids.GridArray) -> grids.GridArray:
    """Returns GridArray without the grid points on the boundary."""
    for axis in range(u.grid.ndim):
      u, _ = self._trim_padding(u, axis)
    if u.shape != u.grid.shape:
      raise ValueError('the GridArray has already been trimmed.')
    for axis in range(u.grid.ndim):
      if np.isclose(u.offset[axis],
                    0.0) and self.types[axis][0] == BCType.DIRICHLET:
        u = self._trim(u, -1, axis)
      elif np.isclose(u.offset[axis],
                      1.0) and self.types[axis][1] == BCType.DIRICHLET:
        u = self._trim(u, 1, axis)
    return u

  def pad_and_impose_bc(
      self,
      u: grids.GridArray,
      offset_to_pad_to: Optional[Tuple[float,
                                       ...]] = None) -> grids.GridVariable:
    """Returns GridVariable with correct boundary condition."""
    if offset_to_pad_to is None:
      offset_to_pad_to = u.offset
    for axis in range(u.grid.ndim):
      if self.types[axis][0] == BCType.DIRICHLET and np.isclose(
          u.offset[axis], 1.0):
        if np.isclose(offset_to_pad_to[axis], 1.0):
          u = self._pad(u, 1, axis)
        elif np.isclose(offset_to_pad_to[axis], 0.0):
          u = self._pad(u, -1, axis)
    return grids.GridVariable(u, self)

  def impose_bc(self, u: grids.GridArray) -> grids.GridVariable:
    """Returns GridVariable with correct boundary condition."""
    offset = u.offset
    if u.shape == u.grid.shape:
      u = self.trim_boundary(u)
    return self.pad_and_impose_bc(u, offset)

  trim = _trim
  pad = _pad


@register_pytree_node_class
class HomogeneousBoundaryConditions(ConstantBoundaryConditions):
  """Boundary conditions for a PDE variable with zero value or flux."""

  def __init__(self, types: Sequence[Tuple[str, str]]):
    ndim = len(types)
    values = ((0.0, 0.0),) * ndim
    bc_fn = lambda x: x
    time_stamp = 0.0
    super(HomogeneousBoundaryConditions, self).__init__(time_stamp, values, types, bc_fn)

  def tree_flatten(self):
      """Correct flattening recipe for this simpler class."""
      children = ()
      aux_data = (self.types,)
      return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
      """Correct unflattening recipe for this simpler class."""
      return cls(*aux_data)


@register_pytree_node_class
class TimeDependentBoundaryConditions(ConstantBoundaryConditions):
  """Boundary conditions for a PDE variable."""

  def __init__(self, types: Sequence[Tuple[str, str]],values: Sequence[Tuple[Optional[float], Optional[float]]],boundary_fn: Callable[..., Optional[float]],time_stamp: Optional[float]):
    super(TimeDependentBoundaryConditions, self).__init__(types, values,boundary_fn,time_stamp)

  def tree_flatten(self):
    """Returns flattening recipe for GridVariable JAX pytree."""
    children = (self.bc_values,)
    aux_data = (self.time_stamp,self.types,self.boundary_fn,)
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    """Returns unflattening recipe for GridVariable JAX pytree."""
    return cls(*children, *aux_data)


def boundary_function(t):
  A=1
  B = 1
  freq = 1
  return 1+0*(A*jnp.cos(freq*t)+B*jnp.sin(freq*t))

# --- MODIFIED FUNCTIONS TO FIX THE BUG ---

def Reserve_BC(all_variable: particle_class.All_Variables, step_time: float) -> particle_class.All_Variables:
    """
    Pass-through function. Obsolete for static BCs in the deformable model.
    """
    return all_variable

def update_BC(all_variable: particle_class.All_Variables, step_time: float) -> particle_class.All_Variables:
    """
    Pass-through function. The logic for time-dependent BCs is not needed
    for the current simulation setup with static periodic boundaries.
    """
    # The time_stamp of the velocity field's boundary condition will be updated
    # implicitly through the PyTree mechanics. We don't need to manually
    # reconstruct the boundary condition object here.
    return all_variable

# --- END MODIFIED FUNCTIONS ---

def periodic_boundary_conditions(ndim: int) -> ConstantBoundaryConditions:
  """Returns periodic BCs for a variable with `ndim` spatial dimension."""
  return HomogeneousBoundaryConditions(
      ((BCType.PERIODIC, BCType.PERIODIC),) * ndim)

def Radom_velocity_conditions(ndim: int) -> ConstantBoundaryConditions:
    values = ((0.0, 0.0),) * ndim
    bc_fn = lambda x: x
    time_stamp = 0.0
    return Moving_wall_boundary_conditions(
    ndim,
    bc_vals=values,
    time_stamp=time_stamp,
    bc_fn=bc_fn,)

def dirichlet_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> ConstantBoundaryConditions:
  if not bc_vals:
    return HomogeneousBoundaryConditions(
        ((BCType.DIRICHLET, BCType.DIRICHLET),) * ndim)
  else:
    return ConstantBoundaryConditions(
        ((BCType.DIRICHLET, BCType.DIRICHLET),) * ndim, bc_vals)

def neumann_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> ConstantBoundaryConditions:
  if not bc_vals:
    return HomogeneousBoundaryConditions(
        ((BCType.NEUMANN, BCType.NEUMANN),) * ndim)
  else:
    return ConstantBoundaryConditions(
        ((BCType.NEUMANN, BCType.NEUMANN),) * ndim, bc_vals)

def channel_flow_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> ConstantBoundaryConditions:
  bc_type = ((BCType.PERIODIC, BCType.PERIODIC),
             (BCType.DIRICHLET, BCType.DIRICHLET))
  for _ in range(ndim - 2):
    bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)
  if not bc_vals:
    return HomogeneousBoundaryConditions(bc_type)
  else:
    return ConstantBoundaryConditions(bc_type, bc_vals)

def Moving_wall_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]],
    time_stamp: Optional[float],
    bc_fn: Callable[...,Optional[float]],
) -> ConstantBoundaryConditions:
  bc_type = ((BCType.PERIODIC, BCType.PERIODIC),
             (BCType.DIRICHLET, BCType.DIRICHLET))
  for _ in range(ndim - 2):
    bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)
  return ConstantBoundaryConditions(values=bc_vals,time_stamp=time_stamp,types=bc_type,boundary_fn=bc_fn)

def Far_field_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]],
    time_stamp: Optional[float],
    bc_fn: Callable[...,Optional[float]],
) -> ConstantBoundaryConditions:
  bc_type = ((BCType.DIRICHLET, BCType.DIRICHLET),
             (BCType.DIRICHLET, BCType.DIRICHLET))
  for _ in range(ndim - 2):
    bc_type += ((BCType.DIRICHLET, BCType.DIRICHLET),)
  return ConstantBoundaryConditions(values=bc_vals,time_stamp=time_stamp,types=bc_type,boundary_fn=bc_fn)

def find_extremum(fn,extrema,i_guess):
    if extrema == 'maximum':
      direc = -1
    elif extrema == 'minimum':
      direc = 1
    else:
      raise ValueError('No extrema was correctly identified. For maximum, type "maiximum". For minimization, type "minimum". ')
    return fn(scipy.optimize.fmin(lambda x: direc*fn(x), i_guess))

def periodic_and_neumann_boundary_conditions(
    bc_vals: Optional[Tuple[float,
                            float]] = None) -> ConstantBoundaryConditions:
  if not bc_vals:
    return HomogeneousBoundaryConditions(
        ((BCType.PERIODIC, BCType.PERIODIC), (BCType.NEUMANN, BCType.NEUMANN)))
  else:
    return ConstantBoundaryConditions(
        ((BCType.PERIODIC, BCType.PERIODIC), (BCType.NEUMANN, BCType.NEUMANN)),
        ((None, None), bc_vals))

def periodic_and_dirichlet_boundary_conditions(
    bc_vals: Optional[Tuple[float, float]] = None,
    periodic_axis=0) -> ConstantBoundaryConditions:
  periodic = (BCType.PERIODIC, BCType.PERIODIC)
  dirichlet = (BCType.DIRICHLET, BCType.DIRICHLET)
  if periodic_axis == 0:
    if not bc_vals:
      return HomogeneousBoundaryConditions((periodic, dirichlet))
    else:
      return ConstantBoundaryConditions((periodic, dirichlet),
                                        ((None, None), bc_vals))
  else:
    if not bc_vals:
      return HomogeneousBoundaryConditions((dirichlet, periodic))
    else:
      return ConstantBoundaryConditions((dirichlet, periodic),
                                        (bc_vals, (None, None)))

def is_periodic_boundary_conditions(c: grids.GridVariable, axis: int) -> bool:
  if c.bc.types[axis][0] != BCType.PERIODIC:
    return False
  return True

def has_all_periodic_boundary_conditions(*arrays: GridVariable) -> bool:
  for array in arrays:
    for axis in range(array.grid.ndim):
      if not is_periodic_boundary_conditions(array, axis):
        return False
  return True

def consistent_boundary_conditions(*arrays: GridVariable) -> Tuple[str, ...]:
  bc_types = []
  for axis in range(arrays[0].grid.ndim):
    bcs = {is_periodic_boundary_conditions(array, axis) for array in arrays}
    if len(bcs) != 1:
      raise grids.InconsistentBoundaryConditionsError(
          f'arrays do not have consistent bc: {arrays}')
    elif bcs.pop():
      bc_types.append('periodic')
    else:
      bc_types.append('nonperiodic')
  return tuple(bc_types)

def get_pressure_bc_from_velocity(v: GridVariableVector) -> BoundaryConditions:
  velocity_bc_types = consistent_boundary_conditions(*v)
  pressure_bc_types = []
  bc_value = ((0.0,0.0),(0.0,0.0))
  # Using a simple lambda for the boundary function as it's not time-dependent.
  Bc_f = lambda x: x
  for velocity_bc_type in velocity_bc_types:
    if velocity_bc_type == 'periodic':
      pressure_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
    else:
      pressure_bc_types.append((BCType.NEUMANN, BCType.NEUMANN))
  # The time_stamp can be a fixed value as it's not used for periodic BCs.
  return ConstantBoundaryConditions(values=bc_value,time_stamp=0.0,types=pressure_bc_types,boundary_fn=Bc_f)

def get_advection_flux_bc_from_velocity_and_scalar(
    u: GridVariable, c: GridVariable,
    flux_direction: int) -> BoundaryConditions:
  flux_bc_types = []
  if not isinstance(u.bc, ConstantBoundaryConditions):
    raise NotImplementedError(
        f'Flux boundary condition is not implemented for {u.bc, c.bc}')
  for axis in range(c.grid.ndim):
    if u.bc.types[axis][0] == 'periodic':
      flux_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
    elif flux_direction != axis:
      flux_bc_types.append((BCType.DIRICHLET, BCType.DIRICHLET))
    elif (u.bc.types[axis][0] == BCType.DIRICHLET and
          u.bc.types[axis][1] == BCType.DIRICHLET and
          u.bc.bc_values[axis][0] == 0.0 and u.bc.bc_values[axis][1] == 0.0):
      flux_bc_types.append((BCType.DIRICHLET, BCType.DIRICHLET))
    else:
      raise NotImplementedError(
          f'Flux boundary condition is not implemented for {u.bc, c.bc}')
  return HomogeneousBoundaryConditions(flux_bc_types)

def new_periodic_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]],
    time_stamp: Optional[float],
    bc_fn: Callable[...,Optional[float]],
) -> ConstantBoundaryConditions:
  bc_type = ((BCType.PERIODIC, BCType.PERIODIC),
             (BCType.PERIODIC, BCType.PERIODIC))
  for _ in range(ndim - 2):
    bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)
  return ConstantBoundaryConditions(values=bc_vals,time_stamp=time_stamp,types=bc_type,boundary_fn=bc_fn)
