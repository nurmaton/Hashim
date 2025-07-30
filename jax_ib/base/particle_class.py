import dataclasses
from typing import Any, Callable, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax_ib.base import grids

Array = Union[jnp.ndarray, jnp.ndarray]
PyTree = Any

@dataclasses.dataclass(init=False, frozen=True)
class Grid1d:
    shape: Tuple[int, ...]
    step: Tuple[float, ...]
    domain: Tuple[Tuple[float, float], ...]
    def __init__(self, shape: Sequence[int], domain: Tuple[float, float]):
        shape = shape
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'domain', domain)
        step = (domain[1] - domain[0]) / (shape - 1)
        object.__setattr__(self, 'step', step)
    @property
    def ndim(self) -> int: return 1
    def mesh(self, offset=None) -> Tuple[Array, ...]:
        return self.domain[0] + jnp.arange(self.shape) * self.step

@register_pytree_node_class
@dataclasses.dataclass
class particle:
    # --- DYNAMIC STATE VARIABLES ---
    xp: jax.numpy.ndarray
    yp: jax.numpy.ndarray
    Ym_x: jax.numpy.ndarray
    Ym_y: jax.numpy.ndarray
    Vm_x: jax.numpy.ndarray
    Vm_y: jax.numpy.ndarray
    
    # --- PHYSICAL PROPERTIES ---
    mass_per_marker: float
    stiffness: float
    sigma: float
    
    # --- STATIC GEOMETRY and INITIALIZATION INFO ---
    particle_center: Sequence[Any]
    geometry_param: Sequence[Any]
    Grid: Grid1d
    shape: Callable

    # --- OBSOLETE KINEMATIC ATTRIBUTES ---
    # displacement_param, rotation_param, Displacement_EQ, Rotation_EQ not needed.
    # displacement_param: Sequence[Any]
    # rotation_param: Sequence[Any]
    # Displacement_EQ: Callable
    # Rotation_EQ: Callable

    def tree_flatten(self):
      """Updated flattening recipe."""
      children = (self.xp, self.yp, self.Ym_x, self.Ym_y, self.Vm_x, self.Vm_y,
                  self.mass_per_marker, self.stiffness, self.sigma,
                  self.particle_center, self.geometry_param)
      aux_data = (self.Grid, self.shape)
      return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
       """Updated unflattening recipe."""
       return cls(*children, *aux_data)

@register_pytree_node_class
@dataclasses.dataclass
class particle_lista:
    particles: Sequence[particle,]
    
    def tree_flatten(self):
      children = tuple(self.particles)
      aux_data = None
      return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
       return cls(particles=list(children))

@register_pytree_node_class
@dataclasses.dataclass
class All_Variables: 
    particles: particle_lista
    velocity: grids.GridVariableVector
    pressure: grids.GridVariable
    Drag:Sequence[Any]
    Step_count:int
    MD_var:Any
    def tree_flatten(self):
      children = (self.particles,self.velocity,self.pressure,self.Drag,self.Step_count,self.MD_var,)
      aux_data = None
      return children, aux_data
    @classmethod
    def tree_unflatten(cls, aux_data, children):
       return cls(*children)
