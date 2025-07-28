import dataclasses
import numbers
import operator
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np
from jax_ib.base import grids


Array = Union[np.ndarray, jax.Array]
IntOrSequence = Union[int, Sequence[int]]
PyTree = Any

# (Grid1d class remains the same)
@dataclasses.dataclass(init=False, frozen=True)
class Grid1d:
    shape: Tuple[int, ...]
    step: Tuple[float, ...]
    domain: Tuple[Tuple[float, float], ...]
    def __init__(self, shape: Sequence[int], step: Optional[Union[float, Sequence[float]]] = None, domain: Optional[Union[float, Sequence[Tuple[float, float]]]] = None):
        shape = shape
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'domain', domain)
        step = (domain[1] - domain[0]) / (shape-1) 
        object.__setattr__(self, 'step', step)
    @property
    def ndim(self) -> int: return 1
    @property
    def cell_center(self) -> Tuple[float, ...]: return self.ndim * (0.5,)
    def axes(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
        if offset is None: offset = self.cell_center
        if len(offset) != self.ndim: raise ValueError(f'unexpected offset length: {len(offset)} vs {self.ndim}')
        return self.domain[0] + jnp.arange(self.shape)*self.step
    def mesh(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]: return self.axes(offset)


@register_pytree_node_class
@dataclasses.dataclass
class particle:
    # --- NEW DYNAMIC STATE VARIABLES ---
    xp: jax.numpy.ndarray          # Current fluid marker x-positions (Xm_i in paper)
    yp: jax.numpy.ndarray          # Current fluid marker y-positions (Xm_i in paper)
    Ym_x: jax.numpy.ndarray        # Mass marker x-positions (Ym_i in paper)
    Ym_y: jax.numpy.ndarray        # Mass marker y-positions (Ym_i in paper)
    Vm_x: jax.numpy.ndarray        # Mass marker x-velocities
    Vm_y: jax.numpy.ndarray        # Mass marker y-velocities
    
    # --- NEW PHYSICAL PROPERTIES ---
    mass_per_marker: float         # Mass M for each marker
    stiffness: float               # Spring stiffness Kp
    sigma: float                   # Surface tension coefficient
    
    # --- Kept for initialization and other potential uses ---
    particle_center: Sequence[Any]
    geometry_param: Sequence[Any]
    Grid: Grid1d
    shape: Callable
    
    def tree_flatten(self):
      """Returns flattening recipe for JAX pytree."""
      children = (self.xp, self.yp, self.Ym_x, self.Ym_y, self.Vm_x, self.Vm_y,
                  self.mass_per_marker, self.stiffness, self.sigma,
                  self.particle_center, self.geometry_param)
      aux_data = (self.Grid, self.shape)
      return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
       """Returns unflattening recipe for JAX pytree."""
       return cls(*children, *aux_data)

    def generate_grid(self):
        return self.Grid.mesh()
       
    def calc_Rtheta(self):
      return self.shape(self.geometry_param,self.Grid)


@register_pytree_node_class
@dataclasses.dataclass
class All_Variables: 
    particles: Sequence[particle,]
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
