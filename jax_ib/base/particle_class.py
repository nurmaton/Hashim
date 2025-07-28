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

@dataclasses.dataclass(init=False, frozen=True)
class Grid1d:
    shape: Tuple[int, ...]
    step: Tuple[float, ...]
    domain: Tuple[Tuple[float, float], ...]

    def __init__(
        self,
        shape: Sequence[int],
        step: Optional[Union[float, Sequence[float]]] = None,
        domain: Optional[Union[float, Sequence[Tuple[float, float]]]] = None,
    ):
        shape = shape
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'domain', domain)
        step = (domain[1] - domain[0]) / (shape-1)
        object.__setattr__(self, 'step', step)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def cell_center(self) -> Tuple[float, ...]:
        return self.ndim * (0.5,)

    def axes(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
        if offset is None:
            offset = self.cell_center
        if len(offset) != self.ndim:
            raise ValueError(f'unexpected offset length: {len(offset)} vs {self.ndim}')
        return self.domain[0] + jnp.arange(self.shape)*self.step

    def mesh(self, offset: Optional[Sequence[float]] = None) -> Tuple[Array, ...]:
        return self.axes(offset)

@register_pytree_node_class
@dataclasses.dataclass
class particle:
    particle_center: Sequence[Any]
    geometry_param: Sequence[Any]
    displacement_param: Sequence[Any]
    rotation_param: Sequence[Any]
    Grid: Grid1d
    shape: Callable
    Displacement_EQ: Callable
    Rotation_EQ: Callable

    # We do NOT add marker_positions here, because JAX pytree wants it frozen.
    # Instead, attach it as a mutable attribute after construction.

    def tree_flatten(self):
        children = (self.particle_center, self.geometry_param, self.displacement_param, self.rotation_param,)
        aux_data = (self.Grid, self.shape, self.Displacement_EQ, self.Rotation_EQ,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

    def generate_grid(self):
        return self.Grid.mesh()

    def calc_Rtheta(self):
        return self.shape(self.geometry_param, self.Grid)

    # === NEW: Marker positions at current time ===
    def marker_positions(self, current_t):
        """
        Returns marker positions (Nmarkers, 2) for current particle at given time.
        """
        # Calculate boundary coordinates in (xp0, yp0)
        xp0, yp0 = self.shape(self.geometry_param, self.Grid)
        theta = self.Rotation_EQ(self.rotation_param, current_t)
        xc, yc = self.particle_center[0]
        xp = xp0 * jnp.cos(theta) - yp0 * jnp.sin(theta) + xc
        yp = xp0 * jnp.sin(theta) + yp0 * jnp.cos(theta) + yc
        return jnp.stack([xp, yp], axis=1)  # shape (Nmarkers, 2)

@register_pytree_node_class
@dataclasses.dataclass
class All_Variables:
    particles: Sequence[particle,]
    velocity: grids.GridVariableVector
    pressure: grids.GridVariable
    Drag: Sequence[Any]
    Step_count: int
    MD_var: Any
    def tree_flatten(self):
        children = (self.particles, self.velocity, self.pressure, self.Drag, self.Step_count, self.MD_var,)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@register_pytree_node_class
@dataclasses.dataclass
class particle_lista:
    particles: Sequence[particle,]

    def generate_grid(self):
        return np.stack([grid.mesh() for grid in self.Grid])

    def calc_Rtheta(self):
        return self.shape(self.geometry_param, self.Grid)

    def tree_flatten(self):
        children = (*self.particles,)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
