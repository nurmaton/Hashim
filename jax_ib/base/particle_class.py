import dataclasses
from typing import Any, Callable, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax_ib.base import grids

Array = Union[jnp.ndarray, jnp.ndarray]
PyTree = Any

# --- UNCHANGED UTILITY CLASS ---
# This class for generating a 1D grid of points was simple and effective in both versions.
# It's used to generate the initial Lagrangian marker positions from the shape function.
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

# --- HEAVILY REWRITTEN CORE CLASS ---
#
# OLD vs NEW COMPARISON & EXPLANATION:
#
# WHY THE CHANGE WAS MADE:
# The OLD `particle` class was designed for a kinematically-prescribed rigid body. It stored
# parameters for motion functions (`displacement_param`, `rotation_param`) but did not store
# the actual, evolving state of the particle's boundary points. The NEW version is designed
# for a dynamic, deformable body, so its primary role is to BE the state container.
#
# KEY DIFFERENCES:
# 1. ATTRIBUTES:
#    - OLD: Held parameters and functions (`Displacement_EQ`, `Rotation_EQ`) to CALCULATE
#           the particle's position at any given time `t`.
#    - NEW: Holds the actual, current state variables (`xp`, `yp`, `Ym_x`, `Ym_y`, etc.) that
#           are UPDATED at each time step. It also holds the physical properties (`stiffness`, `sigma`).
#
# 2. JAX PYTREE (`tree_flatten`):
#    - OLD: Only tracked the kinematic parameters as dynamic children.
#    - NEW: Correctly tracks all the dynamic state arrays (positions, velocities) as children.
#           This is CRITICAL for the simulation to work correctly with JAX's compilation and
#           automatic differentiation.
#
# WHY THE NEW METHOD IS BETTER:
#    - It represents the true state of a dynamic object. Its attributes are the variables
#      that evolve according to the equations of motion.
#    - It is much cleaner, having removed all the obsolete kinematic parameters and functions.
#
@register_pytree_node_class
@dataclasses.dataclass
class particle:
    # --- DYNAMIC STATE VARIABLES (The core of the new design) ---
    # These arrays represent the full state of the particle at any given moment.
    xp: jax.numpy.ndarray      # Current fluid marker x-positions (Xm_i in paper)
    yp: jax.numpy.ndarray      # Current fluid marker y-positions (Xm_i in paper)
    Ym_x: jax.numpy.ndarray    # Mass marker x-positions (Ym_i in paper)
    Ym_y: jax.numpy.ndarray    # Mass marker y-positions (Ym_i in paper)
    Vm_x: jax.numpy.ndarray    # Mass marker x-velocities
    Vm_y: jax.numpy.ndarray    # Mass marker y-velocities
    
    # --- PHYSICAL PROPERTIES ---
    # These define the physical nature of the deformable body.
    mass_per_marker: float
    stiffness: float
    sigma: float
    
    # --- STATIC GEOMETRY and INITIALIZATION INFO ---
    # These are used to create the initial state but do not change during the simulation.
    particle_center: Sequence[Any]
    geometry_param: Sequence[Any]
    Grid: Grid1d
    shape: Callable
    
    # --- REMOVED OBSOLETE KINEMATIC ATTRIBUTES ---
    # The old class had: displacement_param, rotation_param, Displacement_EQ, Rotation_EQ.
    # These have been removed because the motion is now CALCULATED, not prescribed.

    def tree_flatten(self):
      """Updated flattening recipe to include all dynamic state variables."""
      children = (self.xp, self.yp, self.Ym_x, self.Ym_y, self.Vm_x, self.Vm_y,
                  self.mass_per_marker, self.stiffness, self.sigma,
                  self.particle_center, self.geometry_param)
      aux_data = (self.Grid, self.shape)
      return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
       """Updated unflattening recipe."""
       return cls(*children, *aux_data)

# --- NEW CONTAINER CLASS ---
# OLD vs NEW: The OLD code passed around a raw Python list of particle objects.
# The NEW approach uses a dedicated JAX Pytree container (`particle_lista`).
# WHY THE NEW METHOD IS BETTER: This makes the code more robust and scalable. By defining
# `tree_flatten` and `tree_unflatten`, we can treat a collection of many particles
# as a single object that JAX can handle efficiently.
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

# --- UPDATED CONTAINER CLASS ---
# OLD vs NEW: This class is structurally similar, but its `particles` attribute
# now holds the new `particle_lista` object instead of a simple list. Its `tree_flatten`
# method ensures that this container is also a valid JAX pytree.
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
