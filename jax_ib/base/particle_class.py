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
Defines the data structures for representing particles in the simulation.

This module contains the core data classes for managing the state of dynamic,
deformable particles. These classes are fundamental to the shift from a
kinematically-driven simulation (where motion is prescribed) to a fully
dynamic, physics-based Fluid-Structure Interaction (FSI) simulation where the
particle's motion is an outcome of the forces acting on it.

The physical model is based on the "penalty IBM" method described by Sustiel &
Grier, which uses two sets of Lagrangian markers:
-   **Mass-carrying markers (`Ym_x`, `Ym_y`)**: These hold the particle's inertia
    and are evolved by a Molecular Dynamics-style integrator.
-   **Fluid-interacting markers (`xp`, `yp`)**: These are massless points that
    define the boundary and are subject to penalty and fluid forces.

A key feature of this module is that all data containers (`particle`,
`particle_lista`, `All_Variables`) are registered as **JAX PyTrees**. This is
what allows the complex, nested state of the simulation to be passed into JAX's
high-performance transformations (`jit`, `vmap`, `scan`, `grad`), enabling an
efficient and fully differentiable simulation pipeline.
"""

import dataclasses
from typing import Any, Callable, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax_ib.base import grids

# Type aliases for clarity.
Array = Union[jnp.ndarray, jnp.ndarray]
PyTree = Any

# --- UNCHANGED UTILITY CLASS ---
@dataclasses.dataclass(init=False, frozen=True)
class Grid1d:
    """
    A simple 1D grid generator.

    This utility class creates a 1D grid of uniformly spaced points within a
    specified domain. Its primary use here is to generate the initial set of
    Lagrangian marker points that define the particle's shape at the start of
    the simulation. Its function is identical to the old version.
    """
    shape: Tuple[int, ...]
    step: Tuple[float, ...]
    domain: Tuple[Tuple[float, float], ...]

    def __init__(self, shape: Sequence[int], domain: Tuple[float, float]):
        """Initializes the 1D grid."""
        shape = shape
        # Use object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'domain', domain)
        # Calculate the step size between points.
        step = (domain[1] - domain[0]) / (shape - 1)
        object.__setattr__(self, 'step', step)

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions, which is always 1."""
        return 1

    def mesh(self, offset=None) -> Tuple[Array, ...]:
        """Generates the array of grid points."""
        return self.domain[0] + jnp.arange(self.shape) * self.step

# --- HEAVILY REWRITTEN CORE CLASS ---
@register_pytree_node_class
@dataclasses.dataclass
class particle:
    """
    Represents the full dynamic state of a single deformable particle.

    WHY THE CHANGE WAS MADE:
    The OLD `particle` class was for a kinematically-prescribed rigid body. It
    stored parameters for motion functions (`displacement_param`, `Rotation_EQ`)
    to CALCULATE the particle's position at any time `t`. It did not store the
    evolving state.
    
    The NEW version is for a DYNAMIC, DEFORMABLE body. Its primary role is to BE
    the state container. Its attributes are the actual state variables (positions,
    velocities) that are updated at each time step according to the physics.

    This class is registered as a JAX PyTree, which is critical for the simulation.
    It tells JAX which attributes are dynamic "children" to be traced and updated
    inside compiled functions.
    """
    # --- DYNAMIC STATE VARIABLES (The core of the new design) ---
    # These arrays represent the full state of the particle at any given moment
    # and are updated by the solver at each time step.
    xp: jax.numpy.ndarray      # Current fluid-interacting marker x-positions (X in paper)
    yp: jax.numpy.ndarray      # Current fluid-interacting marker y-positions (X in paper)
    Ym_x: jax.numpy.ndarray    # Mass-carrying marker x-positions (Y in paper)
    Ym_y: jax.numpy.ndarray    # Mass-carrying marker y-positions (Y in paper)
    Vm_x: jax.numpy.ndarray    # Mass-carrying marker x-velocities
    Vm_y: jax.numpy.ndarray    # Mass-carrying marker y-velocities
    
    # --- PHYSICAL PROPERTIES ---
    # These are physical constants that define the nature of the deformable body.
    mass_per_marker: float
    stiffness: float           # The penalty spring constant, Kp.
    sigma: float               # The surface tension coefficient.
    
    # --- STATIC GEOMETRY and INITIALIZATION INFO ---
    # These are used to create the initial state but do not change during the simulation.
    particle_center: Sequence[Any]
    geometry_param: Sequence[Any]
    Grid: Grid1d
    shape: Callable
    
    # --- REMOVED OBSOLETE KINEMATIC ATTRIBUTES ---
    # The old class had: displacement_param, rotation_param, Displacement_EQ, Rotation_EQ.
    # These have been removed because the motion is now CALCULATED from forces, not prescribed.

    def tree_flatten(self):
      """
      Defines how to flatten this object for JAX PyTree processing.
      
      JAX needs to know which parts are dynamic arrays/tracers ("children")
      and which are static metadata ("aux_data").
      """
      # The children are all the dynamic state arrays and physical properties
      # that might be involved in JAX transformations (like differentiation or jit).
      children = (self.xp, self.yp, self.Ym_x, self.Ym_y, self.Vm_x, self.Vm_y,
                  self.mass_per_marker, self.stiffness, self.sigma,
                  self.particle_center, self.geometry_param)
      # The auxiliary data consists of static elements like the grid generator
      # and the shape function, which do not change and are not traced by JAX.
      aux_data = (self.Grid, self.shape)
      return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
       """Defines how to reconstruct the object from its flattened parts."""
       return cls(*children, *aux_data)

# --- NEW CONTAINER CLASS ---
@register_pytree_node_class
@dataclasses.dataclass
class particle_lista:
    """
    A JAX PyTree container for a sequence of particle objects.

    WHY THIS CLASS IS NEEDED:
    A standard Python list is "opaque" to JAX's `jit` compiler. JAX cannot see
    inside it to trace the operations on the elements. By creating this custom
    container and registering it as a PyTree, we can treat a collection of many
    particles as a single object that JAX can handle efficiently. This is crucial
    for performance and for scaling the simulation to multiple particles.
    """
    particles: Sequence[particle,]
    
    def tree_flatten(self):
      """Flattens the container by treating each particle in the sequence as a child."""
      children = tuple(self.particles)
      aux_data = None  # No static auxiliary data for this simple container.
      return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
       """Reconstructs the container from the sequence of children (particles)."""
       return cls(particles=list(children))

# --- UPDATED CONTAINER CLASS ---
@register_pytree_node_class
@dataclasses.dataclass
class All_Variables: 
    """
    The top-level container for the entire simulation state.

    This class holds all the variables that define the state of the simulation
    at a single point in time. It is a JAX PyTree, allowing the entire state
    to be passed into and out of the main `jax.lax.scan` loop efficiently.

    The primary change from the old version is that the `particles` attribute
    now holds the new `particle_lista` PyTree container.
    """
    particles: particle_lista            # The PyTree container for all particle states.
    velocity: grids.GridVariableVector # The fluid velocity field (u, v).
    pressure: grids.GridVariable       # The fluid pressure field.
    Drag: Sequence[Any]                # For storing simulation outputs like drag force.
    Step_count: int                    # The current simulation step number.
    MD_var: Any                        # For other miscellaneous diagnostic variables.

    def tree_flatten(self):
      """Flattens the entire simulation state into a tuple of children."""
      children = (self.particles, self.velocity, self.pressure, self.Drag, self.Step_count, self.MD_var, )
      aux_data = None
      return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
       """Reconstructs the simulation state from its flattened parts."""
       return cls(*children)
