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
This module provides a simulator for Brownian dynamics coupled with an external
force, such as one derived from a Computational Fluid Dynamics (CFD) simulation.

The core of this module is the `brownian_cfd` function, which sets up a
simulation environment for particles in the overdamped regime (where inertial
effects are negligible). It uses the JAX-MD library for its underlying
physics and simulation framework, enabling high-performance, jit-compilable,
and differentiable simulations.

The main feature is the ability to incorporate a `CFD_force`, allowing for
two-way coupling between a particle simulation and a fluid simulation.
"""

# Standard library imports for type hinting and data structures.
from collections import namedtuple
from typing import Any, Callable, TypeVar, Union, Tuple, Dict, Optional

import functools

# JAX is a library for high-performance numerical computing and ML research.
import jax
# grad is for automatic differentiation of functions.
from jax import grad
# jit is for just-in-time compilation to accelerate Python and NumPy code.
from jax import jit
# random is for generating random numbers in a reproducible way.
from jax import random
# jax.numpy is a JAX-compatible implementation of the NumPy API.
import jax.numpy as jnp
# lax contains low-level JAX primitives.
from jax import lax
# tree_util provides utilities for working with "pytrees" (nested data structures).
from jax.tree_util import tree_map, tree_reduce, tree_flatten, tree_unflatten

# JAX-MD is a library for molecular dynamics simulations in JAX.
# quantity provides tools for physical quantities like energy and force.
from jax_md import quantity
# util contains miscellaneous helper functions.
from jax_md import util
# space defines simulation box properties and displacement functions.
from jax_md import space
# dataclasses provides a JAX-compatible version of dataclasses.
from jax_md import dataclasses
# partition handles neighbor lists for efficient force computation.
from jax_md import partition
# smap provides utilities for mapping functions over particles.
from jax_md import smap
# simulate contains pre-built simulation environments (integrators).
from jax_md import simulate

# static_cast is a utility to ensure variables have a consistent data type.
static_cast = util.static_cast


# Type definitions for improved readability and code clarity.

# An array, typically a JAX numpy array.
Array = util.Array
# 32-bit floating point type.
f32 = util.f32
# 64-bit floating point type.
f64 = util.f64

# Represents the simulation box.
Box = space.Box

# A function that computes displacements, respecting boundary conditions.
ShiftFn = space.ShiftFn

# A generic type variable.
T = TypeVar('T')
# An initialization function that sets up the simulation state.
InitFn = Callable[..., T]
# An application function that advances the simulation state by one step.
ApplyFn = Callable[[T], T]
# A simulator is a pair of an initialization and an application function.
Simulator = Tuple[InitFn, ApplyFn]

def brownian_cfd(energy_or_force: Callable[..., Array],
             shift: ShiftFn,
             dt: float,
             kT: float,
             CFD_force,
             gamma: float=0.1) -> Simulator:
  """Sets up a simulator for Brownian dynamics with a CFD-derived force.

  Simulates Brownian dynamics, which is synonymous with the overdamped
  regime of Langevin dynamics. In this regime, velocities are not explicitly
  tracked, which simplifies the dynamics and can lead to faster simulations.
  The implementation is based on the methods described by Carlon et al. [#carlon]_

  This function extends the standard Brownian dynamics simulation by including
  an external force term, `CFD_force`, which is intended to be supplied by a
  Computational Fluid Dynamics solver.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions. If energy is provided, it will be
      automatically differentiated to get the force.
    shift: A function that displaces positions, `R`, by an amount `dR`,
      correctly applying periodic boundary conditions if necessary.
    dt: A float specifying the simulation time step.
    kT: A float specifying the thermal energy (temperature in units of
      Boltzmann constant). It can be updated dynamically by passing `kT` as a
      keyword argument to the simulation's step function.
    CFD_force: A function that takes the current system state and returns the
      external force from the fluid dynamics simulation.
    gamma: A float specifying the friction coefficient between the particles
      and the solvent.

  Returns:
    A `Simulator`, which is a tuple containing an initialization function
    (`init_fn`) and a simulation step function (`apply_fn`).
  """

  # `canonicalize_force` ensures we have a force function, differentiating
  # the energy function if necessary.
  force_fn = quantity.canonicalize_force(energy_or_force)

  # Cast simulation parameters to a static type to prevent recompilation issues.
  dt, gamma = static_cast(dt, gamma)

  def init_fn(key, R, mass=f32(1)):
    """Initializes the state of the Brownian dynamics simulation."""
    # The state for a Brownian simulation includes particle positions, mass,
    # and a random key for the stochastic component of the motion.
    state = simulate.BrownianState(R, mass, key)
    # `canonicalize_mass` ensures the mass is in a standard format.
    return simulate.canonicalize_mass(state)

  def apply_fn(all_variables, **kwargs):
    """The main simulation step function (integrator)."""
    # The state of the MD simulation is extracted from the input variables.
    state = all_variables.MD_var
    # Allow for dynamic temperature updates during the simulation.
    # If 'kT' is passed as a keyword argument, use it; otherwise, use the
    # default value set when the simulator was created.
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    # Unpack the simulation state into its components.
    R, mass, key = dataclasses.astuple(state)

    # Split the random key to ensure that the random numbers used in this step
    # are unique, which is crucial for reproducibility.
    key, split = random.split(key)

    # Calculate the total force on the particles. This is the sum of the
    # conservative forces from particle interactions and the external CFD force.
    F = force_fn(R, **kwargs) + CFD_force(all_variables)
    # Generate a random force `xi` from a standard normal distribution. This
    # term models the stochastic collisions with solvent molecules.
    xi = random.normal(split, R.shape, R.dtype)

    # Calculate the particle mobility, `nu`, which is inversely proportional
    # to the friction coefficient and mass.
    nu = f32(1) / (mass * gamma)

    # This is the equation of motion for overdamped Langevin dynamics.
    # The displacement `dR` has two parts:
    # 1. A deterministic "drift" term proportional to the total force `F`.
    # 2. A stochastic "diffusion" term proportional to the random force `xi`
    #    and the square root of the temperature.
    dR = F * dt * nu + jnp.sqrt(f32(2) * _kT * dt * nu) * xi
    # Apply the displacement to the particle positions using the `shift`
    # function, which correctly handles boundary conditions.
    R = shift(R, dR, **kwargs)

    # Return the updated state, including the new positions and the new random key.
    return simulate.BrownianState(R, mass, key)

  # The `brownian_cfd` function returns the pair of functions that define the simulator.
  return init_fn, apply_fn
