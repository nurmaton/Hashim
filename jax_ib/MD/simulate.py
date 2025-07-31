# Import necessary libraries and modules
from collections import namedtuple

from typing import Any, Callable, TypeVar, Union, Tuple, Dict, Optional

import functools
# JAX is a library for high-performance machine learning research
import jax
from jax import grad  # For automatic differentiation
from jax import jit   # For just-in-time compilation to speed up functions
from jax import random # For generating random numbers
import jax.numpy as jnp # JAX's implementation of the NumPy API
from jax import lax # For low-level JAX primitives
from jax.tree_util import tree_map, tree_reduce, tree_flatten, tree_unflatten # For working with nested data structures (pytrees)

# Import components from the JAX-MD library
from jax_md import quantity
from jax_md import util
from jax_md import space
from jax_md import dataclasses
from jax_md import partition
from jax_md import smap
from jax_md import simulate

# A utility function to cast variables to a static type
static_cast = util.static_cast


# Define type aliases for clarity and conciseness
Array = util.Array
f32 = util.f32
f64 = util.f64

Box = space.Box

ShiftFn = space.ShiftFn

T = TypeVar('T')
# An initialization function takes a key and positions and returns a state
InitFn = Callable[..., T]
# An application function takes a state and returns an updated state
ApplyFn = Callable[[T], T]
# A simulator is a pair of an initialization function and an application function
Simulator = Tuple[InitFn, ApplyFn]

def brownian_cfd(energy_or_force: Callable[..., Array],
             shift: ShiftFn,
             dt: float,
             kT: float,
             CFD_force,
             gamma: float=0.1) -> Simulator:
  """Simulation of Brownian dynamics with an external CFD force.

  This function simulates Brownian dynamics, which is equivalent to the overdamped
  regime of Langevin dynamics. In this regime, velocity is not explicitly tracked,
  which can make simulations faster. The implementation is based on the work by
  Carlon et al. This version is extended to include an external force from a
  Computational Fluid Dynamics (CFD) simulation.

  Args:
    energy_or_force: A function that returns either the potential energy of the
      system or the force on each particle, given their positions.
    shift: A function that displaces particle positions by a given amount,
      correctly handling boundary conditions.
    dt: The time step for the simulation (a float).
    kT: The thermal energy of the system (temperature * Boltzmann constant).
      This can be updated dynamically during the simulation.
    CFD_force: A function that provides the force on the particles from the
      surrounding fluid, as determined by a CFD simulation.
    gamma: The friction coefficient between the particles and the solvent (a float).

  Returns:
    A tuple containing the initialization function (`init_fn`) and the
    simulation step function (`apply_fn`).
  """

  # Ensure that we have a force function, converting from energy if necessary.
  force_fn = quantity.canonicalize_force(energy_or_force)

  # Cast dt and gamma to a consistent floating point type for stability.
  dt, gamma = static_cast(dt, gamma)

  def init_fn(key, R, mass=f32(1)):
    """Initializes the simulation state."""
    # Create a BrownianState object to hold the simulation state.
    state = simulate.BrownianState(R, mass, key)
    # Ensure the mass is in a canonical format.
    return simulate.canonicalize_mass(state)

  def apply_fn(all_variables, **kwargs):
    """Performs one step of the Brownian dynamics simulation."""
    # Extract the molecular dynamics state from all variables
    state = all_variables.MD_var
    # Use the provided kT, or the default if not given in kwargs.
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    # Unpack the state into positions, mass, and the random key.
    R, mass, key = dataclasses.astuple(state)

    # Split the random key for use in this step, ensuring randomness is maintained.
    key, split = random.split(key)

    # Calculate the total force on the particles.
    # This is the sum of the internal forces and the external CFD force.
    F = force_fn(R, **kwargs) + CFD_force(all_variables)
    # Generate random forces to model thermal fluctuations from the solvent.
    xi = random.normal(split, R.shape, R.dtype)

    # Calculate the mobility (nu), which is the inverse of the friction term.
    nu = f32(1) / (mass * gamma)

    # Update particle positions using the overdamped Langevin equation.
    # The displacement has a deterministic part from the forces and a
    # stochastic part from the random thermal fluctuations.
    dR = F * dt * nu + jnp.sqrt(f32(2) * _kT * dt * nu) * xi
    R = shift(R, dR, **kwargs)

    # Return the new state with updated positions and the new random key.
    return simulate.BrownianState(R, mass, key)

  # Return the initialization and application functions that define the simulator.
  return init_fn, apply_fn
