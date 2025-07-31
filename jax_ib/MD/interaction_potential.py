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
This module defines custom interaction potentials for molecular dynamics simulations.

The primary function provided is `harmonic_morse_pair`, which creates a pairwise
interaction potential that combines a harmonic potential for repulsion at short
distances and a Morse potential to model attraction and bonding at longer
distances. This type of hybrid potential is useful for modeling covalent bonds
or other interactions where a soft repulsion and a distinct energy well are
required.

The module uses the `jax_md.smap` utility to lift the pairwise potential function
to a function that computes the total energy of a system of particles.
"""

# JAX's implementation of the NumPy API for high-performance numerical computing.
import jax.numpy as jnp
# Standard NumPy, often used for initial array creation on the CPU.
import numpy as np
# Import necessary components from the JAX-MD library.
from jax_md import space, smap, energy, minimize, quantity, simulate, partition


def harmonic_morse(dr, h=0.5, D0=5.0, alpha=5.0, r0=1.0, k=300.0, **kwargs):
    """Computes a hybrid harmonic-Morse potential for a given distance.

    This potential uses a harmonic function for distances less than the
    equilibrium distance `r0` (modeling a stiff repulsion) and a Morse
    potential for distances greater than `r0` (modeling attraction and
    bond dissociation).

    Args:
        dr: The distance between two particles (a float or JAX array).
        h: A scaling factor for the harmonic potential.
        D0: The depth of the potential well (dissociation energy).
        alpha: A parameter controlling the width of the potential well.
        r0: The equilibrium bond distance.
        k: The spring constant for the harmonic part of the potential.
        **kwargs: Allows for extra arguments to be passed, which are ignored.

    Returns:
        The potential energy `U` for the given distance `dr`.
    """
    # Use `jnp.where` for a conditional evaluation that is JAX-compatible.
    U = jnp.where(
        dr < r0,
        # For distances less than r0, use a harmonic potential: h*k*(dr-r0)^2.
        # The -D0 term sets the minimum of the potential at the Morse well depth.
        h * k * (dr - r0)**2 - D0,
        # For distances greater than or equal to r0, use the Morse potential.
        D0 * (jnp.exp(-2. * alpha * (dr - r0)) - 2. * jnp.exp(-alpha * (dr - r0)))
    )
    return jnp.array(U, dtype=dr.dtype)

# Define floating point type aliases for convenience.
f32 = np.float32
f64 = np.float64

def harmonic_morse_pair(displacement_or_metric, species=None, h=0.5, D0=5.0, alpha=10.0, r0=1.0, k=50.0):
    """Creates a total energy function for the harmonic-Morse potential.

    This function is a factory that takes potential parameters and returns a
    function that can compute the total pairwise potential energy of a system
    of particles. It uses `jax_md.smap.pair` to apply the `harmonic_morse`
    potential to all pairs of particles in the system.

    Args:
        displacement_or_metric: A function from `jax_md.space` that computes
            the displacement or distance between two points, respecting
            boundary conditions.
        species: An optional array of species identifiers for each particle.
            This allows for different interaction parameters between different
            types of particles.
        h: Scaling factor for the harmonic potential.
        D0: The depth of the potential well.
        alpha: A parameter controlling the width of the potential well.
        r0: The equilibrium bond distance.
        k: The spring constant for the harmonic part of the potential.

    Returns:
        A function that takes particle positions (`R`) and computes the total
        potential energy of the system.
    """
    # Convert parameters to JAX arrays of a specific type. This helps prevent
    # JAX from recompiling the function if the parameter types change.
    h = jnp.array(h, dtype=f32)
    D0 = jnp.array(D0, dtype=f32)
    alpha = jnp.array(alpha, dtype=f32)
    r0 = jnp.array(r0, dtype=f32)
    k = jnp.array(k, dtype=f32)

    # `smap.pair` creates a function that sums a pairwise potential over all
    # pairs of particles in a system.
    # `space.canonicalize_displacement_or_metric` ensures the displacement
    # function is in a standard format that `smap.pair` can use.
    return smap.pair(
        harmonic_morse,
        space.canonicalize_displacement_or_metric(displacement_or_metric),
        species=species,
        h=h,
        D0=D0,
        alpha=alpha,
        r0=r0,
        k=k
    )
