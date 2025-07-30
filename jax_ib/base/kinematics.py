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
Functions for defining prescribed kinematic motions.

This module contains a collection of functions that describe the pre-determined
trajectory (displacement) and orientation (rotation) of a rigid body as a
function of time. These functions are used in a "kinematically-driven" immersed
boundary simulation, where the motion of the body is an input, and the goal is
to calculate the resulting fluid flow and forces.

This approach contrasts with a "dynamically-driven" simulation (like the penalty
method), where the motion is an output calculated from the fluid forces.
"""

import jax
import jax.numpy as jnp


def displacement(parameters: list, t: float) -> jnp.ndarray:
    """
    Calculates a simple sinusoidal displacement along the x-axis.

    Args:
      parameters: A list containing `[A0, f]`, where `A0` is the amplitude
        and `f` is the frequency of the oscillation.
      t: The current simulation time.

    Returns:
      A 2D JAX array representing the `[x, y]` displacement vector.
    """
    # Unpack the amplitude and frequency from the input parameters.
    A0, f = list(*parameters)
    # The motion is purely in the x-direction.
    return jnp.array([A0 / 2 * jnp.cos(2 * jnp.pi * f * t), 0.0])


def rotation(parameters: list, t: float) -> float:
    """
    Calculates a simple sinusoidal rotation angle.

    Args:
      parameters: A list containing `[alpha0, beta, f, phi]`, where `alpha0` is
        the mean angle, `beta` is the amplitude of oscillation, `f` is the
        frequency, and `phi` is the phase offset.
      t: The current simulation time.

    Returns:
      A float representing the rotation angle in radians.
    """
    # Unpack the rotational parameters.
    alpha0, beta, f, phi = list(*parameters)
    # The angle oscillates around a mean value `alpha0`.
    return alpha0 + beta * jnp.sin(2 * jnp.pi * f * t + phi)


def Displacement_Foil_Fourier_Dotted_Mutliple(parameters: list, t: float) -> jnp.ndarray:
    """
    Calculates a complex foil displacement based on a Fourier series.
    This function is designed to work with a batch of multiple particles.

    The displacement is a combination of a constant-velocity forward motion and
    a vertical "heaving" motion described by a Fourier series.

    Args:
      parameters: A list of parameter sets, one for each particle. Each set is
        a tuple: `(alpha0, f, phi, alpha, beta, p)`.
        - `alpha0`: Forward velocity component.
        - `f`: Fundamental frequency.
        - `phi`: Phase offset.
        - `alpha`: Amplitudes for the sine (heaving) terms of the Fourier series.
        - `beta`: Amplitudes for the cosine (pitching) terms of the Fourier series.
        - `p`: A coupling factor between pitch and heave.
      t: The current simulation time.

    Returns:
      A JAX array of shape `(2, N_particles)` containing the [x, y]
      displacement for each of the N particles.
    """
    # --- Unpack parameters for all particles ---
    # This complex unpacking reshapes the list of tuples into separate arrays
    # for each parameter, where each row corresponds to a particle.
    alpha0 = jnp.array(list(list(zip(*parameters))[0]))
    f = jnp.array(list(list(zip(*parameters))[1]))
    phi = jnp.array(list(list(zip(*parameters))[2]))
    alpha = jnp.array(list(list(zip(*parameters))[3])) # Shape: (N_particles, N_fourier_terms)
    beta = jnp.array(list(list(zip(*parameters))[4]))
    p = jnp.array(list(list(zip(*parameters))[5]))
    
    N_fourier_terms = alpha.shape[1]
    N_particles = len(alpha)

    # Create an array of harmonic numbers [1, 2, 3, ...] for the Fourier series.
    frequencies = jnp.arange(1, N_fourier_terms + 1)

    # --- Calculate the argument of the sin/cos functions ---
    # This computes `(2*pi*n*f*t + phi)` for each particle and each Fourier term `n`.
    # Reshaping allows for broadcasting across all terms.
    inside_function = (2 * jnp.pi * t * frequencies * f.reshape(N_particles, 1) +
                       phi.reshape(N_particles, 1))
  
    # --- Sum the Fourier series to get the vertical (heaving) motion ---
    # `alpha_1 = Σ [alpha_n * sin(arg_n) + p * beta_n * cos(arg_n)]`
    heave_motion = (alpha * jnp.sin(inside_function)).sum(axis=1)
    heave_motion += p * (beta * jnp.cos(inside_function)).sum(axis=1)
    
    # The x-displacement is a simple constant velocity motion.
    forward_motion = -alpha0 * t
    
    # Combine forward and heaving motion into the final displacement vector.
    return jnp.array([forward_motion, heave_motion])


def rotation_Foil_Fourier_Dotted_Mutliple(parameters: list, t: float) -> jnp.ndarray:
    """
    Calculates a complex foil rotation (pitching) based on a Fourier series.
    This function is designed to work with a batch of multiple particles.

    The structure is very similar to the displacement function, but it computes
    the rotational angle instead of the y-position.

    Args:
      parameters: A list of parameter sets, one for each particle. Each set is
        a tuple: `(alpha0, f, phi, alpha, beta, p)`.
        - `alpha0`: Mean angular velocity component.
        - `f`, `phi`, `alpha`, `beta`, `p`: Fourier series parameters for pitching motion.
      t: The current simulation time.

    Returns:
      A 1D JAX array of shape `(N_particles,)` containing the rotation angle for
      each particle.
    """
    # --- Unpack parameters (identical to the displacement function) ---
    alpha0 = jnp.array(list(list(zip(*parameters))[0]))
    f = jnp.array(list(list(zip(*parameters))[1]))
    phi = jnp.array(list(list(zip(*parameters))[2]))
    alpha = jnp.array(list(list(zip(*parameters))[3]))
    beta = jnp.array(list(list(zip(*parameters))[4]))
    p = jnp.array(list(list(zip(*parameters))[5]))
    
    N_fourier_terms = alpha.shape[1]
    N_particles = len(alpha)
    frequencies = jnp.arange(1, N_fourier_terms + 1)

    # --- Calculate the argument of the sin/cos functions (identical) ---
    inside_function = (2 * jnp.pi * t * frequencies * f.reshape(N_particles, 1) +
                       phi.reshape(N_particles, 1))
  
    # --- Sum the Fourier series to get the oscillatory part of the pitching motion ---
    pitch_oscillation = (alpha * jnp.sin(inside_function)).sum(axis=1)
    pitch_oscillation += p * (beta * jnp.cos(inside_function)).sum(axis=1)
    
    # The final angle is the sum of a steady rotation and the oscillation.
    return alpha0 * t + pitch_oscillation
