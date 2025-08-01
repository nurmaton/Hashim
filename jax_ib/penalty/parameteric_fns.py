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
This module provides a collection of differentiable, parametric functions for
defining 2D shapes in polar coordinates.

These functions are fundamental to the project's shape optimization capabilities.
They are used to generate the target positions of the Lagrangian boundary markers
that define the immersed object's geometry.

Each function takes a `geometry_param` argument, which is a JAX array
containing the shape's defining parameters (e.g., radii, number of lobes).
Because these parameters are JAX arrays, the entire shape definition is
differentiable. This allows the main solver to compute gradients of a loss
function (e.g., drag) with respect to these geometric parameters, enabling
the automatic optimization of the object's shape for a desired outcome.
"""

# JAX's implementation of the NumPy API for high-performance, differentiable arrays.
import jax.numpy as jnp
# Standard NumPy, used here for defining floating-point type aliases.
import numpy as np


def param_ellipse(geometry_param, theta):
    """Defines an ellipse in polar coordinates, centered at the origin.

    Args:
        geometry_param: A JAX array `[A, B]`, where `A` is the semi-major
                        axis and `B` is the semi-minor axis.
        theta: A JAX array of angles (in radians) at which to calculate the radius.

    Returns:
        A JAX array of radial distances `r` for each angle in `theta`.
    """
    A = geometry_param[0]
    B = geometry_param[1]
    # Standard polar equation for an ellipse.
    return A * B / jnp.sqrt((B * jnp.cos(theta))**2 + (A * jnp.sin(theta))**2)


def param_rose(geometry_param, theta):
    """Defines a rose curve (rhodonea curve) in polar coordinates.

    This can be used to create objects with multiple lobes or petals.

    Args:
        geometry_param: A JAX array `[A, B]`, where `A` is the maximum radius
                        (petal length) and `B` determines the number of petals.
                        If `B` is an integer, it creates `B` or `2*B` petals.
        theta: A JAX array of angles (in radians) at which to calculate the radius.

    Returns:
        A JAX array of radial distances `r` for each angle in `theta`.
    """
    A = geometry_param[0]
    B = geometry_param[1]
    return A * jnp.sin(B * theta)


def param_snail(geometry_param, theta):
    """Defines an Archimedean spiral (a "snail" shape) in polar coordinates.
    
    This function has been misnamed as `param_snail`; it actually describes
    an Archimedean spiral, not a limaçon (which is sometimes called a snail).

    Args:
        geometry_param: A JAX array `[A]`, where `A` controls the distance
                        between successive turns of the spiral.
        theta: A JAX array of angles (in radians) at which to calculate the radius.

    Returns:
        A JAX array of radial distances `r` for each angle in `theta`.
    """
    A = geometry_param[0]
    # This parameter is unused in the current implementation.
    # B = geometry_param[1]
    return A * theta


def param_circle(geometry_param, theta):
    """Defines a circle centered at the origin.

    Args:
        geometry_param: A JAX array `[A]`, where `A` is the radius of the circle.
        theta: A JAX array of angles. The shape of this array determines the
               shape of the output array.

    Returns:
        A JAX array with the same shape as `theta`, where every element is the radius `A`.
    """
    A = geometry_param[0]
    return A * jnp.ones_like(theta)


def param_rot_ellipse(phi, geometry_param, theta):
    """Defines an ellipse that is rotated by an angle `phi`.

    This function is critical for simulating flapping or pitching motions,
    where the orientation of the object changes over time. The rotation angle
    `phi` can itself be a differentiable function of time.

    Args:
        phi: The angle of rotation for the ellipse (a float or JAX array).
        geometry_param: A JAX array `[A, B]`, where `A` is the semi-major
                        axis and `B` is the semi-minor axis.
        theta: A JAX array of angles (in radians) at which to calculate the radius.

    Returns:
        A JAX array of radial distances `r` for the rotated ellipse.
    """
    A = geometry_param[0]
    B = geometry_param[1]
    # Calculate the eccentricity of the ellipse.
    # jnp.round is used for numerical stability in case B is very close to A.
    excc = jnp.sqrt(1 - jnp.round((B / A)**2, 6))
    # This is the polar equation for an ellipse rotated by angle phi.
    return B / jnp.sqrt(1 - (excc * jnp.cos(theta - phi))**2)
