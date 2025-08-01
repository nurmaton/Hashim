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
This `__init__.py` file makes `jax_ib` a Python package.

`jax_ib` is an end-to-end differentiable implementation of the Immersed
Boundary (IB) method, written in JAX. It is designed to solve complex
fluid-structure interaction problems, with a focus on supporting deformable
or elastic bodies.

The core innovation of this library is its differentiability. By leveraging
JAX, the entire simulation pipeline—from the fluid dynamics solver to the
motion of the immersed object—is a differentiable function. This allows for
the use of gradient-based optimization to solve inverse problems, such as:
- Optimizing an object's shape to minimize drag.
- Discovering the most efficient flapping motion for a flexible swimmer.
- Inferring material properties from observed dynamics.

The package is organized into several key subpackages that represent the
different physical components of the IB method.
"""

# The `base` subpackage likely contains the fundamental data structures and
# core components shared across the library, such as the definitions for
# grids and velocity fields (`GridArray`) that form the foundation of the
# Eulerian fluid solver.
import jax_ib.base

# The `MD` (Molecular Dynamics) subpackage is responsible for managing the
# state of the immersed object(s), which are represented as a collection of
# Lagrangian markers. This includes:
#  - `interaction_potential`: Defines the internal elastic forces that hold
#    the deformable object together.
#  - `simulate`: Implements the time-stepping logic (integrator) for the
#    Lagrangian markers, updating their positions based on both internal
#    and external (fluid) forces.
import jax_ib.MD

# The `penalty` subpackage implements the crucial coupling mechanism between
# the fluid (Eulerian) and the solid (Lagrangian) domains. It uses a
# Brinkman-style penalty method to generate a force field on the fluid grid
# that represents the solid object, thereby enforcing the no-slip boundary
# condition. This is where the geometry of the object is rasterized onto the
# grid.
import jax_ib.penalty
