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
This `__init__.py` file makes the `jax_ib.base` directory a Python package.

By importing the key modules here, it allows users to access the core classes
and functions of the library with a simpler import statement, such as:
`from jax_ib.base import grids`

The modules are grouped by their functionality to provide a clear overview of the
library's architecture.
"""

# --- Core Simulation and Immersed Boundary Method (IBM) Components ---
# These modules represent the highest level of abstraction for the fluid-structure
# interaction simulation.

# Defines the data structures for particles and the overall simulation state (`All_Variables`).
import jax_ib.base.particle_class

# Implements the equations of motion for the particles (solid-side solver).
import jax_ib.base.particle_motion

# Calculates the interaction forces between the fluid and the immersed boundary.
import jax_ib.base.IBM_Force

# Contains the main time-stepping loop (`forward_step`) that advances the simulation.
import jax_ib.base.time_stepping


# --- Physics and Numerical Equation Solvers ---
# These modules implement the numerical solutions for the various terms in the
# governing Navier-Stokes equations.

# Defines the main governing equations of the fluid simulation.
import jax_ib.base.equations

# Implements the pressure projection step to enforce fluid incompressibility.
import jax_ib.base.pressure

# Provides numerical schemes for the advection (convection) term.
import jax_ib.base.advection

# Provides numerical schemes for the diffusion (viscosity) term.
import jax_ib.base.diffusion


# --- Foundational Utilities and Numerical Methods ---
# These modules provide the fundamental building blocks and tools used by the
# higher-level physics and IBM components.

# Defines the core data structures for the Eulerian grid (`Grid`, `GridArray`, `GridVariable`).
import jax_ib.base.grids

# Implements boundary conditions (Periodic, Dirichlet, Neumann) for the fluid domain.
import jax_ib.base.boundaries

# Provides the discrete delta function kernels essential for the IBM.
import jax_ib.base.convolution_functions

# Provides general-purpose interpolation functions.
import jax_ib.base.interpolation

# Implements basic finite difference operators (e.g., grad, div, laplacian).
import jax_ib.base.finite_differences

# Contains helper functions for JAX array manipulation.
import jax_ib.base.array_utils


# --- Legacy/Kinematic Components ---
# These modules may contain functionality from the older, kinematically-driven
# version of the code.

# Likely contains functions related to prescribed (non-dynamic) motion.
import jax_ib.base.kinematics
