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
This `__init__.py` file makes the `jax_ib.MD` directory a Python subpackage.

This subpackage appears to contain modules related to Molecular Dynamics (MD)
simulations or a coupling between MD and the main CFD (Computational Fluid
Dynamics) solver.

By importing the key modules here, it allows users to access the core classes
and functions with a simpler import statement, such as:
`from jax_ib.MD import simulate`
"""

# This module likely contains the logic for coupling the MD simulation
# with the fluid dynamics simulation. This could involve passing forces or
# velocities between the two solvers.
import jax_ib.MD.CFD_MD_coupling

# This module likely defines the potential energy functions that govern the
# interactions between particles in the MD simulation (e.g., Lennard-Jones,
# harmonic bonds, electrostatic forces).
import jax_ib.MD.interaction_potential

# This module likely contains the main simulation loop or time integrator
# for the MD part of the system (e.g., a Velocity Verlet integrator).
import jax_ib.MD.simulate
