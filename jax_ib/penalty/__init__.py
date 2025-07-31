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
This `__init__.py` file designates the `jax_ib.penalty` directory as a Python subpackage.

This subpackage is dedicated to implementing penalty methods. In the context of
the Immersed Boundary (IB) method, penalty methods are a common approach to
enforce constraints, such as the no-slip boundary condition at the interface
between a fluid and an immersed object. This is typically done by applying a
force (the "penalty") that is proportional to the violation of the constraint
(e.g., the slip velocity).

By importing the key modules here, it signals their importance and allows users
to access them through the `jax_ib.penalty` namespace.
"""

# This module likely contains various utility and helper functions that support
# the main penalty calculations. This could include mathematical operations,
# geometric transformations, or other common tasks needed across the subpackage.
import jax_ib.penalty.util_funs

# This module likely defines a collection of parametric functions. These are
# flexible mathematical functions (e.g., smoothed step functions,
# Gaussian-like curves) whose behavior is controlled by a set of parameters.
# They can be used to define the shape and strength of the penalty potential
# or force, providing a versatile way to model the interaction at the
# fluid-solid interface.
import jax_ib.penalty.parameteric_fns
