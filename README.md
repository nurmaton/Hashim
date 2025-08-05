JAX-IB: A Differentiable Immersed Boundary Method for Deformable Bodies

This repository contains a [JAX](https://github.com/google/jax)-based implementation of the Immersed Boundary (IB) method for simulating Fluid-Structure Interaction (FSI), with a focus on **dynamic, deformable bodies**.

The entire simulation is constructed as a JAX PyTree, making the solver end-to-end differentiable. This allows for the use of gradient-based optimization to solve inverse problems, such as discovering optimal material properties or control strategies for an immersed object.

The primary example in this repository is the `Deformable_Droplet_Demo.ipynb`. This notebook first runs a forward simulation of a deformable droplet settling under gravity. It then demonstrates how to solve an inverse problem by using differentiation to automatically discover the physical stiffness required to make the droplet's final state match a predefined target shape.

<!-- Recommendation: Create a GIF of your final optimization plot and place it here! -->
<!-- Example: ![Optimization Demo GIF](notebooks/assets/optimization.gif) -->

---

## Background and Acknowledgment

This project, developed in collaboration with [khusrave](https://github.com/khusrave), is an adaptation and extension of the original `jax_ib` framework developed by Mohammed Alhashim, available at:
*   **Original Repository:** [https://github.com/hashimmg/jax_ib](https://github.com/hashimmg/jax_ib)

The original project was a groundbreaking differentiable solver for **rigid bodies**, as described in the PNAS paper:
> Alhashim, M. G., Hausknecht, K., & Brenner, M. P. (2025). Control of flow behavior in complex fluids using automatic differentiation. *PNAS, 122*(8).

This repository builds upon that foundation by introducing the physics for **deformable bodies**, drawing inspiration from the penalty-based IBM for fluid droplets described in:
> Sustiel, J. B., & Grier, D. G. (2022). Complex dynamics of an acoustically levitated fluid droplet captured by a low-order immersed boundary method.

---

## New in this Version: Solving the Inverse Problem

A key feature of this project is the demonstration of solving an **inverse design problem**. While a standard *forward simulation* answers the question:

> "Given a stiffness of 50,000, what will the droplet's final shape be?"

...our differentiable approach allows us to answer the more powerful *inverse* question:

> "Given a desired final shape, what stiffness was required to produce it?"

The `Deformable_Droplet_Demo.ipynb` notebook demonstrates this by:
1.  Running a forward simulation to generate a "target shape".
2.  Starting with an incorrect guess for the droplet's stiffness.
3.  Using `jax.grad()` to get the gradient of the entire simulation's outcome with respect to the stiffness parameter.
4.  Applying gradient descent to automatically and iteratively update the stiffness, "walking" it towards the correct value that minimizes the difference between the simulated shape and the target shape.

This powerful technique turns the simulation into a goal-seeking optimization tool.

---

## Key Modifications for Deformable Body Physics

To transition the framework from pre-determined rigid body motion to dynamic deformable body physics, several core modules in the `jax_ib/base/` directory were significantly rewritten. The key changes include:

*   `particle_class.py`: The `particle` class was fundamentally changed from a kinematic descriptor to a **stateful container**. It no longer holds parameters for motion functions but instead holds the dynamic state variables (marker positions, velocities, etc.) that are evolved by the physics simulation.

*   `particle_motion.py`: The logic was changed from evaluating prescribed motion functions to implementing the **dynamic equations of motion**. It now updates the particle's state by advecting boundary markers with the fluid velocity (Eq. 3 from Sustiel & Grier) and accelerating mass-carrying markers with Newtonian physics (Eq. 5).

*   `IBM_Force.py`: The force calculation was changed from a kinematic correction term to computing **real physical forces**. It now calculates the internal penalty spring force (Eq. 4) and surface tension (Eq. 7) that the deformable body exerts on the fluid.

*   `boundaries.py`: The functions for time-dependent boundary conditions (`update_BC`, `Reserve_BC`) were simplified to pass-through functions, as the deformable body simulation uses static far-field boundaries. A critical bug related to JAX PyTree stability in `get_pressure_bc_from_velocity` was also fixed to ensure compatibility with `jax.lax.scan`.

---

## Getting Started

### Prerequisites

You will need a working Python 3.8+ environment with `pip`.

All specific Python package dependencies are listed in the `pyproject.toml` file and will be installed automatically in the next step.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/nurmaton/M_Alhashim.git
    cd M_Alhashim
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install the package. For development, it is highly recommended to perform an **editable install** using the `-e` flag. This links the installed package to your source files, so any changes you make to the code are immediately effective without needing to reinstall.

    ```bash
    # Install the base package in editable mode
    pip install -e .
    ```

4.  **(Optional)** To run the visualization code in the notebooks, install the optional `[data]` dependencies:
    ```bash
    pip install -e ".[data]"
    ```

*(**Important Note on JAX:** For GPU/TPU support, you may need a specific version of `jaxlib`. Please follow the official [JAX installation instructions](https://github.com/google/jax#installation) *after* the steps above to ensure you have the correct version for your hardware and CUDA setup.)*

### Running the Demo

The main example is a Jupyter Notebook. After installing the package, you can run the demo.

1.  Start a Jupyter server (e.g., JupyterLab):
    ```bash
    jupyter lab
    ```

2.  Navigate to the `notebooks/` directory and open the `Deformable_Droplet_Demo.ipynb` file.

3.  You can run the cells in the notebook to set up, run, visualize, and optimize the deformable droplet simulation. Because you installed the package, the `import jax_ib` statement will work correctly.

---

## Core Concepts of the Deformable Model

The physics of the deformable body is based on the **penalty Immersed Boundary Method**, which uses two sets of Lagrangian markers:

1.  **Mass-carrying markers (`Y`)**: These hold the particle's inertia and are evolved by a Molecular Dynamics-style integrator. Their motion is governed by Newton's Second Law, `M * d²Y/dt² = F_penalty + F_gravity` (Eq. 5).
2.  **Fluid-interacting markers (`X`)**: These are massless points that define the object's boundary. They are advected by the local fluid velocity, calculated via an integral `U(X) = ∫ u(x) δ(x - X) dx` (Eq. 3).

The two sets of markers are connected by stiff springs that generate a **penalty force**, `F = Kp(Y - X)` (Eq. 4), which models the body's elasticity. The model also includes a **surface tension force**, `F = -σ * d(l_hat)/ds` (Eq. 7), which acts to minimize the boundary length.

---

## Citation

If you use this code in your research, please consider citing the original works that this project is based on:

```bibtex
@article{alhashim2025control,
  title={Control of flow behavior in complex fluids using automatic differentiation},
  author={Alhashim, Mohammed G and Hausknecht, Kaylie and Brenner, Michael P},
  journal={Proceedings of the National Academy of Sciences},
  volume={122},
  number={8},
  pages={e2403644122},
  year={2025},
  publisher={National Acad Sciences}
}

@article{sustiel2022complex,
  title={Complex dynamics of an acoustically levitated fluid droplet captured by a low-order immersed boundary method},
  author={Sustiel, Jacqueline B and Grier, David G},
  journal={Physical Review Fluids},
  volume={7},
  number={8},
  pages={084001},
  year={2022},
  publisher={APS}
}
