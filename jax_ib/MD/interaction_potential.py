# Import JAX's version of NumPy for high-performance, differentiable computations.
import jax.numpy as jnp
# Import the standard NumPy library, often used for data type definitions.
import numpy as np
# Import specific modules from the JAX-MD library for molecular dynamics.
from jax_md import space, smap, energy, minimize, quantity, simulate, partition

def harmonic_morse(dr, h=0.5, D0=5.0, alpha=5.0, r0=1.0, k=300.0, **kwargs):
    """
    Defines a custom potential energy function that combines a harmonic
    potential for repulsion with a Morse potential for attraction.

    This creates a potential that is strongly repulsive at short distances
    and behaves like a realistic chemical bond at longer distances.

    Args:
      dr: The distance between two particles.
      h: A scaling factor for the harmonic potential. Defaults to 0.5.
      D0: The dissociation energy, representing the depth of the potential well.
          Defaults to 5.0.
      alpha: A parameter that controls the "width" of the potential well.
             Defaults to 5.0.
      r0: The equilibrium distance between particles (the bottom of the well).
          Defaults to 1.0.
      k: The spring constant for the harmonic part of the potential, controlling
         its stiffness. Defaults to 300.0.
      **kwargs: Catches any extra arguments that might be passed.

    Returns:
      The potential energy U for the given distance dr.
    """
    # jnp.where is a conditional function. It chooses between two values based on a condition.
    # Condition: dr < r0
    # If True (distance is less than equilibrium): Use a harmonic potential for strong repulsion.
    # The term '- D0' sets the minimum of the potential at the equilibrium distance to -D0.
    harmonic_potential = h * k * (dr - r0)**2 - D0
    
    # If False (distance is greater than or equal to equilibrium): Use a Morse potential.
    # This is a standard model for diatomic interactions.
    morse_potential = D0 * (jnp.exp(-2. * alpha * (dr - r0)) - 2. * jnp.exp(-alpha * (dr - r0)))
    
    # Apply the condition to choose between the two potential types.
    U = jnp.where(dr < r0, 
                  harmonic_potential,
                  morse_potential
                 )
    # Ensure the output has the same data type as the input distance for consistency.
    return jnp.array(U, dtype=dr.dtype)

# Define standard floating-point data types for convenience.
f32 = np.float32
f64 = np.float64

def harmonic_morse_pair(displacement_or_metric, species=None, h=0.5, D0=5.0, alpha=10.0, r0=1.0, k=50.0): 
    """
    Creates a function that computes the total pairwise potential energy of a
    system of particles using the custom harmonic_morse potential.

    This function uses jax_md.smap.pair to efficiently apply the harmonic_morse
    potential to all pairs of particles in a simulation.

    Args:
      displacement_or_metric: A JAX-MD function that calculates the displacement
                              and distance between all pairs of particles in the system.
      species: An optional array to specify different types of particles, which can
               have different interaction parameters. Defaults to None (all particles
               are the same type).
      h, D0, alpha, r0, k: Parameters for the harmonic_morse potential. These values
                           will be used for all particle pairs unless different species
                           are defined.

    Returns:
      A function that takes particle positions (R) and computes the total
      pairwise energy of the system.
    """
    # Convert the potential parameters to JAX arrays with a specific data type (float32).
    # This is good practice for JAX to optimize computations.
    h = jnp.array(h, dtype=f32)
    D0 = jnp.array(D0, dtype=f32)
    alpha = jnp.array(alpha, dtype=f32)
    r0 = jnp.array(r0, dtype=f32)
    k = jnp.array(k, dtype=f32)

    # `smap.pair` is a powerful JAX-MD function that takes a pairwise potential
    # (like harmonic_morse) and creates a new function that sums this potential
    # over all pairs in a system.
    return smap.pair(
        # The pairwise potential function to apply.
        harmonic_morse,
        # A standardized displacement function. This handles the geometry of the
        # simulation space (e.g., periodic boundary conditions).
        space.canonicalize_displacement_or_metric(displacement_or_metric),
        # Pass the parameters to the underlying harmonic_morse function.
        # JAX-MD will handle broadcasting these parameters for different species if needed.
        species=species,
        h=h,
        D0=D0,
        alpha=alpha,
        r0=r0,
        k=k
    )
