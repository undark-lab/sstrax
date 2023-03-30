import jax
import jax.numpy as jnp
from .ode import dynamics_solver, mass_solver
from .tidal import get_stars_X0, StrippingSampler
from .constants import Parameters
import numpy as np


@jax.jit
def init_stripping(params: Parameters, mass_sol):
    """
    Generates stochastic sampler for stripping times given stream parametes and mass evolution history
    Args:
      params: Parameters class containing stream model parameters
      mass_sol: mass evolution solution, output of mass_solver
    Returns:
      Sampler for the stripping times
    Examples
    --------
    >>> params = Parameters()
    >>> clust_sol = dynamics_solver(params.cluster_final, params.age, 0.0, dense=True)
    >>> mass_sol = mass_solver(params, clust_sol)
    >>> strip_sampler = init_stripping(Parameters(), mass_sol)
    >>> t_strip = strip_sampler.sample(jax.random.PRNGKey(0))
    """
    mass_fun_vmap = jax.vmap(mass_sol.evaluate, (0,))
    t_sample = jnp.linspace(0.0, params.age, 16**3)
    mass_sample = mass_fun_vmap(t_sample)
    strip_sampler = StrippingSampler.from_mass(t_sample, mass_sample)
    return strip_sampler


@jax.jit
def sample_trace(
    key, strip_sampler, clust_sol, mass_sol, params: Parameters
) -> jnp.ndarray:
    """
    Samples the stripping times and initial conditions for stream stars and evolves them in potential
    Args:
      key: jax.random.PRNGKey random key
      strip_sampler: sampler for the stripping times, output of init_stripping
      clust_sol: cluster evolution solution, output of dynamics_solver
      mass_sol: mass evolution solution, output of mass_solver
      params: Parameters class containing stream model parameters
    Returns:
      Final 6d phase space position for a stripped star (x, y, z [kpc], v_x, v_y, v_z [kpc/Myr])
    Examples
    --------
    >>> params = Parameters()
    >>> clust_sol = dynamics_solver(params.cluster_final, params.age, 0.0, dense=True)
    >>> mass_sol = mass_solver(params, clust_sol)
    >>> strip_sampler = init_stripping(Parameters(), mass_sol)
    >>> Xf = sample_trace(jax.random.PRNGKey(0), strip_sampler, clust_sol, mass_sol, params)
    """
    k1, k2 = jax.random.split(key)
    t_strip = strip_sampler.sample(k1)
    X0 = get_stars_X0(k2, t_strip, clust_sol, mass_sol, params)
    return dynamics_solver(X0, t_strip, params.age)


def simulate_stream(key, params: Parameters) -> jnp.ndarray:
    """
    Simulates the full evoluition of the stream given a random seed and a set of parameters
    Args:
      key: jax.random.PRNGKey random key
      params: Parameters class containing stream model parameters
    Returns:
      Final 6d phase space position for the set of stripped stars.
      Rows each correspond to an individual stars 6d phase space co-ordiantes: (x, y, z [kpc], v_x, v_y, v_z [kpc/Myr]).
      Columrns
    Examples
    --------
    >>> params = Parameters()
    >>> stream = simulate_stream(jax.random.PRNGKey(0), params)
    """
    # Reverse solve for cluster center
    clust_sol = dynamics_solver(params.cluster_final, params.age, 0.0, dense=True)
    # Solve for the mass evolution given the cluster trajectory
    mass_sol = mass_solver(params, clust_sol)

    # Compute the stripping times of the stars
    nstars = int(np.floor((params.msat - mass_sol.evaluate(params.age)) / params.mbar))
    strip_sampler = init_stripping(params, mass_sol)

    keys = jax.random.split(key, nstars)
    dtype = clust_sol.t0.dtype  # use same dtype as used in integrator
    stars_Xf = np.empty((nstars, 6), dtype=dtype)
    for i, key in enumerate(keys):
        # Evolve nstars stripped stars to form the final stream
        stars_Xf[i] = sample_trace(key, strip_sampler, clust_sol, mass_sol, params)

    return stars_Xf
