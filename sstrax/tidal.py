import chex
import jax
import jax.numpy as jnp
from .backgrounds import jacobian_force_mw
from .constants import G, Parameters


@jax.jit
def d2phidr2_mw(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the second derivative of the Milky Way potential at a position x (in the simulation frames)
    Args:
      x: 3d position (x, y, z) in [kpc]
    Returns:
      Second derivative of force (per unit mass) in [1/Myr^2]
    Examples
    --------
    >>> d2phidr2_mw(x=jnp.array([8.0, 0.0, 0.0]))
    """
    rad = jnp.linalg.norm(x)
    return -jnp.matmul(jnp.transpose(x), jnp.matmul(jacobian_force_mw(x), x)) / rad**2


@jax.jit
def omega(x: jnp.ndarray, v: jnp.ndarray) -> float:
    """
    Computes the magnitude of the angular momentum in the simulation frame
    Args:
      x: 3d position (x, y, z) in [kpc]
      v: 3d velocity (v_x, v_y, v_z) in [kpc/Myr]
    Returns:
      Magnitude of angular momentum in [rad/Myr]
    Examples
    --------
    >>> omega(x=jnp.array([8.0, 0.0, 0.0]), v=jnp.array([8.0, 0.0, 0.0]))
    """
    rad = jnp.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    omega_vec = jnp.cross(x, v) / rad**2
    return jnp.linalg.norm(omega_vec)


@jax.jit
def tidalr_mw(x: jnp.ndarray, v: jnp.ndarray, Msat: float) -> float:
    """
    Computes the tidal radius of a clsuter in the Milky Way potential
    Args:
      x: 3d position (x, y, z) in [kpc]
      v: 3d velocity (v_x, v_y, v_z) in [kpc/Myr]
      Msat: Cluster mass in [Msol]
    Returns:
      Tidal radius of the cluster in [kpc]
    Examples
    --------
    >>> tidalr_mw(x=jnp.array([8.0, 0.0, 0.0]), v=jnp.array([8.0, 0.0, 0.0]), Msat=1e4)
    """
    return (G * Msat / (omega(x, v) ** 2 - d2phidr2_mw(x))) ** (1.0 / 3.0)


@jax.jit
def trh(Mc: float, rh: float, mbar: float) -> float:
    """
    Computes the relaxation time of a cluster
    Args:
      Mc: Cluster mass in [Msol]
      rh: half mass radius of cluster in [kpc]
      mbar: Average stellar mass in clsuter in [Msol]
    Returns:
      Relaxation time of the cluster in [Myr]
    Examples
    --------
    >>> trh(Mc=1e4, rh=1e-3, mbar=1.0)
    """
    return (
        0.138
        * jnp.sqrt(Mc)
        * rh ** (3 / 2)
        / (mbar * jnp.sqrt(G) * jnp.log(0.4 * (Mc / mbar)))
    )


@jax.jit
def lagrange_mw(
    key, x: jnp.ndarray, v: jnp.ndarray, Msat: float, params: Parameters
) -> tuple([jnp.ndarray, jnp.ndarray]):
    """
    Computes the two Lagrange points of the cluster in the simulation frame
    Args:
      key: jax.random.PRNGKey random key
      x: 3d position (x, y, z) in [kpc]
      v: 3d velocity (v_x, v_y, v_z) in [kpc/Myr]
      Msat: Cluster mass in [Msol]
      params: Parameters class containing stream model parameters
    Returns:
      Tuple of innermost and outermost lagrange points (x, y, z) in [kpc]
    Examples
    --------
    >>> lagrange_mw(jax.random.PRNGKey(0), jnp.array([1.0, 2.0, 3.0]), jnp.array([1.0, 2.0, 3.0]), 1e4, Parameters())
    """
    rt = tidalr_mw(x, v, Msat)
    rad = jnp.linalg.norm(x)
    rhat = x / rad
    xnear = x * (1.0 - params.lrelease * rt / rad)
    xfar = x * (1.0 + params.lrelease * rt / rad)
    xmatchnear = x * (1.0 - params.lmatch * rt / rad)
    xmatchfar = x * (1.0 + params.lmatch * rt / rad)
    omega = jnp.cross(x, v) / rad**2
    vr = jnp.dot(v, rhat) * rhat
    vt = v - vr
    alphanear = (
        jnp.linalg.norm(xmatchnear) ** 2
        * jnp.linalg.norm(omega)
        / jnp.linalg.norm(jnp.cross(xmatchnear, vt))
    )
    alphafar = (
        jnp.linalg.norm(xmatchfar) ** 2
        * jnp.linalg.norm(omega)
        / jnp.linalg.norm(jnp.cross(xmatchfar, vt))
    )
    vnear = vr + alphanear * vt
    vfar = vr + alphafar * vt
    unit_vector = sample_unit_vector(key)
    Xnear = jnp.concatenate(
        (xnear, vnear + jnp.sqrt(3.0) * params.sigv_kpcMyr * unit_vector)
    )
    Xfar = jnp.concatenate(
        (xfar, vfar + jnp.sqrt(3.0) * params.sigv_kpcMyr * unit_vector)
    )
    return Xnear, Xfar


def get_stars_X0(
    key, t_strip: float, clust_sol, mass_sol, params: Parameters
) -> jnp.ndarray:
    """
    Obtains the 6d initial conditions for a stripped stream star
    Args:
      key: jax.random.PRNGKey random key
      t_strip: stripping time in [Myr]
      clust_sol: cluster evolution solution, output of dynamics_solver
      mass_sol: mass evolution solution, output of mass_solver
      params: Parameters class containing stream model parameters
    Returns:
      6d initial conditions for a stripped star (x, y, z [kpc], v_x, v_y, v_z [kpc/Myr])
    Examples
    --------
    >>> params = Parameters()
    >>> clust_sol = dynamics_solver(params.cluster_final, params.age, 0.0, dense=True)
    >>> mass_sol = mass_solver(params, clust_sol)
    >>> get_stars_X0(jax.random.PRNGKey(0), 0.5 * params.age, clust_sol, mass_sol, params)
    """
    k1, k2 = jax.random.split(key)
    clust_X = clust_sol.evaluate(t_strip)
    Xnear, Xfar = lagrange_mw(
        k1, clust_X[0:3], clust_X[3:], mass_sol.evaluate(t_strip), params
    )
    p_near = jax.random.uniform(k2, (1,))
    return Xfar * jnp.heaviside(p_near - params.stripnear, 0.0) + Xnear * jnp.heaviside(
        params.stripnear - p_near, 1.0
    )


get_stars_X0_vmap = jax.vmap(
    get_stars_X0, (0, 0, None, None, None)
)  # Vectorized version of stripped star initial conditions generator


@jax.jit
def sample_unit_vector(key):
    """
    Sample a vector on the unit sphere
    Args:
      key: jax.random.PRNGKey random key
    Returns:
      3d vector on the unit sphere
    Examples
    --------
    >>> sample_unit_vector(jax.random.PRNGKey(0))
    """
    dv = jax.random.uniform(key, (3,))
    return dv / jnp.linalg.norm(dv)


@jax.jit
def _from_mass(
    t_sample: jnp.ndarray, mass_sample: jnp.ndarray
) -> tuple([chex.Array, chex.Array]):
    """
    Constructs the cumulative distribution and bin midpoints from samples of the stripping times and cluster mass
    Args:
      t_sample: sample of stripping times
      mass_sample: sample of masses at the corresponding strippping times
    Returns:
      (cumulative distribution function from binned samples,
       midpoints of the bins from binned samples)
    Examples
    --------
    >>> _from_mass(jnp.linspace(0., 100., 1000), jnp.linspace(0., 100., 1000))
    """
    freqs = mass_sample[:-1] - mass_sample[1:]
    h, bins = jnp.histogram((t_sample[1:] + t_sample[:-1]) / 2, t_sample, weights=freqs)
    bin_midpoints = bins[:-1] + jnp.diff(bins) / 2
    cdf = jnp.cumsum(h)
    cdf = cdf / cdf[-1]
    return cdf, bin_midpoints


@chex.dataclass
class StrippingSampler:
    """
    Dataclass for the sampler to generate stripping times
    Args:
      bin_midpoints: bin midpoints from time/mass biinning, output of _from_mass
      cdf: cumulative distribution of stripping times, output of _from
    Returns:
      Stripping sampler data class
    Examples
    --------
    >>> cdf, bin_midpoints = _from_mass(jnp.linspace(0., 100., 1000), jnp.linspace(0., 100., 1000))
    >>> strip_sampler = StrippingSampler(cdf=cdf, bin_midpoints=bin_midpoints)
    or
    >>> params = Parameters()
    >>> clust_sol = dynamics_solver(params.cluster_final, params.age, 0.0, dense=True)
    >>> mass_sol = mass_solver(params, clust_sol)
    >>> mass_fun_vmap = jax.vmap(mass_sol.evaluate, (0,))
    >>> t_sample = jnp.linspace(0.0, params.age, 16**3)
    >>> mass_sample = mass_fun_vmap(t_sample)
    >>> strip_sampler = StrippingSampler.from_mass(t_sample, mass_sample)
    """

    bin_midpoints: chex.Array
    cdf: chex.Array

    @classmethod
    def from_mass(cls, t_sample, mass_sample):
        cdf, bin_midpoints = _from_mass(t_sample, mass_sample)
        return cls(cdf=cdf, bin_midpoints=bin_midpoints)

    def sample(self, key, shape: tuple = ()) -> jnp.ndarray:
        """
        Sample a stripping time from the current stripping sampler
        Args:
          key: jax.random.PRNGKey random key
          shape: tuple of desired output shape
        Returns:
          Array of stripping times of given shape
        Examples
        --------
        >>> params = Parameters()
        >>> clust_sol = dynamics_solver(params.cluster_final, params.age, 0.0, dense=True)
        >>> mass_sol = mass_solver(params, clust_sol)
        >>> mass_fun_vmap = jax.vmap(mass_sol.evaluate, (0,))
        >>> t_sample = jnp.linspace(0.0, params.age, 16**3)
        >>> mass_sample = mass_fun_vmap(t_sample)
        >>> strip_sampler = StrippingSampler.from_mass(t_sample, mass_sample)
        >>> strip_sample.sample(jax.random.PRNGKey(0), shape=(6, 6))
        """
        values = jax.random.uniform(key, shape)
        value_bins = jnp.searchsorted(self.cdf, values)
        stripping_times = self.bin_midpoints[value_bins]
        return stripping_times
