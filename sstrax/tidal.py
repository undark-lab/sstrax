import chex
import jax
import jax.numpy as jnp
from .backgrounds import jacobian_force_mw
from .constants import G


@jax.jit
def d2phidr2_mw(x):
    # Tested 17/1
    rad = jnp.linalg.norm(x)
    return -jnp.matmul(jnp.transpose(x), jnp.matmul(jacobian_force_mw(x), x)) / rad**2


@jax.jit
def omega(x, v):
    # Tested 17/1
    rad = jnp.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    omega_vec = jnp.cross(x, v) / rad**2
    return jnp.linalg.norm(omega_vec)


@jax.jit
def tidalr_mw(x, v, Msat):
    # Tested 17/1
    return (G * Msat / (omega(x, v) ** 2 - d2phidr2_mw(x))) ** (1.0 / 3.0)


@jax.jit
def trh(Mc, rh, mbar):
    # Tested 17/1
    return (
        0.138
        * jnp.sqrt(Mc)
        * rh ** (3 / 2)
        / (mbar * jnp.sqrt(G) * jnp.log(0.4 * (Mc / mbar)))
    )


@jax.jit
def lagrange_mw(key, x, v, Msat, params):
    # Tested 17/1
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


def get_stars_X0(key, t_strip, cluster_sol, mass_sol, params):
    k1, k2 = jax.random.split(key)
    clust_X = cluster_sol.evaluate(t_strip)
    Xnear, Xfar = lagrange_mw(
        k1, clust_X[0:3], clust_X[3:], mass_sol.evaluate(t_strip), params
    )
    p_near = jax.random.uniform(k2, (1,))
    return Xfar * jnp.heaviside(p_near - params.stripnear, 0.0) + Xnear * jnp.heaviside(
        params.stripnear - p_near, 1.0
    )


get_stars_X0_vmap = jax.vmap(get_stars_X0, (0, 0, None, None, None))


@jax.jit
def sample_unit_vector(key):
    # Tested 17/1
    dv = jax.random.uniform(key, (3,))
    return dv / jnp.linalg.norm(dv)


@jax.jit
def _from_mass(t_sample, mass_sample):
    freqs = mass_sample[:-1] - mass_sample[1:]
    h, bins = jnp.histogram((t_sample[1:] + t_sample[:-1]) / 2, t_sample, weights=freqs)
    bin_midpoints = bins[:-1] + jnp.diff(bins) / 2
    cdf = jnp.cumsum(h)
    cdf = cdf / cdf[-1]
    return cdf, bin_midpoints


@chex.dataclass
class StrippingSampler:
    bin_midpoints: chex.Array
    cdf: chex.Array

    @classmethod
    def from_mass(cls, t_sample, mass_sample):
        cdf, bin_midpoints = _from_mass(t_sample, mass_sample)
        return cls(cdf=cdf, bin_midpoints=bin_midpoints)

    def sample(self, key, shape=()):
        values = jax.random.uniform(key, shape)
        value_bins = jnp.searchsorted(self.cdf, values)
        stripping_times = self.bin_midpoints[value_bins]
        return stripping_times


if __name__ == "__main__":
    pass
