import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, gammainc
import chex

G = 4.498502151469552e-12  # in units of kpc^3 Myr^-2 Msol^-1


@jax.jit
def to_kpcMyr2(f: float, vc0kms: float, R0kpc: float) -> float:
    """
    Rescales a force (per unit mass) f by a factor vc0kms^2 / R0kpc in kpc^2 / Myr
    Args:
      f: force to be rescaled
      vc0kms: circular velocity in [km/s]
      R0kpc: radius in [kpc]
    Returns:
      Force (per unit mass) in [kpc / Myr^2]
    Examples
    --------
    >>> to_kpcMyr2(1., 220., 8.)
    """
    return (0.000001045940172532453 * vc0kms**2 / R0kpc) * f


@jax.jit
def kms_to_kpcMyr(v_kms: float) -> float:
    """
    Converts velocity from km/s to kpc/Myr
    Args:
      v_kms: velocity in [km/s]
    Returns:
      Force (per unit mass) in [kpc / Myr^2]
    Examples
    --------
    >>> kms_to_kpcMyr(220.)
    """
    return 0.001022712165045695 * v_kms


@jax.jit
def gamma(x: float) -> float:
    """
    Compiled version of the standard Gamma function
    Args:
      x: input value
    Returns:
      Gamma function evaludated at x
    Examples
    --------
    >>> gamma(2.)
    """
    return jnp.exp(gammaln(x))


@jax.jit
def gamma_low(x: float, y: float) -> float:
    """
    Compiled version of the incomplete gamma function from below (integral from 0 to y)
    Args:
      x: input value
      y: upper integration limit
    Returns:
      Incomplete gamma function from below evaludated at x
    Examples
    --------
    >>> gamma_low(2., 10.)
    """
    return jnp.exp(gammaln(x)) * (1.0 - gammainc(x, y))


@chex.dataclass
class DiskParams:
    """
    Dataclass for the parameters defining the Milky Way disk
    Args:
      a: Disk major axis in [kpc]
      b: Disk minor axis in [kpc]
      fD: relative contribution of disk to force
    Returns:
      Disk parameters data class
    Examples
    --------
    >>> DiskParams(a=3.0, b=0.28, fD=0.6)
    """

    a: float = 3.0
    b: float = 0.28
    fD: float = 0.6


@chex.dataclass
class BulgeParams:
    """
    Dataclass for the parameters defining the Milky Way disk
    Args:
      alpha: power law exponent
      rc: cut-off radius of bulge in [kpc]
      fB: relative contribution of bulge to force
    Returns:
      Bulge parameters data class
    Examples
    --------
    >>> BulgeParams(alpha=1.8, rc=1.9, fB=0.05)
    """

    alpha: float = 1.8
    rc: float = 1.9
    fB: float = 0.05

    @property
    def g(self):
        return gamma(1.5 - (self.alpha / 2))


@chex.dataclass
class NFWParams:
    """
    Dataclass for the parameters defining the Milky Way NFW halo
    Args:
      rs: scale radius of NFW halo
      fNFW: relative contribution of NFW halo
    Returns:
      NFW parameters data class
    Examples
    --------
    >>> NFWParams(rc=16.0, fNFW=0.35)
    """

    rs: float = 16.0
    fNFW: float = 0.35


@chex.dataclass
class MWParams:
    """
    Dataclass for the parameters defining the Milky Way force
    Args:
      vc0kms: measured circular velocity at a distance R0kpc from galactic centre in [km/s]
      R0kpc: calibration distance in [kpc]
    Returns:
      MW parameters data class
    Examples
    --------
    >>> MWParams(vc0kms=220.0, R0kpc=8.0)
    """

    vc0kms: float = 220.0
    R0kpc: float = 8.0

    @property
    def ftot(self):
        return to_kpcMyr2(1.0, self.vc0kms, self.R0kpc)


@chex.dataclass
class Parameters:
    """
    Dataclass for the full set of parameters defining the stream simulation model
    Args:
      xc, yc, zc: current cluster location in [kpc]
      vxc, vyc, vzc: current cluster velocity in [km/s]
      age: (disruption) age of the stream in [Myr]
      msat: initial mass of the cluster in [Msol]
      xi0: dimensionless mass loss prefactor
      alpha: tidal strippping mass loss power law
      mbar: average stellar mass in [Msol]
      sigv: internal velocity dispersion of cluster in [km/s]
      lrelease, lmatch: dimensionless tidal position and velocity matching
      stripnear: probability of stripping from the innermost Lagrange point
    Returns:
      Stream parameters data class
    Examples
    --------
    >>> Parameters(xc=12.4, yc=1.5,...)
    """

    xc: float = 12.4
    yc: float = 1.5
    zc: float = 7.1
    vxc: float = 107.0
    vyc: float = -243.0
    vzc: float = -105.0
    age: float = 1000.0
    msat: float = 1e4
    # stripping parameters
    xi0: float = 0.001
    alpha: float = 14.9
    rh: float = 0.001
    mbar: float = 2.0
    sigv: float = 0.5
    lrelease: float = 1.9
    lmatch: float = 1.0
    stripnear: float = 0.5

    @property
    def cluster_pos_final(self):
        return jnp.array([self.xc, self.yc, self.zc])

    @property
    def cluster_vel_final(self):
        return kms_to_kpcMyr(jnp.array([self.vxc, self.vyc, self.vzc]))

    @property
    def cluster_final(self):
        return jnp.concatenate((self.cluster_pos_final, self.cluster_vel_final))

    @property
    def sigv_kpcMyr(self):
        return kms_to_kpcMyr(self.sigv)

    @property
    def mass_args(self):
        return jnp.array([self.xi0, self.alpha, self.rh, self.mbar])

    @classmethod
    def from_values(cls, values, names=None):
        if not isinstance(values, dict):
            values = {k: v for k, v in zip(names, values)}
        return cls(**values)


PRIOR_LIST = [field.name for field in Parameters.__dataclass_fields__.values()]
