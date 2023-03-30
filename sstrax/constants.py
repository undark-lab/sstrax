import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, gammainc
import chex

G = 4.498502151469552e-12


@jax.jit
def to_kpcMyr2(f, vc0kms, R0kpc):
    # Tested 16/1
    return (0.000001045940172532453 * vc0kms**2 / R0kpc) * f


@jax.jit
def kms_to_kpcMyr(v_kms):
    # Tested 18/1
    return 0.001022712165045695 * v_kms


@jax.jit
def gamma(x):
    # Tested 16/1
    return jnp.exp(gammaln(x))


@jax.jit
def gamma_low(x, y):
    # Tested 16/1
    return jnp.exp(gammaln(x)) * (1.0 - gammainc(x, y))


@chex.dataclass
class DiskParams:
    a: float = 3.0
    b: float = 0.28
    fD: float = 0.6


@chex.dataclass
class BulgeParams:
    alpha: float = 1.8
    rc: float = 1.9
    fB: float = 0.05
    g: float = gamma(1.5 - (1.8 / 2))


@chex.dataclass
class NFWParams:
    rs: float = 16.0
    fNFW: float = 0.35


@chex.dataclass
class MWParams:
    vc0kms: float = 220.0
    R0kpc: float = 8.0
    ftot: float = to_kpcMyr2(1.0, 220.0, 8.0)


@chex.dataclass
class Parameters:
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


# Note: could also explicitly copy field names in to a list here.
PRIOR_LIST = [field.name for field in Parameters.__dataclass_fields__.values()]
