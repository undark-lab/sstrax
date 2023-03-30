import jax
import jax.numpy as jnp
from .constants import DiskParams, MWParams, NFWParams, BulgeParams, gamma_low


@jax.jit
def force_mw(x):
    """
    Computes the force at a position x (in the simulation frames).
    Sums over the disk, bulge and NFW potential components.
    Args:
      x: 3d position (x, y, z) in [kpc]
    Returns:
      Force (per unit mass) in [kpc / Myr^2]
    Examples
    --------
    >>> force_mw(x=jnp.array([8.0, 0.0, 0.0]))
    """
    return force_disk(x) + force_bulge(x) + force_nfw(x)


jacobian_force_mw = jax.jit(jax.jacfwd(force_mw))


@jax.jit
def force_disk(x, disk=DiskParams(), mw=MWParams()):
    # Tested 16/1
    R2 = x[0] ** 2 + x[1] ** 2
    dimless_prefactor = (
        (mw.R0kpc**2 + (disk.a + disk.b) ** 2)
        / (R2 + (disk.a + jnp.sqrt(disk.b**2 + x[2] ** 2)) ** 2)
    ) ** (3 / 2)
    direction = (1 / mw.R0kpc) * jnp.array(
        [
            x[0],
            x[1],
            x[2]
            * (disk.a + jnp.sqrt(disk.b**2 + x[2] ** 2))
            / jnp.sqrt(disk.b**2 + x[2] ** 2),
        ]
    )
    return -disk.fD * mw.ftot * dimless_prefactor * direction


@jax.jit
def force_bulge(x, bulge=BulgeParams(), mw=MWParams()):
    # Tested 16/1
    rad = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** (1 / 2)
    dimless_prefactor = (
        mw.R0kpc**2
        * (bulge.g - gamma_low(1.5 - (bulge.alpha / 2), (rad / bulge.rc) ** 2))
    ) / (
        rad**2
        * (bulge.g - gamma_low(1.5 - (bulge.alpha / 2), (mw.R0kpc / bulge.rc) ** 2))
    )
    direction = (1 / rad) * x
    return -bulge.fB * mw.ftot * dimless_prefactor * direction


@jax.jit
def force_nfw(x, nfw=NFWParams(), mw=MWParams()):
    # Tested 16/1
    rad = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** (1 / 2)
    dimless_prefactor = (
        mw.R0kpc**2 * (rad / (nfw.rs + rad) - jnp.log((nfw.rs + rad) / nfw.rs))
    ) / (
        rad**2
        * (mw.R0kpc / (nfw.rs + mw.R0kpc) - jnp.log((nfw.rs + mw.R0kpc) / nfw.rs))
    )
    direction = (1 / rad) * x
    return -nfw.fNFW * mw.ftot * dimless_prefactor * direction


if __name__ == "__main__":
    pass
