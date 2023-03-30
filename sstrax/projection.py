import jax
import jax.numpy as jnp


@jax.jit
def halo_to_sun(Xhalo):
    # Tested 17/1
    # Returns [xsun, ysun, zsun]
    sunx = 8.0
    xsun = sunx - Xhalo[0]
    ysun = Xhalo[1]
    zsun = Xhalo[2]
    return jnp.array([xsun, ysun, zsun])


@jax.jit
def sun_to_gal(Xsun):
    # Tested 17/1
    # Returns [r, b, l]
    r = jnp.linalg.norm(Xsun)
    b = jnp.arcsin(Xsun[2] / r)
    l = jnp.arctan2(Xsun[1], Xsun[0])
    return jnp.array([r, b, l])


@jax.jit
def gal_to_equat(Xgal):
    # Tested 17/1
    # Returns [r, alpha, delta]
    dNGPdeg = 27.12825118085622
    lNGPdeg = 122.9319185680026
    aNGPdeg = 192.85948
    dNGP = dNGPdeg * jnp.pi / 180.0
    lNGP = lNGPdeg * jnp.pi / 180.0
    aNGP = aNGPdeg * jnp.pi / 180.0
    r = Xgal[0]
    b = Xgal[1]
    l = Xgal[2]
    sb = jnp.sin(b)
    cb = jnp.cos(b)
    sl = jnp.sin(lNGP - l)
    cl = jnp.cos(lNGP - l)
    cs = cb * sl
    cc = jnp.cos(dNGP) * sb - jnp.sin(dNGP) * cb * cl
    alpha = jnp.arctan(cs / cc) + aNGP
    delta = jnp.arcsin(jnp.sin(dNGP) * sb + jnp.cos(dNGP) * cb * cl)
    return jnp.array([r, alpha, delta])


@jax.jit
def equat_to_gd1cart(Xequat):
    # Tested 17/1
    # Returns [xgd1, ygd1, zgd1]
    xgd1 = Xequat[0] * (
        -0.4776303088 * jnp.cos(Xequat[1]) * jnp.cos(Xequat[2])
        - 0.1738432154 * jnp.sin(Xequat[1]) * jnp.cos(Xequat[2])
        + 0.8611897727 * jnp.sin(Xequat[2])
    )
    ygd1 = Xequat[0] * (
        0.510844589 * jnp.cos(Xequat[1]) * jnp.cos(Xequat[2])
        - 0.8524449229 * jnp.sin(Xequat[1]) * jnp.cos(Xequat[2])
        + 0.111245042 * jnp.sin(Xequat[2])
    )
    zgd1 = Xequat[0] * (
        0.7147776536 * jnp.cos(Xequat[1]) * jnp.cos(Xequat[2])
        + 0.4930681392 * jnp.sin(Xequat[1]) * jnp.cos(Xequat[2])
        + 0.4959603976 * jnp.sin(Xequat[2])
    )
    return jnp.array([xgd1, ygd1, zgd1])


@jax.jit
def gd1cart_to_gd1(Xgd1cart):
    # Tested 17/1
    # Returns [r, phi1, phi2]
    r = jnp.linalg.norm(Xgd1cart)
    phi1 = jnp.arctan2(Xgd1cart[1], Xgd1cart[0])
    phi2 = jnp.arcsin(Xgd1cart[2] / r)
    return jnp.array([r, phi1, phi2])


@jax.jit
def halo_to_gd1(Xhalo):
    # Tested 17/1
    # Returns [r, phi1, phi2]
    Xsun = halo_to_sun(Xhalo)
    Xgal = sun_to_gal(Xsun)
    Xequat = gal_to_equat(Xgal)
    Xgd1cart = equat_to_gd1cart(Xequat)
    Xgd1 = gd1cart_to_gd1(Xgd1cart)
    return Xgd1


jacobian_halo_to_gd1 = jax.jit(jax.jacfwd(halo_to_gd1))

halo_to_gd1_vmap = jax.jit(jax.vmap(halo_to_gd1, (0,)))


@jax.jit
def equat_to_gd1(Xequat):
    # Tested 17/1
    # Returns [r, phi1, phi2]
    Xgd1cart = equat_to_gd1cart(Xequat)
    Xgd1 = gd1cart_to_gd1(Xgd1cart)
    return Xgd1


jacobian_equat_to_gd1 = jax.jit(jax.jacfwd(equat_to_gd1))


@jax.jit
def equat_to_gd1_velocity(Xequat, Vequat):
    # Tested 17/1
    # Returns [Vr, Vphi1, Vphi2]
    return jnp.matmul(jacobian_equat_to_gd1(Xequat), Vequat)


@jax.jit
def halo_to_gd1_velocity(Xhalo, Vhalo):
    # Tested 17/1
    # Returns [Vr, Vphi1, Vphi2]
    return jnp.matmul(jacobian_halo_to_gd1(Xhalo), Vhalo)


halo_to_gd1_velocity_vmap = jax.jit(jax.vmap(halo_to_gd1_velocity, (0, 0)))


@jax.jit
def halo_to_gd1_all(Xhalo, Vhalo):
    return jnp.concatenate((halo_to_gd1(Xhalo), halo_to_gd1_velocity(Xhalo, Vhalo)))


gd1_projection_vmap = jax.jit(jax.vmap(halo_to_gd1_all, (0, 0)))

if __name__ == "__main__":
    pass
