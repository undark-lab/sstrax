import jax
import jax.numpy as jnp


@jax.jit
def halo_to_sun(Xhalo: jnp.ndarray) -> jnp.ndarray:
    """
    Conversion from simulation frame to cartesian frame centred at Sun
    Args:
      Xhalo: 3d position (x [kpc], y [kpc], z [kpc]) in simulation frame
    Returns:
      3d position (x_s [kpc], y_s [kpc], z_s [kpc]) in Sun frame
    Examples
    --------
    >>> halo_to_sun(jnp.array([1.0, 2.0, 3.0]))
    """
    sunx = 8.0
    xsun = sunx - Xhalo[0]
    ysun = Xhalo[1]
    zsun = Xhalo[2]
    return jnp.array([xsun, ysun, zsun])


@jax.jit
def sun_to_gal(Xsun: jnp.ndarray) -> jnp.ndarray:
    """
    Conversion from sun cartesian frame to galactic co-ordinates
    Args:
      Xsun: 3d position (x_s [kpc], y_s [kpc], z_s [kpc]) in Sun frame
    Returns:
      3d position (r [kpc], b [rad], l [rad]) in galactic frame
    Examples
    --------
    >>> sun_to_gal(jnp.array([1.0, 2.0, 3.0]))
    """
    r = jnp.linalg.norm(Xsun)
    b = jnp.arcsin(Xsun[2] / r)
    l = jnp.arctan2(Xsun[1], Xsun[0])
    return jnp.array([r, b, l])


@jax.jit
def gal_to_equat(Xgal: jnp.ndarray) -> jnp.ndarray:
    """
    Conversion from galactic co-ordinates to equatorial co-ordinates
    Args:
      Xgal: 3d position (r [kpc], b [rad], l [rad]) in galactic frame
    Returns:
      3d position (r [kpc], alpha [rad], delta [rad]) in equatorial frame
    Examples
    --------
    >>> gal_to_equat(jnp.array([1.0, 2.0, 3.0]))
    """
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
def equat_to_gd1cart(Xequat: jnp.ndarray) -> jnp.ndarray:
    """
    Conversion from equatorial co-ordinates to cartesian GD1 co-ordinates
    Args:
      Xequat: 3d position (r [kpc], alpha [rad], delta [rad]) in equatorial frame
    Returns:
      3d position (x_gd1 [kpc], y_gd1 [kpc], z_gd1 [kpc]) in cartesian GD1 frame
    Examples
    --------
    >>> equat_to_gd1cart(jnp.array([1.0, 2.0, 3.0]))
    """
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
def gd1cart_to_gd1(Xgd1cart: jnp.ndarray) -> jnp.ndarray:
    """
    Conversion from cartesian GD1 co-ordinates to angular GD1 co-ordinates
    Args:
      Xgd1cart: 3d position (x_gd1 [kpc], y_gd1 [kpc], z_gd1 [kpc]) in cartesian GD1 frame
    Returns:
      3d position (r [kpc], phi1 [rad], phi2 [rad]) in angular GD1 frame
    Examples
    --------
    >>> gd1cart_to_gd1(jnp.array([1.0, 2.0, 3.0]))
    """
    r = jnp.linalg.norm(Xgd1cart)
    phi1 = jnp.arctan2(Xgd1cart[1], Xgd1cart[0])
    phi2 = jnp.arcsin(Xgd1cart[2] / r)
    return jnp.array([r, phi1, phi2])


@jax.jit
def halo_to_gd1(Xhalo: jnp.ndarray) -> jnp.ndarray:
    """
    Composed conversion from simulation frame co-ordinates to angular GD1 co-ordinates
    Args:
      Xhalo: 3d position (x [kpc], y [kpc], z [kpc]) in simulation frame
    Returns:
      3d position (r [kpc], phi1 [rad], phi2 [rad]) in angular GD1 frame
    Examples
    --------
    >>> halo_to_gd1(jnp.array([1.0, 2.0, 3.0]))
    """
    Xsun = halo_to_sun(Xhalo)
    Xgal = sun_to_gal(Xsun)
    Xequat = gal_to_equat(Xgal)
    Xgd1cart = equat_to_gd1cart(Xequat)
    Xgd1 = gd1cart_to_gd1(Xgd1cart)
    return Xgd1


jacobian_halo_to_gd1 = jax.jit(
    jax.jacfwd(halo_to_gd1)
)  # Jacobian for computing the velocity transformation from simulation frame to angular GD1 co-ordinates

halo_to_gd1_vmap = jax.jit(
    jax.vmap(halo_to_gd1, (0,))
)  # Vectorised version of co-ordinate transformation from simulation frame to angular GD1 co-ordinates


@jax.jit
def equat_to_gd1(Xequat: jnp.ndarray) -> jnp.ndarray:
    """
    Composed conversion from equatorial frame co-ordinates to angular GD1 co-ordinates
    Args:
      Xhalo: 3d position (x [kpc], y [kpc], z [kpc]) in simulation frame
    Returns:
      3d position (r [kpc], phi1 [rad], phi2 [rad]) in angular GD1 frame
    Examples
    --------
    >>> equat_to_gd1(jnp.array([1.0, 2.0, 3.0]))
    """
    Xgd1cart = equat_to_gd1cart(Xequat)
    Xgd1 = gd1cart_to_gd1(Xgd1cart)
    return Xgd1


jacobian_equat_to_gd1 = jax.jit(
    jax.jacfwd(equat_to_gd1)
)  # Jacobian for computing the velocity transformation from equatorial frame to angular GD1 co-ordinates


@jax.jit
def equat_to_gd1_velocity(Xequat: jnp.ndarray, Vequat: jnp.ndarray) -> jnp.ndarray:
    """
    Velocity conversion from equatorial frame co-ordinates to angular GD1 co-ordinates
    Args:
      Xequat: 3d position (r [kpc], alpha [rad], delta [rad]) in equatorial frame
      Vequat: 3d velocity (v_r [kpc/Myr], v_alpha [rad/Myr], v_delta [rad/Myr]) in equatorial frame
    Returns:
      3d velocity (v_r [kpc/Myr], v_phi1 [rad/Myr], v_phi2 [rad/Myr]) in angular GD1 frame
    Examples
    --------
    >>> equat_to_gd1_velocity(jnp.array([1.0, 2.0, 3.0]), jnp.array([1.0, 2.0, 3.0]))
    """
    return jnp.matmul(jacobian_equat_to_gd1(Xequat), Vequat)


@jax.jit
def halo_to_gd1_velocity(Xhalo: jnp.ndarray, Vhalo: jnp.ndarray) -> jnp.ndarray:
    """
    Velocity conversion from equatorial frame co-ordinates to angular GD1 co-ordinates
    Args:
      Xhalo: 3d position (x [kpc], y [kpc], z [kpc]) in simulation frame
      Vhalo: 3d velocity (v_x [kpc/Myr], v_y [kpc/Myr], v_z [kpc/Myr]) in simulation frame
    Returns:
      3d velocity (v_r [kpc/Myr], v_phi1 [rad/Myr], v_phi2 [rad/Myr]) in angular GD1 frame
    Examples
    --------
    >>> halo_to_gd1_velocity(jnp.array([1.0, 2.0, 3.0]), jnp.array([1.0, 2.0, 3.0]))
    """
    return jnp.matmul(jacobian_halo_to_gd1(Xhalo), Vhalo)


halo_to_gd1_velocity_vmap = jax.jit(
    jax.vmap(halo_to_gd1_velocity, (0, 0))
)  # Vectorised version of velocity co-ordinate transformation from simulation frame to angular GD1 co-ordinates


@jax.jit
def halo_to_gd1_all(Xhalo: jnp.ndarray, Vhalo: jnp.ndarray) -> jnp.ndarray:
    """
    Position and Velocity conversion from equatorial frame co-ordinates to angular GD1 co-ordinates
    Args:
      Xhalo: 3d position (x [kpc], y [kpc], z [kpc]) in simulation frame
      Vhalo: 3d velocity (v_x [kpc/Myr], v_y [kpc/Myr], v_z [kpc/Myr]) in simulation frame
    Returns:
      6d phase space (x [kpc], y [kpc], z[kpv], v_r [kpc/Myr], v_phi1 [rad/Myr], v_phi2 [rad/Myr]) in angular GD1 frame
    Examples
    --------
    >>> halo_to_gd1_all(jnp.array([1.0, 2.0, 3.0]), jnp.array([1.0, 2.0, 3.0]))
    """
    return jnp.concatenate((halo_to_gd1(Xhalo), halo_to_gd1_velocity(Xhalo, Vhalo)))


gd1_projection_vmap = jax.jit(
    jax.vmap(halo_to_gd1_all, (0, 0))
)  # Vectorised version of position and velocity co-ordinate transformation from simulation frame to angular GD1 co-ordinates
