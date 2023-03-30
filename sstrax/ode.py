import jax
import jax.numpy as jnp
import numpy as np
from diffrax import (
    diffeqsolve,
    ODETerm,
    SaveAt,
    PIDController,
    Dopri5,
    DiscreteTerminatingEvent,
)
from functools import partial
from .backgrounds import force_mw
from .tidal import tidalr_mw, trh
from .constants import Parameters

_dynamics_solver = Dopri5()
_mass_solver = Dopri5()


@jax.jit
def dynamics_deriv(t, Y, args) -> jnp.ndarray:
    """
    Differential equation derivative for solving dynamics in Milky Way potential
    Args:
      t: independent variable (here time in [Myr])
      Y: array of dependent variables (here 6d phase space co-ordinates)
      args: Currently unused, but required as default
    Returns:
      dY/dt array from Milky Way force calculation
    Examples
    --------
    >>> dynamics_deriv(0.0, jnp.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]), None)
    """
    dx = Y[3:]
    dv = force_mw(Y[0:3])
    return jnp.concatenate((dx, dv))


@partial(jax.jit, static_argnames=("dense", "max_steps"))
def dynamics_solver(
    y0: jnp.ndarray,
    t0: float,
    t1: float,
    dense: bool = False,
    rtol: float = 1e-7,
    atol: float = 1e-7,
    max_steps: int = 16**3,
):
    """
    Differential equation solver for solving dynamics in Milky Way potential
    Args:
      y0: initial phase space position (x, y, z [kpc], v_x, v_y, v_z [kpc/Myr])
      t0: initial starting time in [Myr]
      t1: end time in [Myr]
      dense: choice to return solution at t1 (False) or interpolated function (True)
      rtol: relative tolerance
      atol: absolute tolerance
      max_steps: maximum number of solver steps (can raise ValueErorr if too small)
    Returns:
      Solution to the dynamics differential equation at t1 or an interpolated function
    Examples
    --------
    >>> params = Parameters()
    >>> clust_sol = dynamics_solver(params.cluster_final, params.age, 0.0, dense=True)
    """
    term = ODETerm(dynamics_deriv)
    solver = _dynamics_solver
    saveat = SaveAt(t0=False, t1=True, ts=None, dense=dense)
    stepsize_controller = PIDController(rtol=rtol, atol=atol)
    solution = diffeqsolve(
        terms=term,
        solver=solver,
        t0=t0,
        t1=t1,
        y0=y0,
        dt0=None,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        discrete_terminating_event=None,
        max_steps=(max_steps if dense else None),
    )
    return solution if dense else solution.ys[-1]


@jax.jit
def mass_deriv(t, M, args: dict):
    """
    Differential equation derivative for solving dynamics in Milky Way potential
    Args:
      t: independent variable (here time in [Myr])
      M: array of dependent variables (here cluster mass in [Msol])
      args: Dictionary containing the mass loss equation parameters and the numerical cluster solution
    Returns:
      dM/dt array from cluster mass loss equation
    Examples
    --------
    >>> params = Parameters()
    >>> clust_sol = dynamics_solver(params.cluster_final, params.age, 0.0, dense=True)
    >>> mass_sol = mass_solver(params, clust_sol)
    >>> args = {'mass_args': params.mass_args, 'cluster_solution': clust_sol}
    >>> mass_deriv(0.0, 1e4, args)
    """
    xi0, alpha, rh, mbar = args["mass_args"]
    cluster_solution = args["cluster_solution"]
    X = cluster_solution.evaluate(t)
    tidal_rad = tidalr_mw(X[0:3], X[3:], M)
    return -xi0 * jnp.sqrt(1.0 + (alpha * rh / tidal_rad) ** 3) * M / trh(M, rh, mbar)


def mass_termination_condition(state, **_):
    """
    Termination condition for the mass loss solver (stops if mass drops below 0)
    Args:
      state: current solver star
    Returns:
      Boolean value as to whether current value is suitably close to zero
    """
    return state.y <= np.finfo(state.y.dtype).eps


@partial(jax.jit, static_argnames=("max_steps",))
def mass_solver(
    params: Parameters,
    cluster_solution,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 16**3,
):
    """
    Differential equation solver for solving dynamics in Milky Way potential
    Args:
      params: Parameters class containing stream model parameters
      cluster_solution: cluster evolution solution, output of dynamics_solver
      rtol: relative tolerance
      atol: absolute tolerance
      max_steps: maximum number of solver steps (can raise ValueErorr if too small)
    Returns:
      Solution to the mass loss equation
    Examples
    --------
    >>> params = Parameters()
    >>> clust_sol = dynamics_solver(params.cluster_final, params.age, 0.0, dense=True)
    >>> mass_sol = mass_solver(params, clust_sol)
    """
    args = {"cluster_solution": cluster_solution, "mass_args": params.mass_args}
    term = ODETerm(mass_deriv)
    solver = _mass_solver
    saveat = SaveAt(t0=False, t1=True, ts=None, dense=True)
    stepsize_controller = PIDController(rtol=rtol, atol=atol)
    discrete_terminating_event = DiscreteTerminatingEvent(
        cond_fn=mass_termination_condition
    )
    solution = diffeqsolve(
        terms=term,
        solver=solver,
        t0=0.0,
        t1=params.age,
        y0=params.msat,
        dt0=None,
        args=args,
        max_steps=max_steps,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        discrete_terminating_event=discrete_terminating_event,
    )
    return solution
