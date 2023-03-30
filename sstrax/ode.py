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

_dynamics_solver = Dopri5()
_mass_solver = Dopri5()


@jax.jit
def dynamics_deriv(t, Y, args):
    # Tested 17/1
    dx = Y[3:]
    dv = force_mw(Y[0:3])
    return jnp.concatenate((dx, dv))


@partial(jax.jit, static_argnames=("dense", "max_steps"))
def dynamics_solver(y0, t0, t1, dense=False, rtol=1e-7, atol=1e-7, max_steps=16**3):
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
def mass_deriv(t, M, args):
    # Tested 17/1
    xi0, alpha, rh, mbar = args["mass_args"]
    cluster_solution = args["cluster_solution"]
    X = cluster_solution.evaluate(t)
    tidal_rad = tidalr_mw(X[0:3], X[3:], M)
    return -xi0 * jnp.sqrt(1.0 + (alpha * rh / tidal_rad) ** 3) * M / trh(M, rh, mbar)


def mass_termination_condition(state, **_):
    return state.y <= np.finfo(state.y.dtype).eps


@partial(jax.jit, static_argnames=("max_steps",))
def mass_solver(params, cluster_solution, rtol=1e-8, atol=1e-8, max_steps=16**3):
    # Tested 17/1
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


if __name__ == "__main__":
    pass
