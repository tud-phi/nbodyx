import jax

jax.config.update("jax_enable_x64", True)
from diffrax import diffeqsolve, Dopri5, Euler, ODETerm, SaveAt
from jax import Array, jit, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
from nbodyx.constants import G, M_sun, M_earth, AU
from nbodyx.ode import make_n_body_ode

if __name__ == "__main__":
    ode_fn = make_n_body_ode(jnp.array([M_sun, M_earth]))

    # initial conditions for earth
    x_earth = jnp.array([-AU, 0.0])
    theta_earth = jnp.arctan2(x_earth[1], x_earth[0])
    v0_earth = jnp.sqrt(G * M_sun / AU)
    v0_earth = jnp.array([-v0_earth * jnp.sin(theta_earth), v0_earth * jnp.cos(theta_earth)])

    # initial conditions for sun
    x_sun = jnp.array([0.0, 0.0])
    v_sun = jnp.array([0.0, 0.0])

    # initial state
    y0 = jnp.concatenate([x_sun, x_earth, v_sun, v0_earth])
    print("y0", y0)

    # evaluate the ODE at the initial state
    y_d0 = jit(ode_fn)(0.0, y0)
    print("y_d", y_d0)

    # simulation settings
    year_sec = 365 * 24 * 3600
    ts = jnp.linspace(0.0, 10 * year_sec, 365)
    dt = year_sec / (1e1 * 365)

    # solve the ODE
    ode_term = ODETerm(ode_fn)
    sol = diffeqsolve(ode_term, Dopri5(), ts[0], ts[-1], dt, y0, saveat=SaveAt(ts=ts), max_steps=None)
    y_bds_ts = sol.ys.reshape((len(ts), 2, 2, 2))
    # positions
    x_bds_ts = y_bds_ts[:, 0, ...]

    # plot the solution
    fig, ax = plt.subplots()
    ax.plot(x_bds_ts[:, 0, 0], x_bds_ts[:, 0, 1], label="Sun")
    ax.plot(x_bds_ts[:, 1, 0], x_bds_ts[:, 1, 1], label="Earth")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    ax.set_aspect("equal")
    plt.grid(True)
    plt.box(True)
    plt.show()
