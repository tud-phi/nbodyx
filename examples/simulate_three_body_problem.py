import cv2
import jax

jax.config.update("jax_enable_x64", True)
from diffrax import diffeqsolve, Dopri5, Euler, ODETerm, SaveAt
from jax import Array, jit, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
from nbodyx.constants import G
from nbodyx.ode import make_n_body_ode
from nbodyx.rendering.opencv import animate_n_body, render_n_body

if __name__ == "__main__":
    body_masses = jnp.array([1.0, 1.0, 1.0]) / G
    ode_fn = make_n_body_ode(body_masses)

    # initial positions
    x0 = jnp.array(
        [
            [-0.97000436, 0.24208753],
            [0.0, 0.0],
            [0.97000436, -0.24208753],
        ]
    )
    # initial velocities
    v0 = jnp.array(
        [
            [0.4662036850, 0.4323657300],
            [-0.93240737, -0.86473146],
            [0.4662036850, 0.4323657300],
        ]
    )

    # initial state
    y0 = jnp.stack([x0, v0], axis=0).reshape(-1)
    print("y0", y0)
    # external torques
    tau = jnp.zeros((6,))

    # state bounds
    x_min, x_max = -1.5 * jnp.ones((1,)), 1.5 * jnp.ones((1,))

    # render the image at the initial state
    img = render_n_body(x0, 500, 500, x_min, x_max)
    plt.figure(num="Sample rendering")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    # simulation settings
    ts = jnp.linspace(0.0, 6.3259, 1001)
    dt = ts[-1] * 1e-4

    # solve the ODE
    ode_term = ODETerm(ode_fn)
    sol = diffeqsolve(ode_term, Dopri5(), ts[0], ts[-1], dt, y0, tau, saveat=SaveAt(ts=ts), max_steps=None)
    y_bds_ts = sol.ys.reshape((len(ts), 2, 3, 2))
    # positions
    x_bds_ts = y_bds_ts[:, 0, ...]

    # plot the solution
    fig, ax = plt.subplots()
    ax.plot(x_bds_ts[:, 0, 0], x_bds_ts[:, 0, 1], label="Body 1")
    ax.plot(x_bds_ts[:, 1, 0], x_bds_ts[:, 1, 1], label="Body 2")
    ax.plot(x_bds_ts[:, 2, 0], x_bds_ts[:, 2, 1], label="Body 3")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    ax.set_aspect("equal")
    plt.grid(True)
    plt.box(True)
    plt.show()

    # animate the solution
    animate_n_body(
        ts,
        x_bds_ts,
        500,
        500,
        video_path="examples/outputs/three_body.mp4",
        speed_up=1.0,
        skip_step=5,
        x_min=x_min,
        x_max=x_max,
        timestamp_unit="s",
    )
