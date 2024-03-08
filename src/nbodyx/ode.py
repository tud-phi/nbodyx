import jax
from jax import Array, debug, vmap
import jax.numpy as jnp
import numpy as onp
from typing import Callable, Union

from nbodyx.constants import G


def make_n_body_ode(body_masses: Array) -> Callable:
    """Generate the ODE function for the n-body problem.

    Args:
        body_masses: The masses of the bodies.
    Returns:
        ode_fn: The ODE function implementing the signature `ode(t, y) -> y_d`.
    """
    num_bodies = len(body_masses)

    def ode_fn(t: Union[float, Array], y: Array, tau: Array, *args) -> Array:
        """The ODE function for the n-body problem.

        Args:
            t: The time.
            y: The state vector of shape (2 * num_bodies * num_dims, )
            tau: Any external forces acting on the bodies. Array of shape (num_bodies*num_dims).
        Returns:
            y_d: The derivative of the state vector of shape (2 * num_bodies * num_dims, )
        """
        y_bds = y.reshape((2, num_bodies, -1))  # shape (2, num_bodies, num_dims)
        tau_bds = tau.reshape((num_bodies, -1))  # shape (num_bodies, num_dims)
        # positions
        x_bds = y_bds[0]
        # velocities
        v_bds = y_bds[1]

        # Compute the accelerations.
        a_bds = []
        for i in range(num_bodies):
            # element mask
            # exclude forces of the body on itself
            mask = onp.arange(num_bodies) != i

            # acceleration for the i-th body
            a_bd = -G * jnp.einsum(
                "j,jk,j->k",
                body_masses[mask],
                (x_bds[i] - x_bds[mask, :]),
                1 / jnp.linalg.norm(x_bds[i] - x_bds[mask, :], axis=-1) ** 3,
            ) + tau_bds[i] / body_masses[i]

            a_bds.append(a_bd)
        a_bds = jnp.stack(a_bds, axis=0)

        # Define the state derivative.
        y_d_bds = jnp.stack([v_bds, a_bds], axis=0)
        y_d = y_d_bds.reshape(-1)

        return y_d

    return ode_fn
