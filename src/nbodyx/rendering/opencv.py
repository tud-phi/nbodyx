import cv2
from jax import Array
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from typing import List


def render_n_body_in_opencv(
    x_bds: Array,
    width: int,
    height: int,
    x_min: Array,
    x_max: Array,
    body_radii: Array = None,
    body_colors: Array = None,
) -> onp.ndarray:
    """Render the n-body problem using OpenCV.

    Args:
        x_bds: The positions of the bodies. Array of shape (num_bodies, 2).
        width: The width of the image.
        height: The height of the image.
        body_radii: The radii of the bodies. Array of shape (num_bodies, ).
        body_colors: The RGB colors of the bodies. Array of shape (num_bodies, 3).
    Returns:
        img: The rendered image.
    """
    # define ppm (pixels per meter)
    ppm = 0.9 * min(width, height) / jnp.max(x_max - x_min)

    # default body radii
    if body_radii is None:
        body_radii = (jnp.ones(x_bds.shape[0]) * 0.05 * min(width, height)).astype(int)

    # default body colors
    if body_colors is None:
        colors = plt.cm.get_cmap("tab10").colors
        body_colors = (jnp.array(colors) * 255).astype(int)

    # convert RGB to BGR colors
    body_colors = body_colors[:, ::-1]

    def x_to_uv(x: Array) -> Array:
        """Convert the position to the pixel coordinates."""
        normalized_position = (x - x_min) * ppm
        uv = jnp.array(
            [normalized_position[0], height - normalized_position[1]],
            dtype=int,
        )
        return uv

    # create the image
    img = onp.zeros((height, width, 3), dtype=onp.uint8)

    # draw the bodies
    for i in range(x_bds.shape[0]):
        uv = x_to_uv(x_bds[i])
        center = (int(uv[0]), int(uv[1]))
        color = tuple(body_colors[i].tolist())
        cv2.circle(img, center, int(body_radii[i]), color, -1)

    return img
