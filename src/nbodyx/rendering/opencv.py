import cv2
from jax import Array
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from os import PathLike
from pathlib import Path
from tqdm import tqdm


def render_n_body(
    x_bds: Array,
    width: int,
    height: int,
    x_min: Array,
    x_max: Array,
    body_radii: Array = None,
    body_colors: Array = None,
    label: str = None,
) -> onp.ndarray:
    """Render the n-body problem using OpenCV.

    Args:
        x_bds: The positions of the bodies. Array of shape (num_bodies, 2).
        width: The width of the image.
        height: The height of the image.
        body_radii: The radii of the bodies. Array of shape (num_bodies, ).
        body_colors: The RGB colors of the bodies. Array of shape (num_bodies, 3).
        label: The label of the image.
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

    # draw the label
    if label is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (10, 50)
        font_scale = 0.5
        font_color = (255, 255, 255)
        line_type = 2
        cv2.putText(
            img,
            label,
            bottom_left_corner_of_text,
            font,
            font_scale,
            font_color,
            line_type,
        )

    return img


def animate_n_body(
    ts: Array,
    x_bds_ts: Array,
    width: int,
    height: int,
    video_path: PathLike,
    speed_up: int = 1,
    skip_step: int = 1,
    add_timestamp: bool = True,
    timestamp_unit: str = "s",
    **kwargs,
):
    dt = jnp.mean(jnp.diff(ts)).item()
    fps = float(speed_up / (skip_step * dt))
    print(f"fps: {fps}")

    # create video
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(
        str(video_path),
        fourcc,
        fps,  # fps,
        (width, height),
    )

    # skip frames
    ts = ts[::skip_step]
    x_bds_ts = x_bds_ts[::skip_step]

    for time_idx, t in (pbar := tqdm(enumerate(ts))):
        pbar.set_description(f"Rendering frame {time_idx + 1}/{len(ts)}")

        label = None
        if add_timestamp:
            if timestamp_unit == "s":
                label = f"t = {t:.1f} seconds"
            elif timestamp_unit == "d":
                label = f"t = {t / (24 * 3600):.1f} days"
            elif timestamp_unit == "M":
                label = f"t = {t / (30 * 24 * 3600):.1f} months"
            elif timestamp_unit == "y":
                label = f"t = {t / (365 * 24 * 3600):.1f} years"
            else:
                raise ValueError(f"Invalid timestamp unit: {timestamp_unit}")

        # render the image
        img = render_n_body(x_bds_ts[time_idx], width, height, label=label, **kwargs)

        video.write(img)

    video.release()
