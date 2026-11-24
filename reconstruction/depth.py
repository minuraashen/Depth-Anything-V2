"""Turning raw network predictions into usable depth and validity masks."""
import cv2
import numpy as np


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def prediction_to_depth(prediction, config):
    """Return (depth, background_mask) for a raw prediction.

    Relative models predict affine-invariant inverse depth: true disparity = a * pred + b
    with unknown a and b. The scale a only zooms the scene, but the shift b changes its
    shape, so we choose b such that depth spans [1, near_far_ratio].
    """
    prediction = prediction.astype(np.float32)

    if config.depth_type == 'relative':
        d = normalize(prediction)
        d_min = 1.0 / config.near_far_ratio
        depth = 1.0 / (d_min + d * (1.0 - d_min))
        background = d <= config.background_disparity
    elif config.depth_type == 'metric':
        depth = np.maximum(prediction, 1e-6)
        if config.max_depth is not None:
            background = depth >= config.background_depth_fraction * config.max_depth
        else:
            background = np.zeros_like(depth, dtype=bool)
    else:
        raise ValueError(f'unknown depth_type {config.depth_type!r}')

    return depth, background


def depth_edge_mask(depth, rel_threshold):
    """True where the 3x3 neighbourhood spans a relative depth jump above rel_threshold.

    The network blends foreground and background depth at object boundaries; unprojected,
    those pixels become streaks ("flying pixels") between the two surfaces.
    """
    kernel = np.ones((3, 3), np.uint8)
    d_max = cv2.dilate(depth, kernel)
    d_min = cv2.erode(depth, kernel)
    return (d_max - d_min) / d_min > rel_threshold
