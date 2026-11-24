"""Depth map visualisation."""
import matplotlib
import numpy as np

_CMAP = matplotlib.colormaps.get_cmap('Spectral_r')


def normalize_to_uint8(depth, lo=None, hi=None):
    """Map depth to 0..255. Uses the map's own min/max unless a range is given."""
    lo = depth.min() if lo is None else lo
    hi = depth.max() if hi is None else hi
    scaled = (depth - lo) / max(hi - lo, 1e-8)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def colorize(depth, lo=None, hi=None):
    """Relative depth -> RGB uint8 image with the Spectral colormap (red = near)."""
    return (_CMAP(normalize_to_uint8(depth, lo, hi))[:, :, :3] * 255).astype(np.uint8)
