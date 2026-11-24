"""RGB image + depth prediction -> coloured 3D scene."""
import cv2
import numpy as np

from .camera import Intrinsics
from .config import ReconstructionConfig
from .depth import depth_edge_mask, prediction_to_depth
from .scene import Scene


def _resample(rgb, prediction, max_side):
    """Bring rgb to the prediction's size, then shrink both so the longest side <= max_side."""
    h, w = prediction.shape
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    scale = 1.0 if max_side is None else min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return rgb, prediction, 1.0, 1.0
    gw, gh = max(2, round(w * scale)), max(2, round(h * scale))
    rgb = cv2.resize(rgb, (gw, gh), interpolation=cv2.INTER_AREA)
    prediction = cv2.resize(prediction, (gw, gh), interpolation=cv2.INTER_AREA)
    return rgb, prediction, gw / w, gh / h


def _grid_faces(h, w):
    """Two triangles per pixel quad, counter-clockwise as seen from the camera (OpenGL frame)."""
    idx = np.arange(h * w).reshape(h, w)
    tl, tr = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    bl, br = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    return np.concatenate([np.stack([tl, bl, tr], 1), np.stack([tr, bl, br], 1)])


def reconstruct(rgb, prediction, config=None):
    """Reconstruct a coloured 3D scene from a single image.

    Args:
        rgb: (H, W, 3) uint8 RGB image (extra channels such as alpha are ignored).
        prediction: (H', W') raw Depth Anything output (relative disparity or metric depth).
        config: ReconstructionConfig (defaults to relative depth).
    Returns:
        Scene whose points/colours form the point cloud and whose faces form a mesh
        that never bridges a depth discontinuity.
    """
    config = config or ReconstructionConfig()
    rgb = np.ascontiguousarray(rgb[..., :3])
    h, w = prediction.shape

    if config.focal_length_x is not None:
        K = Intrinsics.from_focal(w, h, config.focal_length_x, config.focal_length_y)
    else:
        K = Intrinsics.from_fov(w, h, config.fov_deg)

    rgb, prediction, sx, sy = _resample(rgb, prediction.astype(np.float32), config.max_side)
    K = K.scaled(sx, sy)
    gh, gw = prediction.shape

    depth, background = prediction_to_depth(prediction, config)
    valid = np.isfinite(depth) & (depth > 0)
    if config.edge_threshold > 0:
        valid &= ~depth_edge_mask(depth, config.edge_threshold)
    if config.remove_background:
        valid &= ~background

    points = K.unproject(depth).reshape(-1, 3)
    faces = _grid_faces(gh, gw)
    if config.coordinate_frame == 'opengl':
        points = points * np.array([1, -1, -1], dtype=np.float32)
    elif config.coordinate_frame == 'opencv':
        faces = faces[:, ::-1]  # keep triangles facing the camera in the y-down frame
    else:
        raise ValueError(f'unknown coordinate_frame {config.coordinate_frame!r}')
    colors = rgb.reshape(-1, 3)

    valid = valid.reshape(-1)
    faces = faces[valid[faces].all(axis=1)]
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[valid] = np.arange(valid.sum())
    points, colors, faces = points[valid], colors[valid], remap[faces]

    if config.center and len(points):
        points = points - (points.min(0) + points.max(0)) / 2.0

    return Scene(points.astype(np.float32), colors.astype(np.uint8), faces)
