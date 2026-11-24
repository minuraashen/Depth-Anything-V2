"""Reconstruction settings."""
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ReconstructionConfig:
    # 'relative': affine-invariant disparity from the relative models (larger = closer).
    # 'metric':   depth in metres from the metric_depth models.
    depth_type: Literal['relative', 'metric'] = 'relative'

    # Camera. A known focal length (pixels, at the input resolution) overrides the FOV guess.
    fov_deg: float = 60.0
    focal_length_x: Optional[float] = None
    focal_length_y: Optional[float] = None

    # Relative depth only: farthest / nearest depth. Fixes the unknown disparity shift.
    near_far_ratio: float = 8.0

    # Drop pixels whose 3x3 neighbourhood spans more than this relative depth jump
    # (removes "flying pixels" streaking between foreground and background). 0 disables.
    edge_threshold: float = 0.04

    # Drop sky / infinitely far pixels.
    remove_background: bool = True
    # Relative: normalised disparity at or below this is background.
    background_disparity: float = 0.02
    # Metric: depth at or above this fraction of the model's max depth is background.
    max_depth: Optional[float] = None
    background_depth_fraction: float = 0.98

    # Resample so the longest side is at most this many pixels (None = full resolution).
    max_side: Optional[int] = 640

    # 'opengl': x right, y up, camera looks down -z (glTF / web viewers).
    # 'opencv': x right, y down, z forward.
    coordinate_frame: Literal['opengl', 'opencv'] = 'opengl'

    # Move the bounding-box centre to the origin (good for orbit viewers; loses the camera origin).
    center: bool = False
