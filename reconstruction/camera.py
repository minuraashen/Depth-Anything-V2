"""Pinhole camera model."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width, height, fov_deg):
        """Intrinsics from an assumed horizontal field of view (square pixels, centred principal point)."""
        f = (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
        return cls(f, f, (width - 1) / 2.0, (height - 1) / 2.0)

    @classmethod
    def from_focal(cls, width, height, fx, fy=None):
        """Intrinsics from a known focal length in pixels."""
        return cls(fx, fy if fy is not None else fx, (width - 1) / 2.0, (height - 1) / 2.0)

    def scaled(self, sx, sy):
        """Intrinsics of the same camera after resizing the image by (sx, sy)."""
        return Intrinsics(self.fx * sx, self.fy * sy, (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5)

    def unproject(self, depth):
        """Back-project a depth map (H, W) to camera-space points (H, W, 3).

        OpenCV axes: x right, y down, z forward.
        """
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        return np.stack((x, y, depth), axis=-1)
