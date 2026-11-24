"""Reconstructed scene container and exporters."""
from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class Scene:
    points: np.ndarray  # (N, 3) float32
    colors: np.ndarray  # (N, 3) uint8 RGB
    faces: np.ndarray   # (M, 3) int64 indices into points

    def __len__(self):
        return len(self.points)

    def to_point_cloud(self):
        return trimesh.PointCloud(self.points, colors=self.colors)

    def to_mesh(self):
        return trimesh.Trimesh(vertices=self.points, faces=self.faces, vertex_colors=self.colors, process=False)

    def save_ply(self, path):
        """Coloured point cloud (MeshLab, CloudCompare, Open3D, Blender)."""
        self.to_point_cloud().export(path, file_type='ply')
        return path

    def save_glb(self, path):
        """Coloured triangle mesh (web viewers, Blender, Windows 3D Viewer)."""
        self.to_mesh().export(path, file_type='glb')
        return path
