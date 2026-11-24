"""Single-image 3D reconstruction from Depth Anything V2 predictions.

One implementation shared by the Gradio demo (relative depth) and
``metric_depth/depth_to_pointcloud.py`` (metric depth):

    from reconstruction import ReconstructionConfig, reconstruct
    scene = reconstruct(rgb, prediction, ReconstructionConfig(depth_type='relative'))
    scene.save_ply('scene.ply')
    scene.save_glb('scene.glb')
"""
from .camera import Intrinsics
from .config import ReconstructionConfig
from .depth import depth_edge_mask, prediction_to_depth
from .pipeline import reconstruct
from .scene import Scene

__all__ = ['Intrinsics', 'ReconstructionConfig', 'prediction_to_depth', 'depth_edge_mask', 'reconstruct', 'Scene']
