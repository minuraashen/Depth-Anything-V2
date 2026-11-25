"""Loading Depth Anything V2 relative and metric depth models."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch

from depth_anything_v2.dpt import DepthAnythingV2
# The metric model is a separate copy of the code (Sigmoid x max_depth head); importing it through
# the `metric_depth` namespace package keeps it apart from the relative `depth_anything_v2` package.
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as MetricDepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}


def load_model(encoder='vits', checkpoint_dir='checkpoints'):
    """Build the model and load `checkpoints/depth_anything_v2_<encoder>.pth`."""
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    state_dict = torch.load(f'{checkpoint_dir}/depth_anything_v2_{encoder}.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()


@dataclass(frozen=True)
class DepthModelSpec:
    label: str
    checkpoint: str
    metric: bool                       # False: relative disparity (larger = closer); True: depth in metres
    max_depth: Optional[float] = None  # metric only: the model's output range in metres


# The three selectable models (all ViT-S checkpoints).
DEPTH_MODELS = {
    'relative': DepthModelSpec('Relative', 'checkpoints/depth_anything_v2_vits.pth', metric=False),
    'indoor': DepthModelSpec('Metric · Indoor (Hypersim, ≤ 20 m)',
                             'metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth',
                             metric=True, max_depth=20.0),
    'outdoor': DepthModelSpec('Metric · Outdoor (Virtual KITTI, ≤ 80 m)',
                              'metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vits.pth',
                              metric=True, max_depth=80.0),
}


@lru_cache(maxsize=None)
def get_depth_model(key, encoder='vits'):
    """Load a model from DEPTH_MODELS on first use and keep it cached."""
    spec = DEPTH_MODELS[key]
    if spec.metric:
        model = MetricDepthAnythingV2(**MODEL_CONFIGS[encoder], max_depth=spec.max_depth)
    else:
        model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    model.load_state_dict(torch.load(spec.checkpoint, map_location='cpu'))
    return model.to(DEVICE).eval()
