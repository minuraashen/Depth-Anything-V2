"""Loading Depth Anything V2 relative-depth models."""
import torch

from depth_anything_v2.dpt import DepthAnythingV2

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
