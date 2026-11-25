# =============================================================================
# run.py  —  ENTRY POINT for relative depth estimation on images
# -----------------------------------------------------------------------------
# What it does (end to end):
#   1. Builds a Depth Anything V2 model (DINOv2 encoder + DPT decoder) of the
#      chosen size (Small / Base / Large / Giant).
#   2. Loads the released pretrained weights from ./checkpoints/.
#   3. For every input image: runs the model -> gets a depth map ->
#      normalises it to 0..255 -> colours it -> saves a PNG.
#
# Example:
#   python run.py --encoder vits --img-path assets/examples --outdir depth_vis
#
# IMPORTANT: the model output is RELATIVE INVERSE DEPTH (disparity-like):
#   bigger value = CLOSER to the camera. It has no unit (not metres).
#   (For metric depth in metres, see metric_depth/run.py.)
# =============================================================================
import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2  # the full model (encoder + decoder)


if __name__ == '__main__':
    # ---------------- 1) Command-line arguments ----------------
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str)  # one image, a folder, or a .txt list of image paths
    parser.add_argument('--input-size', type=int, default=518)  # shorter side is resized to this (518 = training size).
                                                                # Larger values -> sharper maps ("test-time resolution scaling", paper App. B.8)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    # Which model size to use (paper: 25M -> 1.3B parameters)
    #   vits = ViT-Small (24.8M), vitb = ViT-Base (97.5M), vitl = ViT-Large (335.3M), vitg = ViT-Giant (1.3B)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    # Use the GPU if available (CUDA for NVIDIA, MPS for Apple Silicon), otherwise CPU
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # ---------------- 2) Model configurations ----------------
    # For each encoder size:
    #   'features'     = channel width used inside the DPT decoder (fusion blocks)
    #   'out_channels' = channels of the 4 feature maps taken from 4 different
    #                    ViT layers, after projection in the DPT "reassemble" step
    #                    (shallow layer -> deep layer)
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Build the network and load the pretrained weights (these are the STUDENT
    # models from the paper, trained only on the 62M pseudo-labelled real images)
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()  # .eval(): inference mode (no dropout / BN updates)
    
    # ---------------- 3) Collect input files ----------------
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:  # a text file with one image path per line
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]  # a single image
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)  # every file in the folder (recursively)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # 'Spectral_r' colour map: low values -> blue, high values -> red.
    # Since the output is inverse depth (high = near), NEAR objects look RED/warm and FAR objects BLUE.
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    # ---------------- 4) Inference loop ----------------
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)  # H x W x 3, uint8, BGR channel order (OpenCV default)
        
        # The core call: preprocess -> network forward -> resize back to H x W.
        # Returns a float numpy array of shape (H, W) = relative inverse depth.
        depth = depth_anything.infer_image(raw_image, args.input_size)
        
        # Min-max normalise to 0..255 for visualisation only.
        # (This is valid because the output is relative: only the ordering matters, not the scale.)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)  # copy the single channel to 3 channels
        else:
            # cmap() returns RGBA in 0..1 -> keep RGB, scale to 0..255, flip RGB -> BGR for OpenCV
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        else:
            # Save "input | white gap | depth map" side by side
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)