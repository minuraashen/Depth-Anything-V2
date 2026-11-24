"""
Born out of Depth Anything V1 Issue 36
Make sure you have the necessary libraries installed.
Code by @1ssb

This script processes a set of images to generate depth maps and corresponding point clouds.
The resulting point clouds are saved in the specified output directory.

Usage:
    python script.py --encoder vitl --load-from path_to_model --max-depth 20 --img-path path_to_images --outdir output_directory --focal-length-x 470.4 --focal-length-y 470.4

Arguments:
    --encoder: Model encoder to use. Choices are ['vits', 'vitb', 'vitl', 'vitg'].
    --load-from: Path to the pre-trained model weights.
    --max-depth: Maximum depth value for the depth map.
    --img-path: Path to the input image or directory containing images.
    --outdir: Directory to save the output point clouds.
    --focal-length-x: Focal length along the x-axis.
    --focal-length-y: Focal length along the y-axis.
"""

import argparse
import cv2
import glob
import os
import sys
import torch

from depth_anything_v2.dpt import DepthAnythingV2

# Shared 3D reconstruction module lives at the repository root.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from reconstruction import ReconstructionConfig, reconstruct  # noqa: E402


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate depth maps and point clouds from images.')
    parser.add_argument('--encoder', default='vitl', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder to use.')
    parser.add_argument('--load-from', default='', type=str, required=True,
                        help='Path to the pre-trained model weights.')
    parser.add_argument('--max-depth', default=20, type=float,
                        help='Maximum depth value for the depth map.')
    parser.add_argument('--img-path', type=str, required=True,
                        help='Path to the input image or directory containing images.')
    parser.add_argument('--outdir', type=str, default='./vis_pointcloud',
                        help='Directory to save the output point clouds.')
    parser.add_argument('--focal-length-x', default=470.4, type=float,
                        help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', default=470.4, type=float,
                        help='Focal length along the y-axis.')
    parser.add_argument('--edge-threshold', default=0.04, type=float,
                        help='Relative depth jump treated as an object edge (removes flying pixels). 0 disables.')
    parser.add_argument('--keep-background', action='store_true',
                        help='Keep pixels saturated at --max-depth (usually sky).')
    parser.add_argument('--mesh', action='store_true',
                        help='Also save a coloured triangle mesh (.glb).')

    args = parser.parse_args()

    # Determine the device to use (CUDA, MPS, or CPU)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Model configuration based on the chosen encoder
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Initialize the DepthAnythingV2 model with the specified configuration
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Get the list of image files to process
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    # Create the output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    # Process each image file
    for k, filename in enumerate(filenames):
        print(f'Processing {k+1}/{len(filenames)}: {filename}')

        image = cv2.imread(filename)
        if image is None:
            print(f'  skipped (not an image)')
            continue
        pred = depth_anything.infer_image(image, image.shape[0])

        scene = reconstruct(image[:, :, ::-1], pred, ReconstructionConfig(
            depth_type='metric',
            focal_length_x=args.focal_length_x,
            focal_length_y=args.focal_length_y,
            max_depth=args.max_depth,
            edge_threshold=args.edge_threshold,
            remove_background=not args.keep_background,
            max_side=None,              # full resolution
            coordinate_frame='opencv',  # x right, y down, z forward, in metres
        ))

        stem = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0])
        scene.save_ply(stem + '.ply')
        if args.mesh:
            scene.save_glb(stem + '.glb')


if __name__ == '__main__':
    main()
