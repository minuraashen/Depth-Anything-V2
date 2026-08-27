import glob
import gradio as gr
import matplotlib
import numpy as np
from PIL import Image
import torch
import tempfile
import trimesh

from depth_anything_v2.dpt import DepthAnythingV2

css = """
#img-display-container {
    max-height: 100vh;
}
#img-display-input {
    max-height: 80vh;
}
#img-display-output {
    max-height: 80vh;
}
#download {
    height: 62px;
}
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
encoder = 'vits'
model = DepthAnythingV2(**model_configs[encoder])
state_dict = torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location="cpu")
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()

title = "# Depth Anything V2"
description = """Official demo for **Depth Anything V2**.
Please refer to our [paper](https://arxiv.org/abs/2406.09414), [project page](https://depth-anything-v2.github.io), or [github](https://github.com/DepthAnything/Depth-Anything-V2) for more details."""

def predict_depth(image):
    return model.infer_image(image)

def generate_3d_point_cloud(image, depth):
    h, w = depth.shape
    
    # 1. Approximate camera intrinsics
    fx, fy = w, w
    cx, cy = w / 2.0, h / 2.0
    
    # 2. Normalize and Invert Depth
    # Depth Anything outputs relative inverse depth (larger = closer). 
    # We must convert this back to spatial Z coordinates.
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    Z = 1.0 / (depth_norm + 0.1) # Small epsilon prevents division by zero
    
    # 3. Create a grid of 2D pixel coordinates (u, v)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # 4. Unproject 2D to 3D Space using linear algebra
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    # Stack and flatten into a list of 3D points. 
    # We invert Y and Z so the model doesn't render upside down.
    points = np.stack((X, -Y, -Z), axis=-1).reshape(-1, 3)
    colors = image.reshape(-1, 3)
    
    # 5. Subsample points to prevent the browser from crashing
    max_points = 150000
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
        
    # 6. Generate the Point Cloud and save as a .ply file
    cloud = trimesh.PointCloud(points, colors=colors)
    temp_ply = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
    cloud.export(temp_ply.name, file_type='ply')
    
    return temp_ply.name

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown("### Depth Prediction demo")

    # Row 1: The 2D Images
    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
        depth_image = gr.Image(label="Depth Map", type='numpy', elem_id='img-display-output')
    
    # Row 2: The 3D Viewer gets its own wide space
    with gr.Row():
        scene_3d = gr.Model3D(label="Interactive 3D Point Cloud", clear_color=[0, 0, 0, 0])
    
    # Buttons and Files also belong inside the Blocks context
    submit = gr.Button(value="Compute Depth")
    gray_depth_file = gr.File(label="Grayscale depth map", elem_id="download")
    raw_file = gr.File(label="16-bit raw output (can be considered as disparity)", elem_id="download")

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    def on_submit(image):
        original_image = image.copy()
        h, w = image.shape[:2]

        depth = predict_depth(image[:, :, ::-1])

        # Generate 3D point cloud
        point_cloud_path = generate_3d_point_cloud(image, depth)

        raw_depth = Image.fromarray(depth.astype('uint16'))
        tmp_raw_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        raw_depth.save(tmp_raw_depth.name)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)

        gray_depth = Image.fromarray(depth)
        tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        gray_depth.save(tmp_gray_depth.name)

        return [colored_depth, tmp_gray_depth.name, tmp_raw_depth.name, point_cloud_path]

    # The click event and Examples must also be indented inside the Blocks context
    submit.click(on_submit, inputs=[input_image], outputs=[depth_image, gray_depth_file, raw_file, scene_3d])

    example_files = glob.glob('assets/examples/*')
    examples = gr.Examples(examples=example_files, inputs=[input_image], outputs=[depth_image, gray_depth_file, raw_file, scene_3d], fn=on_submit)

if __name__ == '__main__':
    demo.queue().launch()