import glob
import tempfile

import gradio as gr
import matplotlib
import numpy as np
import torch
from PIL import Image

from depth_anything_v2.dpt import DepthAnythingV2
from reconstruction import ReconstructionConfig, reconstruct

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

cmap = matplotlib.colormaps.get_cmap('Spectral_r')


def tmp_path(suffix):
    return tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name


def predict_depth(image):
    return model.infer_image(image)


def on_submit(image):
    if image is None:
        raise gr.Error("Please upload an image first.")

    depth = predict_depth(image[:, :, ::-1])

    raw_depth_path = tmp_path('.png')
    Image.fromarray(depth.astype('uint16')).save(raw_depth_path)

    depth_u8 = ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0).astype(np.uint8)
    colored_depth = (cmap(depth_u8)[:, :, :3] * 255).astype(np.uint8)
    gray_depth_path = tmp_path('.png')
    Image.fromarray(depth_u8).save(gray_depth_path)

    return (image, colored_depth), gray_depth_path, raw_depth_path, (image, depth)


def on_reconstruct(state, view, fov_deg, near_far_ratio, edge_threshold, remove_background):
    if state is None:
        return None, None, None
    image, depth = state
    config = ReconstructionConfig(
        depth_type='relative',
        fov_deg=fov_deg,
        near_far_ratio=near_far_ratio,
        edge_threshold=edge_threshold,
        remove_background=remove_background,
        center=True,
    )
    scene = reconstruct(image, depth, config)
    ply_path = scene.save_ply(tmp_path('.ply'))
    glb_path = scene.save_glb(tmp_path('.glb'))
    return (glb_path if view == 'Mesh' else ply_path), ply_path, glb_path


theme = gr.themes.Soft(
    primary_hue='indigo',
    secondary_hue='slate',
    neutral_hue='slate',
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
).set(block_radius='14px', button_large_radius='12px')

css = """
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
#header { text-align: center; padding: 18px 0 6px; }
#header h1 { font-size: 2.1rem; font-weight: 700; margin: 0; }
#header p { color: var(--body-text-color-subdued); margin: 6px 0 12px; }
#header .links a {
    display: inline-block; margin: 0 4px; padding: 4px 14px; border-radius: 999px;
    border: 1px solid var(--border-color-primary); text-decoration: none; font-size: 0.9rem;
}
#header .links a:hover { background: var(--background-fill-secondary); }
.panel { padding: 14px !important; border-radius: 16px !important; }
#run-btn { margin-top: 4px; }
.note { font-size: 0.85rem; color: var(--body-text-color-subdued); }
"""

header = """
<div id="header">
  <h1>Depth Anything V2</h1>
  <p>Monocular depth estimation and single-image 3D reconstruction</p>
  <div class="links">
    <a href="https://arxiv.org/abs/2406.09414" target="_blank">Paper</a>
    <a href="https://depth-anything-v2.github.io" target="_blank">Project page</a>
    <a href="https://github.com/DepthAnything/Depth-Anything-V2" target="_blank">GitHub</a>
  </div>
</div>
"""

with gr.Blocks(title="Depth Anything V2") as demo:
    gr.HTML(header)
    depth_state = gr.State()  # (rgb image, raw relative depth) of the last prediction

    with gr.Row(equal_height=False):
        # Left: input and controls
        with gr.Column(scale=5, elem_classes='panel', variant='panel'):
            input_image = gr.Image(label="Input image", type='numpy', height=380)
            submit = gr.Button("Estimate depth & reconstruct 3D", variant='primary', size='lg', elem_id='run-btn')

            with gr.Accordion("3D reconstruction settings", open=False):
                view_mode = gr.Radio(["Mesh", "Point cloud"], value="Mesh", label="3D viewer shows")
                fov = gr.Slider(30, 120, value=60, step=1, label="Camera horizontal field of view (°)",
                                info="Phone main cameras are roughly 65–75°.")
                near_far = gr.Slider(1.5, 50, value=8, step=0.5, label="Depth range (far / near)",
                                     info="Relative depth has no absolute scale; this sets how deep the scene is.")
                edge_thr = gr.Slider(0, 0.3, value=0.04, step=0.01, label="Edge cleanup",
                                     info="Removes streaks at object boundaries. 0 = off.")
                remove_bg = gr.Checkbox(value=True, label="Remove sky / far background")

            gr.Examples(examples=sorted(glob.glob('assets/examples/*')), inputs=[input_image],
                        examples_per_page=12, label="Examples")

        # Right: results
        with gr.Column(scale=7, elem_classes='panel', variant='panel'):
            with gr.Tabs():
                with gr.Tab("Depth map"):
                    depth_slider = gr.ImageSlider(label="Input ↔ depth (drag the slider)", type='numpy', height=520)
                with gr.Tab("3D scene"):
                    scene_3d = gr.Model3D(label="3D reconstruction", clear_color=[0, 0, 0, 0], height=520)
                    gr.Markdown("Drag to orbit, scroll to zoom, right-drag to pan. Geometry comes from "
                                "*relative* depth, so shapes are plausible but not to metric scale.",
                                elem_classes='note')
                with gr.Tab("Downloads"):
                    with gr.Row():
                        gray_depth_file = gr.File(label="Grayscale depth map (.png)")
                        raw_file = gr.File(label="16-bit raw output (disparity, .png)")
                    with gr.Row():
                        ply_file = gr.File(label="Point cloud (.ply)")
                        glb_file = gr.File(label="Mesh (.glb)")

    settings_3d = [depth_state, view_mode, fov, near_far, edge_thr, remove_bg]
    outputs_3d = [scene_3d, ply_file, glb_file]

    submit.click(on_submit, inputs=[input_image],
                 outputs=[depth_slider, gray_depth_file, raw_file, depth_state]).then(
        on_reconstruct, inputs=settings_3d, outputs=outputs_3d)

    # Changing a 3D setting rebuilds only the 3D scene; the network is not re-run.
    for control in [view_mode, remove_bg]:
        control.change(on_reconstruct, inputs=settings_3d, outputs=outputs_3d)
    for control in [fov, near_far, edge_thr]:
        control.release(on_reconstruct, inputs=settings_3d, outputs=outputs_3d)

if __name__ == '__main__':
    demo.queue().launch(theme=theme, css=css)
