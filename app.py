import glob
import tempfile

import gradio as gr
import numpy as np

from depth_tools import DEPTH_MODELS, colorize, get_depth_model, video_to_depth
from reconstruction import ReconstructionConfig, reconstruct

get_depth_model('relative')  # load the default model at start-up; metric models load on first use

MODEL_CHOICES = [(spec.label, key) for key, spec in DEPTH_MODELS.items()]


def tmp_path(suffix):
    return tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name


def model_selector():
    return gr.Radio(MODEL_CHOICES, value='relative', label="Depth model",
                    info="Relative: shape only, no units. Metric: distance in metres; pick indoor or outdoor to match the scene.")


# ---------------------------------------------------------------- image
def on_image(image, model_key):
    if image is None:
        raise gr.Error("Please upload an image first.")
    spec = DEPTH_MODELS[model_key]
    depth = get_depth_model(model_key).infer_image(image[:, :, ::-1])  # model expects BGR

    if spec.metric:
        colored = colorize(-depth)  # metric depth grows with distance; flip so near stays red
        info = (f"**{spec.label}** · nearest {depth.min():.2f} m · median {float(np.median(depth)):.2f} m · "
                f"farthest {depth.max():.2f} m")
    else:
        colored = colorize(depth)
        info = f"**{spec.label}** · relative inverse depth (no units); red = near, blue = far"
    return (image, colored), (image, depth, model_key), info


def on_reconstruct(state, view, fov_deg, near_far_ratio, edge_threshold, remove_background):
    if state is None:
        return None
    image, depth, model_key = state
    spec = DEPTH_MODELS[model_key]
    scene = reconstruct(image, depth, ReconstructionConfig(
        depth_type='metric' if spec.metric else 'relative',
        max_depth=spec.max_depth,
        fov_deg=fov_deg,
        near_far_ratio=near_far_ratio,
        edge_threshold=edge_threshold,
        remove_background=remove_background,
        center=True,
    ))
    return scene.save_glb(tmp_path('.glb')) if view == 'Mesh' else scene.save_ply(tmp_path('.ply'))


# ---------------------------------------------------------------- video
def on_video(video_path, model_key, progress=gr.Progress()):
    if video_path is None:
        raise gr.Error("Please upload a video first.")

    def report(done, total):
        if total:
            progress(done / total, desc=f"Frame {done}/{total}")
        else:
            progress(None, desc=f"Frame {done}")

    return video_to_depth(get_depth_model(model_key), video_path, tmp_path('.webm'), progress=report,
                          metric=DEPTH_MODELS[model_key].metric)


# ---------------------------------------------------------------- UI
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
#mode-tabs > .tab-wrapper button { font-size: 1rem; font-weight: 600; }
.panel { padding: 14px !important; border-radius: 16px !important; }
.note { font-size: 0.85rem; color: var(--body-text-color-subdued); }
"""

header = """
<div id="header">
  <h1>Depth Anything V2</h1>
  <p>Monocular depth estimation for images and videos, with single-image 3D reconstruction</p>
  <div class="links">
    <a href="https://arxiv.org/abs/2406.09414" target="_blank">Paper</a>
    <a href="https://depth-anything-v2.github.io" target="_blank">Project page</a>
    <a href="https://github.com/DepthAnything/Depth-Anything-V2" target="_blank">GitHub</a>
  </div>
</div>
"""

with gr.Blocks(title="Depth Anything V2") as demo:
    gr.HTML(header)

    with gr.Tabs(elem_id='mode-tabs'):
        # ============================ Image ============================
        with gr.Tab("Image"):
            depth_state = gr.State()  # (rgb image, raw prediction, model key) of the last prediction

            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes='panel', variant='panel'):
                    input_image = gr.Image(label="Input image", type='numpy', height=380)
                    image_model = model_selector()
                    image_btn = gr.Button("Estimate depth & reconstruct 3D", variant='primary', size='lg')

                    with gr.Accordion("3D reconstruction settings", open=False):
                        view_mode = gr.Radio(["Mesh", "Point cloud"], value="Mesh", label="3D viewer shows")
                        fov = gr.Slider(30, 120, value=60, step=1, label="Camera horizontal field of view (°)",
                                        info="Phone main cameras are roughly 65–75°.")
                        near_far = gr.Slider(1.5, 50, value=8, step=0.5, label="Depth range (far / near)",
                                             info="Relative model only (metric depth already has a scale): "
                                                  "sets how deep the scene is.")
                        edge_thr = gr.Slider(0, 0.3, value=0.04, step=0.01, label="Edge cleanup",
                                             info="Removes streaks at object boundaries. 0 = off.")
                        remove_bg = gr.Checkbox(value=True, label="Remove sky / far background")

                    gr.Examples(examples=sorted(glob.glob('assets/examples/*')), inputs=[input_image],
                                examples_per_page=12, label="Examples")

                with gr.Column(scale=7, elem_classes='panel', variant='panel'):
                    with gr.Tabs():
                        with gr.Tab("Depth map"):
                            depth_slider = gr.ImageSlider(label="Input ↔ depth (drag the slider)",
                                                          type='numpy', height=520)
                            depth_info = gr.Markdown(elem_classes='note')
                        with gr.Tab("3D scene"):
                            scene_3d = gr.Model3D(label="3D reconstruction", clear_color=[0, 0, 0, 0], height=520)
                            gr.Markdown("Drag to orbit, scroll to zoom, right-drag to pan. With the relative model "
                                        "shapes are plausible but not to scale; metric models give a scene in metres "
                                        "(as accurate as the field-of-view setting).",
                                        elem_classes='note')

            settings_3d = [depth_state, view_mode, fov, near_far, edge_thr, remove_bg]
            image_btn.click(on_image, inputs=[input_image, image_model],
                            outputs=[depth_slider, depth_state, depth_info]).then(
                on_reconstruct, inputs=settings_3d, outputs=[scene_3d])
            # Changing a 3D setting rebuilds only the 3D scene; the network is not re-run.
            for control in [view_mode, remove_bg]:
                control.change(on_reconstruct, inputs=settings_3d, outputs=[scene_3d])
            for control in [fov, near_far, edge_thr]:
                control.release(on_reconstruct, inputs=settings_3d, outputs=[scene_3d])

        # ============================ Video ============================
        with gr.Tab("Video"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes='panel', variant='panel'):
                    input_video = gr.Video(label="Input video", height=380)
                    video_model = model_selector()
                    video_btn = gr.Button("Estimate video depth", variant='primary', size='lg')
                    gr.Markdown("Every frame goes through the network, so long or high-resolution clips take "
                                "a while (much longer on CPU). Frames are processed at up to 1280 px.",
                                elem_classes='note')
                    gr.Examples(examples=sorted(glob.glob('assets/examples_video/*')), inputs=[input_video],
                                label="Examples")

                with gr.Column(scale=7, elem_classes='panel', variant='panel'):
                    output_video = gr.Video(label="Depth video", height=520, autoplay=True, loop=True)

            video_btn.click(on_video, inputs=[input_video, video_model], outputs=[output_video])

if __name__ == '__main__':
    demo.queue().launch(theme=theme, css=css)
