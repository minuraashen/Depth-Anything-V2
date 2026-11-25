"""Frame-by-frame video depth estimation."""
import cv2
import numpy as np

from .visualize import colorize


def _fit(frame, max_side):
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def video_to_depth(model, video_path, output_path, input_size=518, max_side=1280, smoothing=0.2, progress=None,
                   metric=False):
    """Write a colourised depth video for `video_path` to `output_path` (.webm, VP8).

    Each frame's depth is relative, so normalising every frame by its own min/max makes
    colours flicker. Instead the display range is an exponential moving average of the
    per-frame 1st/99th percentiles (`smoothing` = weight of the newest frame).

    Args:
        progress: optional callable(done_frames, total_frames).
        metric: True if `model` outputs metric depth (larger = farther) instead of relative disparity.
    Returns:
        output_path
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f'cannot open video {video_path}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    writer, lo, hi, done = None, None, None, 0
    try:
        while True:
            ok, frame = capture.read()  # BGR
            if not ok:
                break
            frame = _fit(frame, max_side)
            depth = model.infer_image(frame, input_size)
            if metric:
                depth = -depth  # metric depth grows with distance; flip so near stays red as in relative mode

            f_lo, f_hi = np.percentile(depth, [1, 99])
            lo = f_lo if lo is None else (1 - smoothing) * lo + smoothing * f_lo
            hi = f_hi if hi is None else (1 - smoothing) * hi + smoothing * f_hi
            depth_bgr = colorize(depth, lo, hi)[:, :, ::-1]

            if writer is None:
                h, w = depth_bgr.shape[:2]
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'VP80'), fps, (w, h))
            writer.write(np.ascontiguousarray(depth_bgr))

            done += 1
            if progress is not None:
                progress(done, total)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if done == 0:
        raise ValueError('the video has no readable frames')
    return output_path
