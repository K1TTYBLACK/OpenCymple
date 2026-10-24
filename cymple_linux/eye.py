"""Eye pipeline: stored ROI quad -> 96x96 crop -> model2 landmarks + 3 expression channels.

Runtime path (cympleface.dll sub_18003f390):
  1. ROI quad for this eye from calibration (sub_18003fb60). Valid -> perspective warp to
     96x96 (sub_1800412a0); otherwise resize the whole frame to 96x96, INTER_CUBIC.
  2. Right eye (index 1): horizontal flip of the crop, so the model always sees a left eye.
  3. model2 -> landmark_output[16] (8 points, 0..1 in the crop), blendshape_output[3].
  4. Right eye: un-mirror (sub_1800409d0): x -> 1 - x for all points, swap pairs (0,1),
     (2,3), (4,5). pt6 / pt7 are single points and are not swapped.

The ROI model (model0) does not run per frame. It runs only for ROI calibration
(sub_180032370, "ROI calibration saved and will be reused at runtime"), see calibrate_roi().
"""
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import imgproc

LEFT, RIGHT = 0, 1
EYE_NAMES = ("left", "right")
LANDMARK_SWAPS = ((0, 1), (2, 3), (4, 5))
ROI_SWAPS = ((0, 2), (1, 3))


@dataclass
class EyeObservation:
    eye: int
    landmarks: np.ndarray        # (8, 2) float32, image-ordered, normalized in the 96x96 crop
    expression: np.ndarray       # (3,) raw model output: widen, brow_down, brow_in_raise [labels inferred]
    crop: np.ndarray             # 96x96 uint8 crop as seen before the mirror flip
    roi_used: bool
    t: float                     # time.monotonic() of the source frame
    seq: int = -1
    infer_ms: float = 0.0


def unmirror_points(pts, swaps):
    out = np.array(pts, np.float32, copy=True)
    out[:, 0] = 1.0 - out[:, 0]
    for a, b in swaps:
        out[[a, b]] = out[[b, a]]
    return out


def neutral_rescale(x, neutral):
    """Expression channel relative to its captured neutral (sub_1800cf960):
    clamp((clamp(x, 0, 1) - n) / (1 - n), 0, 1)."""
    x = min(max(float(x), 0.0), 1.0)
    d = 1.0 - neutral
    if not np.isfinite(x) or d <= 1e-6:
        return 0.0
    return min(max((x - neutral) / d, 0.0), 1.0)


class EyeTracker:
    def __init__(self, landmark_model, eye, roi_quad=None):
        self.model = landmark_model
        self.eye = eye
        self.set_roi(roi_quad)

    def set_roi(self, quad):
        self.roi_quad = np.asarray(quad, np.float32) if imgproc.quad_valid(quad) else None

    def crop(self, frame):
        if self.roi_quad is not None:
            return imgproc.warp_quad(frame, self.roi_quad), True
        return imgproc.resize_eye_fallback(frame), False

    def process(self, frame, t=None, seq=-1):
        crop, roi_used = self.crop(frame)
        model_in = cv2.flip(crop, 1) if self.eye == RIGHT else crop
        t0 = time.perf_counter()
        out = self.model.run(imgproc.to_tensor(model_in, imgproc.CROP))
        infer_ms = (time.perf_counter() - t0) * 1000.0
        lm = np.asarray(out["landmark_output"], np.float32).reshape(8, 2)
        expr = np.asarray(out["blendshape_output"], np.float32).reshape(-1)[:3]
        if self.eye == RIGHT:
            lm = unmirror_points(lm, LANDMARK_SWAPS)
        return EyeObservation(self.eye, lm, expr, crop, roi_used,
                              time.monotonic() if t is None else t, seq, infer_ms)


def calibrate_roi(roi_model, frame, eye):
    """One-shot ROI calibration on a raw frame (sub_180032370 -> sub_18004cfe0).
    Right eye: flip the frame, run model0 at 64x64, then un-mirror (x -> 1 - x, swap P0<->P2,
    P1<->P3). Returns a validated 4x2 quad or None."""
    src = cv2.flip(frame, 1) if eye == RIGHT else frame
    out = roi_model.run(imgproc.to_tensor(src, 64))
    quad = imgproc.points_from_output(next(iter(out.values())))
    if eye == RIGHT:
        quad = unmirror_points(quad, ROI_SWAPS)
    return quad if imgproc.quad_valid(quad) else None


def landmark_sanity(obs):
    """Cheap plausibility checks used by tests and the status line (not part of the original)."""
    lm = obs.landmarks
    upper_y = (lm[2, 1] + lm[3, 1]) / 2
    lower_y = (lm[4, 1] + lm[5, 1]) / 2
    return {
        "corner_span": float(np.hypot(*(lm[1] - lm[0]))),
        "lids_ordered": bool(lower_y > upper_y),
        "in_crop": bool(np.all((lm > -0.1) & (lm < 1.1))),
    }
