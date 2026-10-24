"""Mouth V1 pipeline (stream 2): rotate -> ROI (auto quad or fixed rectangle) -> model4.

cympleface.dll sub_180035f80, per frame:
  1. Rotate by [Cymple_Mouth] rotation (imgproc.rotate_frame).
  2. Auto ROI (v1_auto_roi_calibration, sub_1800368a0): at most once per 1000 ms, run model3
     on the rotated frame (96x96 recipe) -> 4 points. A valid quad is copied the first time and
     afterwards smoothed point by point: quad += 0.25 * (new - quad) (0.25 at 0x1801d1130).
  3. With a quad: perspective-crop to 96x96 (sub_180037cf0). Otherwise the fixed rectangle
     from upper / bottom / width / x_offset (sub_1800370c0).
  4. model4 (96x96 recipe, INTER_AREA) -> 30 raw values.

The mapping of the 30 outputs to Unified Expressions is not reverse-engineered yet; this
module reports them raw.
"""
import time
from dataclasses import dataclass

import numpy as np

from . import imgproc

ROI_INTERVAL_MS = 1000
ROI_SMOOTHING = 0.25


@dataclass
class MouthObservation:
    values: np.ndarray           # (30,) raw model4 output
    crop: np.ndarray             # model input before resizing (96x96 for the quad path)
    quad: np.ndarray             # current auto-ROI quad on the rotated frame, or None
    rect: tuple                  # fixed rectangle (x, y, w, h) when it was used, else None
    t: float
    seq: int = -1
    infer_ms: float = 0.0


class MouthV1Tracker:
    def __init__(self, blend_model, roi_model, calib):
        self.model, self.roi_model, self.calib = blend_model, roi_model, calib
        self.quad = None
        self.last_roi_ms = None
        self.roi_runs = 0
        self.rotation = calib.rotation

    def update_roi(self, rotated, now_ms):
        """sub_1800368a0. The timestamp is taken before inference, success or not."""
        if not self.calib.auto_roi or self.roi_model is None:
            return
        if self.last_roi_ms is not None and now_ms - self.last_roi_ms < ROI_INTERVAL_MS:
            return
        self.last_roi_ms = now_ms
        out = self.roi_model.run(imgproc.to_tensor(rotated, 96))
        self.roi_runs += 1
        quad = imgproc.points_from_output(next(iter(out.values())))
        if not imgproc.quad_valid(quad):
            return
        if self.quad is None:
            self.quad = quad
        else:
            self.quad = self.quad + (quad - self.quad) * np.float32(ROI_SMOOTHING)

    def process(self, frame, t=None, seq=-1, now_ms=None):
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        c = self.calib
        if (not c.auto_roi or c.rotation != self.rotation) and self.quad is not None:
            # Changed from the UI: a quad from before no longer fits; auto ROI starts over.
            self.quad, self.last_roi_ms = None, None
        self.rotation = c.rotation
        rotated = imgproc.rotate_frame(frame, c.rotation)
        self.update_roi(rotated, now_ms)
        crop, rect = None, None
        if self.quad is not None:
            crop = imgproc.warp_quad(rotated, self.quad)
        if crop is None or crop.size == 0:
            rect = imgproc.manual_mouth_rect(rotated.shape[1], rotated.shape[0], c.upper, c.bottom,
                                             c.width, c.x_offset)
            if rect is None:
                return None
            x, y, w, h = rect
            crop = rotated[y:y + h, x:x + w]
        t0 = time.perf_counter()
        out = self.model.run(imgproc.to_tensor(crop, 96))
        infer_ms = (time.perf_counter() - t0) * 1000.0
        values = np.asarray(next(iter(out.values())), np.float32).reshape(-1)
        return MouthObservation(values, crop, None if self.quad is None else self.quad.copy(), rect,
                                time.monotonic() if t is None else t, seq, infer_ms)
