"""Render one debug image of the pipeline state: raw frames with ROI, crops with landmarks.

Stopgap until the preview window exists; `cymple-linux run --debug-image out.png` refreshes it
once per second.
"""
import cv2
import numpy as np

from . import imgproc
from . import protocol as P

H = 176
LANDMARK_COLORS = [(0, 200, 255)] * 2 + [(0, 255, 0)] * 2 + [(255, 160, 0)] * 2 + [(0, 0, 255), (255, 0, 255)]


def _quad_px(quad, w, h):
    q = np.asarray(quad, np.float32) * np.float32([w - 1, h - 1])
    return q[[0, 2, 3, 1]].astype(np.int32)          # LT, RT, RB, LB for drawing


def _fit(img, h=H):
    if img is None:
        return np.zeros((h, h, 3), np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (max(1, round(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_NEAREST)


def _label(img, text, y=14):
    cv2.putText(img, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def eye_panels(frame, obs, quad, name):
    raw = _fit(frame)
    if frame is not None and quad is not None:
        cv2.polylines(raw, [_quad_px(quad, raw.shape[1], raw.shape[0])], True, (0, 255, 255), 1)
    _label(raw, "%s eye raw" % name)
    crop = _fit(obs.crop if obs is not None else None)
    if obs is not None:
        for (x, y), c in zip(obs.landmarks, LANDMARK_COLORS):
            cv2.circle(crop, (int(x * (H - 1)), int(y * (H - 1))), 3, c, -1)
        _label(crop, "crop roi=%s" % ("yes" if obs.roi_used else "no"))
    return [raw, crop]


def render(pipeline):
    snap = pipeline.snapshot()
    rows = []
    for stream, eye in P.EYE_STREAMS.items():
        w = pipeline.workers.get(stream)
        if w is None:
            continue
        rows.append(eye_panels(w.last_frame, snap["eyes"][eye], w.tracker.roi_quad, w.name.split("-")[1]))
    mw = pipeline.workers.get(P.MOUTH_V1_STREAM)
    if mw is not None:
        m = snap["mouth"]
        rotated = None if mw.last_frame is None else imgproc.rotate_frame(mw.last_frame, pipeline.calib.mouth.rotation)
        raw = _fit(rotated)
        if m is not None and m.quad is not None:
            cv2.polylines(raw, [_quad_px(m.quad, raw.shape[1], raw.shape[0])], True, (0, 255, 255), 1)
        elif m is not None and m.rect is not None and rotated is not None:
            s = H / rotated.shape[0]
            x, y, ww, hh = m.rect
            cv2.rectangle(raw, (int(x * s), int(y * s)), (int((x + ww) * s), int((y + hh) * s)), (0, 255, 255), 1)
        _label(raw, "mouth raw (rotated)")
        crop = _fit(m.crop if m is not None else None)
        _label(crop, "mouth crop")
        rows.append([raw, crop])
    if not rows:
        return np.zeros((H, H, 3), np.uint8)
    rows = [np.hstack(r) for r in rows]
    width = max(r.shape[1] for r in rows)
    img = np.vstack([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows])
    g = snap["gaze"]
    info = np.zeros((20, width, 3), np.uint8)
    if g:
        _label(info, "yaw L %+.2f R %+.2f  pitch %+.2f" % (g["EyeYaw_L"], g["EyeYaw_R"], g["EyePitch_L"]))
    return np.vstack([img, info])
