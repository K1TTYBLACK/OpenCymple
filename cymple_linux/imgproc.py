"""Image steps shared by the eye and mouth pipelines, matching cympleface.dll call for call.

Every model input goes through the same recipe (sub_18004dc50 eye ROI, sub_180040b80 eye
landmark, sub_18004fba0 mouth ROI, sub_180060150 MouthV1):
  1. cvtColor(BGR2GRAY) if the input is not single-channel (frames already are)
  2. resize to N x N, INTER_AREA
  3. x = (pixel - 127.5f) * 0.00784314f
  4. replicate the gray plane into 3 channels -> float32 NCHW [1, 3, N, N]
"""
import math

import cv2
import numpy as np

# 240x176 frames and 96x96 crops are too small for OpenCV's thread pool: with it, three
# streams burned ~1.2 cores; without it, ~0.27 (measured on the replayed live capture).
cv2.setNumThreads(1)

NORM_OFFSET = np.float32(127.5)          # 0x1801d18d4
NORM_SCALE = np.float32(0.00784314)      # 0x1801d18d0
CROP = 96
CROP_DST = np.float32([[0, 0], [0, CROP - 1], [CROP - 1, 0], [CROP - 1, CROP - 1]])
QUAD_MIN_EDGE = 0.02                     # sub_18003d570 / sub_180037ab0


def decode_gray(jpeg):
    """JPEG -> 8-bit gray. Equivalent to tj3Decompress8(TJPF_GRAY): both keep the Y plane."""
    buf = np.frombuffer(jpeg, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape[0] > 0x2000 or img.shape[1] > 0x2000:
        return None
    return img


def to_tensor(img, size):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), fx=0, fy=0, interpolation=cv2.INTER_AREA)
    x = (img.astype(np.float32) - NORM_OFFSET) * NORM_SCALE
    return np.ascontiguousarray(np.broadcast_to(x, (1, 3, size, size)), dtype=np.float32)


def points_from_output(values):
    """8 floats -> 4 (x, y) points, P_i = (o[2i], o[2i+1]), normalized to the input frame."""
    return np.asarray(values, np.float32).reshape(-1)[:8].reshape(4, 2)


def quad_valid(q):
    """All coordinates finite and the four quad edges (0-1, 2-3, 0-2, 1-3) longer than 0.02."""
    if q is None:
        return False
    q = np.asarray(q, np.float32)
    if q.shape != (4, 2) or not np.all(np.isfinite(q)):
        return False
    return all(float(np.hypot(*(q[a] - q[b]))) > QUAD_MIN_EDGE for a, b in ((1, 0), (3, 2), (2, 0), (3, 1)))


def warp_quad(img, quad):
    """Perspective-crop a normalized quad to 96x96 (sub_1800412a0 eye, sub_180037cf0 mouth).
    Point order: P0 -> (0,0), P1 -> (0,95), P2 -> (95,0), P3 -> (95,95), i.e. LeftTop,
    LeftBottom, RightTop, RightBottom of the crop."""
    rows, cols = img.shape[:2]
    q = np.asarray(quad, np.float32)
    src = np.stack([q[:, 0] * np.float32(cols - 1), q[:, 1] * np.float32(rows - 1)], axis=1)
    m = cv2.getPerspectiveTransform(src, CROP_DST)
    return cv2.warpPerspective(img, m, (CROP, CROP), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def resize_eye_fallback(img):
    """No valid ROI quad: whole frame to 96x96 with INTER_CUBIC (sub_18003fa60)."""
    return cv2.resize(img, (CROP, CROP), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)


def rotate_frame(img, angle_deg):
    """Mouth V1 frame rotation (sub_180035f80): skipped when |angle| <= 1e-4, otherwise
    warpAffine about (cols/2, rows/2) (integer halves), scale 1, same size, INTER_LINEAR,
    constant 0 border."""
    angle = float(np.float32(angle_deg))
    if abs(angle) <= 1e-4:
        return img
    rows, cols = img.shape[:2]
    m = cv2.getRotationMatrix2D((float(int(cols / 2)), float(int(rows / 2))), angle, 1.0)
    return cv2.warpAffine(img, m, (cols, rows), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def manual_mouth_rect(cols, rows, upper, bottom, width, x_offset):
    """Fixed mouth crop rectangle (x, y, w, h) from the [Cymple_Mouth] sliders (sub_1800370c0).
    Returns None for an empty frame. All math is in double, as in the original."""
    if cols <= 0 or rows <= 0:
        return None
    upper, bottom, width, x_offset = (float(np.float32(v)) for v in (upper, bottom, width, x_offset))
    min_w, min_h = 1.0 / cols, 1.0 / rows
    w = _clamp(width, min_w, 1.0)
    top = _clamp(upper, 0.0, 1.0 - min_h)
    bot = _clamp(bottom, top + min_h, 1.0)
    half = 0.5 * w
    cx = _clamp(x_offset, half, 1.0 - half)
    x0 = _clamp(int(math.floor((cx - half) * cols)), 0, cols - 1)
    y0 = _clamp(int(math.floor(top * rows)), 0, rows - 1)
    x1 = _clamp(int(math.ceil((cx + half) * cols)), x0 + 1, cols)
    y1 = _clamp(int(math.ceil(bot * rows)), y0 + 1, rows)
    return x0, y0, x1 - x0, y1 - y0
