"""Cymple's eye calibration flows (cympleface.dll faceOscClass + EyeGazeCalibrationController).

Function names in comments are the addresses each part mirrors. All flows work on the live
EyeCore state (the latest landmarks per eye, pt7 already low-passed) and write Cymple's INI keys
through an IniWriter, so a calibration made here also works in Windows Cymple after an export.

Recenter (eyeRecenterSlot sub_1800c1220; per eye the body of sub_1800b8690): every eye that is
fresh, open (sub_1800c8a00) and has a valid feature and lid points gets, from its current landmarks:
  centre lid geometry and the left/centre/right lid references moved to it (sub_1800b9c10); lid-close
  references 0.8 o / k o / o, with o the lid opening at the pupil and k the eye-closing slider
  (sub_1800c9930); squint and wide references; reset lid state; the expression neutral
  (sub_1800b9460); and the gaze centre: precise mode with a valid mesh shifts all 13 raw mesh points
  by (feature - centre point) and makes the ranges the mesh's bounding box, otherwise each range keeps
  its spans and moves its mid to the feature (sub_1800c5370).
Coarse range (sub_1800c01c0, precise mode off): ranges collapse to their mid, grow with every open-eye
feature (sub_1800c52a0, in sub_1800c0950) and after 5 s keep the old mid and each new extreme only if
it reaches more than 0.08 past it (sub_1800c4f10).
13-point mesh, Cymple's "full-view" calibration (sub_18017efc0, sub_18017dd40): one capture per
confirm, points 0..12 in order, raw = each eye's current mode-2 feature (sub_1800ba3a0); point 0 also
runs the centre references without the gaze part, points 9 and 5 the left and right lid references
(sub_1800ba100). An eye with all 13 points gets its mesh saved (feature mode 2) and its ranges
set to the mesh's bounding box.
"""
import logging
import math

from .calibration import LID_KEYS, LID_REF_POSITIONS, SIDES
from .face import EYE_SEEN_S, lid_at_pupil, lid_points, lids_valid, pupil_weights
from .gaze import blend_reference, clamp, corners_valid, feature_mode0, feature_mode2, finite

log = logging.getLogger(__name__)

EPS = 1e-5                     # 0x1801da634
OPEN_FRACTION = 0.05           # 0x1801d0ffc: open if mean opening > 0.05 x corner distance
RECENTER_HALF = (0.08, 0.035)  # 0x1801dc888 / 0x1801dc884: minimum half-width, yaw / pitch
COARSE_HALF = 0.08             # 0x1801dc888, both axes
COARSE_DURATION_S = 5.0        # eye page: "ends automatically after 5 seconds" [from the UI text]
NEUTRAL_MAX = 0.9999           # 0x1801dc8e4: channels at or above count as saturated
WIDE_REF_FACTOR = 1.2          # 0x1801dc940
MESH_LAYOUT_VERSION = 4
_D = math.sqrt(0.5)
# sub_180182ee0: centre, inner ring (r 0.45: up, left, down, right), outer ring (r 1, every 45 deg from
# the right, counter-clockwise). (target_yaw, target_pitch); your saved meshes have exactly these.
MESH_TARGETS = ((0.0, 0.0), (0.0, 0.45), (-0.45, 0.0), (0.0, -0.45), (0.45, 0.0),
                (1.0, 0.0), (_D, _D), (0.0, 1.0), (-_D, _D), (-1.0, 0.0), (-_D, -_D), (0.0, -1.0), (_D, -_D))
LID_REF_AT_POINT = {9: 0, 5: 2}  # mesh point -> lid reference position (left, right)


# ------------------------------------------------------------------ pure helpers
def eye_open(lm, lids):                      # sub_1800c8a00
    if not (lids_valid(lids) and corners_valid(lm)):
        return False
    opening = 0.5 * ((lids[2] - lids[0]) + (lids[3] - lids[1]))
    span = math.hypot(lm[0][0] - lm[1][0], lm[0][1] - lm[1][1])
    return opening > max(OPEN_FRACTION * span, EPS)


def gaze_feature(lm, cal, eye, force_mode2=False):   # sub_1800c7590
    """(yaw, pitch) or None. Precise mode off: sub_1800c6940, whose iris-circle feature (sub_1800c6700)
    needs a circle the EYE_V3 landmark path never sets [Inferred], so mode 0 (sub_1800c6350)."""
    if not force_mode2 and not cal.precise_gaze:
        return feature_mode0(lm)
    mode = 2 if force_mode2 else cal.eyes[eye].feature_mode
    f = feature_mode2(lm) if mode == 2 else None      # mode 1 (sub_1800c7210) is not ported
    return f if f is not None else feature_mode0(lm)


def landmarks_usable(lm, cal, eye):          # sub_1800c7650: feature and the four lid ys finite
    return gaze_feature(lm, cal, eye) is not None and finite(lm[2][1], lm[3][1], lm[4][1], lm[5][1])


def lid_values(opening, blink_amp):          # sub_1800c9930 -> (value0, value1, value2)
    o = max(opening, EPS)
    closed = clamp(blink_amp, 0.1, 0.3) * o   # sub_1800c5d40 id 2
    opened = 0.8 * o
    if closed >= opened:
        opened = closed + EPS
    return opened, closed, opened / 0.8


def remap_column(x, a0, a1, b0, b1):         # sub_1800c7f60: map x from span a0..a1 onto b0..b1
    sa, sb = max(a1 - a0, EPS), max(b1 - b0, EPS)
    return 0.5 * (b0 + b1) + (x - 0.5 * (a0 + a1)) * (sb / sa)


def remap_lids(rec, old, cur):               # sub_1800c7de0: outer column (uo, lo), inner column (ui, li)
    return (remap_column(rec[0], old[0], old[2], cur[0], cur[2]),
            remap_column(rec[1], old[1], old[3], cur[1], cur[3]),
            remap_column(rec[2], old[0], old[2], cur[0], cur[2]),
            remap_column(rec[3], old[1], old[3], cur[1], cur[3]))


def sanitize_range(r):                       # sub_1800c4af0 (+ sub_1800c4a50: (0, 0.5, 1) means unset)
    lo, mid, hi = r
    if all(finite(v) for v in r) and abs(lo) <= 1e-4 and abs(mid - 0.5) <= 1e-4 and abs(hi - 1.0) <= 1e-4:
        return (-1.0, 0.0, 1.0)
    if not finite(mid):
        mid = 0.0
    if not (finite(lo) and finite(hi)) or hi <= lo:
        lo, hi = -1.0, 1.0
    lo, mid, hi = (clamp(v, -1.0, 1.0) for v in (lo, mid, hi))
    if hi <= lo:
        return (-1.0, 0.0, 1.0)
    if not (lo < mid < hi):
        mid = 0.5 * (lo + hi)
    return (lo, mid, hi) if lo < mid < hi else (-1.0, 0.0, 1.0)


def recenter_range(r, value, min_half):      # sub_1800c5370
    lo, mid, hi = sanitize_range(r)
    m = max(min_half, EPS)
    below, above = max(mid - lo, m), max(hi - mid, m)
    mid = clamp(value, -0.99999, 0.99999)
    return sanitize_range((max(-1.0, mid - below), mid, min(1.0, mid + above)))


def ranges_from_mesh(mesh):                  # sub_1800c46b0: bounding box of the raw points, mid = centre
    ys, ps = [p[0] for p in mesh], [p[1] for p in mesh]
    return (sanitize_range((min(ys), mesh[0][0], max(ys))), sanitize_range((min(ps), mesh[0][1], max(ps))))


def collapse_range(r):                       # sub_1800c4e30 with the range's own mid
    lo, mid, hi = sanitize_range(r)
    mid = clamp(mid, -0.99999, 0.99999)
    return (mid, mid, mid)


def expand_range(r, v):                      # sub_1800c52a0
    if not finite(v):
        return r
    v = clamp(v, -1.0, 1.0)
    return (min(r[0], v), r[1], max(r[2], v))


def finalize_range(live, backup, min_half=COARSE_HALF):   # sub_1800c4f10 -> (range, changed)
    b = sanitize_range(backup)
    if not (finite(live[0]) and finite(live[2])):
        return b, False
    m = max(min_half, EPS)
    mid = b[1]
    ok_lo, ok_hi = mid - m > live[0], live[2] > mid + m
    lo = live[0] if ok_lo else b[0]
    hi = live[2] if ok_hi else b[2]
    if mid - lo < m:
        lo, ok_lo = b[0], False
    if hi - mid < m:
        hi, ok_hi = b[2], False
    out = sanitize_range((lo, mid, hi))
    if not (out[0] < out[1] < out[2]):
        return b, False
    return out, ok_lo or ok_hi


def neutral_update(expr, current):           # sub_1800cf890 + sub_1800d0200 -> (neutral, captured mask)
    out, mask = list(current), 0
    for i in range(3):
        x = float(expr[i])
        if finite(x) and 0.0 <= x < NEUTRAL_MAX:
            out[i] = x
            mask |= 1 << i
    return tuple(out), mask


def shift_mesh(mesh, dyaw, dpitch):          # sub_1800c4980
    return [(p[0] + dyaw, p[1] + dpitch, p[2], p[3]) for p in mesh]


# ------------------------------------------------------------------ INI keys (cympleFacelib)
def _f(v):
    return float(v)


def lid_geometry_keys(eye, lids):            # updateEyeLidGeometry
    return {"%s_Eye%s" % (SIDES[eye], k): _f(v) for k, v in zip(LID_KEYS, lids)}


def lid_ref_keys(eye, pos, yaw, lids):       # updateEyeLidGeometryAtGaze: "%1lid_ref_%2_%3"
    prefix = "%s_Eyelid_ref_%s_" % (SIDES[eye], LID_REF_POSITIONS[pos])
    out = {prefix + "yaw": _f(yaw)}
    out.update({prefix + k: _f(v) for k, v in zip(LID_KEYS, lids)})
    return out


def triple_keys(fmt, eye, values):           # updateEyeBlendshapeRange: <name><Side>value0..2
    return {fmt % (SIDES[eye], i): _f(v) for i, v in enumerate(values)}


def range_keys(eye, yaw, pitch):             # updateEyeYawRange / updateEyePitchRange
    out = {}
    for axis, r in (("x", yaw), ("y", pitch)):
        for k, v in zip(("min", "mid", "max"), r):
            out["%s_Eyepupil_%s_%s" % (SIDES[eye], axis, k)] = _f(v)
    return out


def neutral_keys(eye, neutral):              # updateEyeExpressionNeutral
    return {"%s_Eyeexpression_neutral_%s" % (SIDES[eye], k): _f(v)
            for k, v in zip(("widen", "brow_down", "brow_in_raise"), neutral)}


def mesh_keys(eye, mesh, feature_mode=2, overlay=None):   # updateEyeGazeMeshCalibration[WithOverlayGeometry]
    side = SIDES[eye] + "_Eyegaze_"
    out = {}
    for i, p in enumerate(mesh):
        for k, v in zip(("raw_yaw", "raw_pitch", "target_yaw", "target_pitch"), p):
            out["%smesh_%d_%s" % (side, i, k)] = _f(v)
    out[side + "mesh_valid"] = True
    out[side + "feature_mode"] = int(feature_mode)
    if overlay is not None:
        y_offset, ring_scale = overlay
        out.update({side + "mesh_layout_version": MESH_LAYOUT_VERSION,
                    side + "mesh_overlay_geometry_valid": True,
                    side + "mesh_overlay_geometry_layout_version": MESH_LAYOUT_VERSION,
                    side + "mesh_overlay_y_offset_meters": _f(y_offset),
                    side + "mesh_overlay_ring_scale": _f(ring_scale),
                    "eye_gaze_mesh_layout_version": MESH_LAYOUT_VERSION})
    return out


# ------------------------------------------------------------------ the flows
class EyeCalibrator:
    """Runs the flows on an EyeCore. Call every method with the pipeline lock held; each returns a
    short report and leaves the INI writes pending in `writer` (flush them afterwards)."""

    def __init__(self, core, writer):
        self.core, self.cal, self.writer = core, core.cal, writer
        self.coarse = None                   # {"until": t, "backup": [(yaw, pitch), (yaw, pitch)]}
        self.mesh = None                     # MeshCapture while the 13-point flow runs

    def _write(self, keys):
        for k, v in keys.items():
            self.writer.set("OSC", k, v)

    def _fresh(self, eye, now):
        c = self.core
        return c.lm[eye] is not None and now - c.seen[eye] <= EYE_SEEN_S

    def _check(self, eye, now):
        """(landmarks, lids) when the eye can be calibrated now, else a reason string."""
        if not self._fresh(eye, now):
            return "no fresh eye input"
        lm = self.core.lm[eye]
        lids = lid_points(lm, eye)
        if not eye_open(lm, lids):
            return "eye is closed or half closed"
        if not landmarks_usable(lm, self.cal, eye) or not lids_valid(lids):
            return "invalid corner/lid landmarks"
        return lm, lids

    # ---- pieces -------------------------------------------------------------------------
    def update_lid_refs(self, eye, lm, lids):          # sub_1800b9c10
        e = self.cal.eyes[eye]
        f = gaze_feature(lm, self.cal, eye, force_mode2=True)
        if f is None or not finite(f[0]):
            return
        yaw = f[0]
        old = e.lid_refs[1] if e.lid_refs else None
        old_valid = old is not None and lids_valid(old[1])
        old_yaw_ok = old is not None and finite(old[0])
        delta = yaw - old[0] if old_yaw_ok else 0.0
        refs = []
        for pos in range(3):
            rec = e.lid_refs[pos] if e.lid_refs else (math.nan, lids)
            new = remap_lids(rec[1], old[1], lids) if old_valid and lids_valid(rec[1]) else lids
            ref_yaw = rec[0]
            if pos == 1:
                new, ref_yaw = lids, yaw
            elif old_yaw_ok and finite(ref_yaw):
                ref_yaw += delta
            refs.append((ref_yaw, tuple(new)))
            self._write(lid_ref_keys(eye, pos, ref_yaw, new))
        e.lid_refs = refs

    def save_neutral(self, eye, now):                  # sub_1800b9460
        c = self.core
        if not self._fresh(eye, now) or c.expr[eye] is None:
            return 0
        e = self.cal.eyes[eye]
        neutral, mask = neutral_update(c.expr[eye], e.neutral)
        if mask:
            e.neutral = neutral
            self._write(neutral_keys(eye, neutral))
            c.reset_expression_filters(eye, mask)
        return mask

    def center_reference(self, eye, lm, lids, now, recenter_gaze):   # sub_1800b8690
        c, cal, e = self.core, self.cal, self.cal.eyes[eye]
        w = pupil_weights(lm, eye)
        opening = lid_at_pupil(lids, w, False) - lid_at_pupil(lids, w, True)
        lower_mid = 0.5 * (lm[4][1] + lm[5][1])
        wide = (lm[4][1] - lm[2][1]) + (lm[5][1] - lm[3][1])     # sub_1800c9020
        ref_y = lm[7][1]
        e.center_refs = tuple(lids)
        self._write(lid_geometry_keys(eye, lids))
        self.update_lid_refs(eye, lm, lids)
        e.lid_values = lid_values(opening, cal.eye_blink_close_amplitude)
        self._write(triple_keys("vrcft%sLidvalue%d", eye, e.lid_values))
        c.reset_lid_state(eye)
        e.squint_values = (0.5 * (lower_mid + lm[6][1]) - ref_y, lower_mid - ref_y, e.squint_values[2])
        self._write(triple_keys("squint%svalue%d", eye, e.squint_values))
        e.wide_values = (wide, wide * WIDE_REF_FACTOR, e.wide_values[2])
        self._write(triple_keys("eyeWide%svalue%d", eye, e.wide_values))
        report = {"neutral": self.save_neutral(eye, now)}
        if recenter_gaze:
            report["gaze"] = self._recenter_gaze(eye, lm)
        return report

    def _recenter_gaze(self, eye, lm):
        cal, e = self.cal, self.cal.eyes[eye]
        f = gaze_feature(lm, cal, eye)
        result = "ranges kept"
        if f is None:
            e.range_yaw, e.range_pitch = sanitize_range(e.range_yaw), sanitize_range(e.range_pitch)
        elif not (cal.precise_gaze and e.mesh and e.mesh_valid):
            e.range_yaw = recenter_range(e.range_yaw, f[0], RECENTER_HALF[0])
            e.range_pitch = recenter_range(e.range_pitch, f[1], RECENTER_HALF[1])
            result = "range centre moved"
        else:
            d = (f[0] - e.mesh[0][0], f[1] - e.mesh[0][1])
            if abs(d[0]) > EPS or abs(d[1]) > EPS:
                e.mesh = shift_mesh(e.mesh, *d)
                self._write(mesh_keys(eye, e.mesh, e.feature_mode))
                result = "mesh shifted by %+.4f, %+.4f" % d
            else:
                result = "mesh centre unchanged"
            e.range_yaw, e.range_pitch = ranges_from_mesh(e.mesh)
        self._write(range_keys(eye, e.range_yaw, e.range_pitch))
        return result

    # ---- recenter -----------------------------------------------------------------------
    def recenter(self, now):
        """eyeRecenterSlot: {eye: report dict or skip reason}."""
        out = {}
        for eye in (0, 1):
            chk = self._check(eye, now)
            if isinstance(chk, str):
                out[eye] = chk
                log.info("recenter: eye %d skipped, %s", eye, chk)
                continue
            out[eye] = self.center_reference(eye, *chk, now, recenter_gaze=True)
            log.info("recenter: eye %d %s", eye, out[eye])
        self.core.reload_calibration()
        return out

    # ---- coarse range -------------------------------------------------------------------
    def start_coarse(self, now):
        self.coarse = {"until": now + COARSE_DURATION_S,
                       "backup": [(e.range_yaw, e.range_pitch) for e in self.cal.eyes]}
        for e in self.cal.eyes:
            e.range_yaw, e.range_pitch = collapse_range(e.range_yaw), collapse_range(e.range_pitch)
        self.core.reload_calibration()

    def coarse_sample(self, eye, lm, feature):
        """Called by EyeCore for every open eye with a feature while the coarse flow runs."""
        if self.coarse is None or feature is None or not eye_open(lm, lid_points(lm, eye)):
            return
        e = self.cal.eyes[eye]
        e.range_yaw, e.range_pitch = expand_range(e.range_yaw, feature[0]), expand_range(e.range_pitch, feature[1])
        self.core.gaze.ranges[eye] = {"yaw": e.range_yaw, "pitch": e.range_pitch}

    def stop_coarse(self):
        if self.coarse is None:
            return None
        out = {}
        for eye, e in enumerate(self.cal.eyes):
            by, bp = self.coarse["backup"][eye]
            e.range_yaw, cy = finalize_range(e.range_yaw, by)
            e.range_pitch, cp = finalize_range(e.range_pitch, bp)
            self._write(range_keys(eye, e.range_yaw, e.range_pitch))
            out[eye] = {"yaw": e.range_yaw, "pitch": e.range_pitch, "changed": cy or cp}
        self.coarse = None
        self.core.reload_calibration()
        return out

    # ---- 13-point mesh ------------------------------------------------------------------
    def start_mesh(self, overlay=None):
        """overlay: (y offset m, ring scale) to store with the mesh; default: the current values."""
        if overlay is None:
            overlay = (self.cal.eyes[0].overlay_y_offset, self.cal.eyes[0].overlay_ring_scale)
        self.mesh = MeshCapture(overlay)
        return self.mesh

    def capture_mesh_point(self, now):
        """sub_18017efc0 phase 2, on one confirm: {"point", "eyes", "done", "saved"}."""
        m, c = self.mesh, self.core
        if m is None or m.step >= len(MESH_TARGETS):
            return None
        k = m.step
        captured = []
        for eye in (0, 1):
            lm = c.lm[eye]
            if not self._fresh(eye, now) or not eye_open(lm, lid_points(lm, eye)):
                continue
            f = feature_mode2(blend_reference(lm, c.wide_raw[eye]))     # sub_1800ba3a0
            if f is None:
                continue
            m.raw[eye][k] = f
            captured.append(eye)
        report = {"point": k, "eyes": captured, "done": False, "saved": {}}
        if not captured:
            report["skipped"] = "no reliable eye data"
            return report
        for eye in captured:
            chk = self._check(eye, now)
            if isinstance(chk, str):
                continue
            if k == 0:
                self.center_reference(eye, *chk, now, recenter_gaze=False)
            elif k in LID_REF_AT_POINT:
                self.lid_ref_at(eye, LID_REF_AT_POINT[k], *chk)
        m.step += 1
        if m.step >= len(MESH_TARGETS):
            report["done"], report["saved"] = True, self.finish_mesh()
        return report

    def eye_ready(self, eye, now):
        """True when capture_mesh_point would take this eye now (same tests)."""
        c = self.core
        lm = c.lm[eye]
        return (self._fresh(eye, now) and eye_open(lm, lid_points(lm, eye))
                and feature_mode2(blend_reference(lm, c.wide_raw[eye])) is not None)

    def lid_ref_at(self, eye, pos, lm, lids):          # sub_1800ba100
        f = gaze_feature(lm, self.cal, eye, force_mode2=True)
        if f is None or not finite(f[0]):
            return
        e = self.cal.eyes[eye]
        refs = list(e.lid_refs) if e.lid_refs else [(math.nan, lids)] * 3
        refs[pos] = (f[0], tuple(lids))
        e.lid_refs = refs
        self._write(lid_ref_keys(eye, pos, f[0], lids))

    def finish_mesh(self):                             # sub_18017dd40 with save
        m, saved = self.mesh, {}
        for eye, e in enumerate(self.cal.eyes):
            if any(p is None for p in m.raw[eye]):
                saved[eye] = False
                log.info("mesh: eye %d skipped, not all 13 points captured", eye)
                continue
            e.mesh = [(r[0], r[1], t[0], t[1]) for r, t in zip(m.raw[eye], MESH_TARGETS)]
            e.mesh_valid, e.feature_mode = True, 2
            e.overlay_y_offset, e.overlay_ring_scale = m.overlay
            self._write(mesh_keys(eye, e.mesh, 2, m.overlay))
            e.range_yaw, e.range_pitch = ranges_from_mesh(e.mesh)      # sub_180183890
            self._write(range_keys(eye, e.range_yaw, e.range_pitch))
            saved[eye] = True
        self.mesh = None
        self.core.reload_calibration()
        return saved

    def cancel_mesh(self):
        self.mesh = None


class MeshCapture:
    def __init__(self, overlay):
        self.overlay = tuple(overlay)
        self.step = 0
        self.raw = [[None] * len(MESH_TARGETS) for _ in range(2)]

    @property
    def target(self):
        return MESH_TARGETS[self.step] if self.step < len(MESH_TARGETS) else None
