"""Cymple's 70-value array (cympleface.dll): EyeCore fills fields 0..19 (lids, squint, wide,
brows, gaze), MouthCore fields 20..69.

One build() call mirrors one run of the 10 ms packet timer (sub_1800c2820):
  sub_1800bbca0  per-eye step (sub_1800b7c90) for each eye seen recently, one-eye hold,
                 blink state machine + lid filters (sub_1800bc0c0), binocular gaze
                 (sub_1800ba5f0, see gaze.GazeSolver), smoothing (sub_1800bc250 + sub_1800bc760),
                 hold again, packet fill (sub_1800bcb00)
  sub_1800bb920  switches, lid coupling (sub_18017c680), BrowPinch
Landmarks arrive through on_landmarks() (sub_1800b78f0), which low-passes pt7 in place.
Function names in comments are the addresses each helper mirrors.
"""
import math

from .calibration import EYE_AMP_KEYS, MOUTH_AMP_NAMES, amp_slots
from .eye import neutral_rescale
from .gaze import GazeSolver, OneEuro, clamp, feature_mode0, feature_mode2

EYE_SEEN_S = 0.2           # the 100 ms flag-refresh timer ages the "eye present" mask
PT7_FILTER = (1.0, 0.0)    # bank +0x1038: min cutoff 1.0, beta 0 (constructor)
EXPR_FILTER = (5.36, 0.092)   # filters 7..18: fixed slider 0.2 on the expression mapping
POST_FILTER = (2.2, 0.06)     # bank +0xEE8 (sub_1800bbd60)
BLINK_AMP_RANGE = (0.1, 0.3)  # sub_1800c5d40 id 2
BALANCE_RANGE = (0.78, 1.22)  # sub_1800c8810
ASYM_MAX = 0.2                # sub_1800c94f0


def clamp01(x):
    return clamp(x, 0.0, 1.0)


def finite(*xs):
    return all(math.isfinite(x) for x in xs)


def smoothstep01(x):                         # sub_18017c830 / sub_1800c2be0
    u = clamp01(x)
    return u * u * (3.0 - 2.0 * u)


def expression_filter_params(slider):        # sub_1800bbd60, filters 5-6 (0..0.5 -> 8/0.14 .. 1.4/0.02)
    t = clamp01((clamp(slider, 0.0, 0.5) - 0.0) / 0.5)
    return 8.0 + (1.4 - 8.0) * t, 0.14 + (0.02 - 0.14) * t


# ------------------------------------------------------------------ per-eye geometry
def lid_points(lm, eye):                     # sub_1800c79e0 -> (upper_outer, upper_inner, lower_outer, lower_inner)
    if eye == 1:
        return (lm[2][1], lm[3][1], lm[4][1], lm[5][1])
    return (lm[3][1], lm[2][1], lm[5][1], lm[4][1])


def lids_valid(l):                           # sub_1800c7930
    return l is not None and finite(*l) and l[2] > l[0] and l[3] > l[1]


def corners_valid(lm):                       # sub_1800c6110
    (x0, y0), (x1, y1) = lm[0], lm[1]
    return finite(x0, y0, x1, y1) and max(abs(x0 - x1), abs(y0 - y1)) > 0.01


def pupil_weights(lm, eye):                  # sub_1800c8df0 -> (w_outer, w_inner)
    if not (corners_valid(lm) and finite(*lm[6])):
        return (0.5, 0.5)
    outer, inner = (lm[0][0], lm[1][0]) if eye == 1 else (lm[1][0], lm[0][0])
    span = inner - outer
    t = 0.5
    if abs(span) > 1e-5:
        t = clamp01((lm[6][0] - outer) / span)
    return (1.0 - t, t)


def lid_at_pupil(lids, w, upper):            # sub_1800c8fa0
    return lids[0] * w[0] + lids[1] * w[1] if upper else lids[2] * w[0] + lids[3] * w[1]


def mean_opening(l):                         # sub_1800c86c0
    return 0.5 * ((l[2] - l[0]) + (l[3] - l[1])) if lids_valid(l) else 0.0


def lid_map(opening, open_ref, closed_ref):  # sub_1800c9830
    if not finite(opening, open_ref, closed_ref):
        return 0.0
    c = clamp(closed_ref, 0.0, max(open_ref - 1e-5, 0.0))
    d = open_ref - c
    if d <= 1e-5:
        return 0.0
    return clamp01((open_ref - opening) / d)


def column_close(opening, ref_opening, amp):     # sub_1800c9090
    if not finite(opening, ref_opening) or ref_opening <= 1e-5:
        return 0.0
    k = clamp(amp, *BLINK_AMP_RANGE)
    closed, opened = k * ref_opening, 0.8 * ref_opening
    d = opened - closed
    if d <= 1e-5:
        return 0.0
    return 1.0 - clamp01((opening - closed) / d)


def lid_close_vs_ref(lids, refs, amp):       # sub_1800c9370: median over outer / middle / inner columns
    outer = column_close(lids[2] - lids[0], refs[2] - refs[0], amp)
    mid = column_close(0.5 * (lids[2] + lids[3]) - 0.5 * (lids[0] + lids[1]),
                       0.5 * (refs[2] + refs[3]) - 0.5 * (refs[0] + refs[1]), amp)
    inner = column_close(lids[3] - lids[1], refs[3] - refs[1], amp)
    return sorted((outer, mid, inner))[1]    # sub_1800c9220: sum - min - max


def blend_close(x, scale, offset):           # sub_18017c590
    x = clamp01(x)
    y = clamp01(x * scale - offset)
    w = smoothstep01((x - 0.5) / 0.5)
    return y + (x - y) * w


def yaw_ramp(a, b):                          # sub_1800c2ea0
    m = max(abs(a), abs(b))
    if m <= 0.2:
        return 0.0
    if m >= 0.5:
        return 1.0
    return smoothstep01((m - 0.2) / 0.3)


def squint_deadzone(x):                      # sub_1800c2c60
    x = clamp01(x)
    return 0.0 if x <= 0.3 else (x - 0.3) / 0.7


def pitch_attenuation(p):                    # sub_1800c2e40
    if p <= 0.0:
        return 1.0
    return 0.0 if p >= 0.5 else 1.0 - 2.0 * p


def expr_deadzone(x):                        # sub_1800cfc30
    return clamp01((x - 0.2) / 0.8)


def pair_pull(a, b, strength):               # sub_1800c2ce0: pull the smaller value toward the larger
    d = abs(clamp01(a) - clamp01(b))
    w = 1.0 - smoothstep01(d / 0.6)
    w *= strength
    if w <= 0.0:
        return a, b
    if a > b:
        b += (a - b) * w
    elif b > a:
        a += (b - a) * w
    return a, b


def asymmetry(a, b):                         # sub_18017c8b0
    return smoothstep01((abs(a - b) - 0.2) / 0.25)


def couple_lids(l, r, k):                    # sub_18017c900
    hi_is_l = l > r
    hi, lo = (l, r) if hi_is_l else (r, l)
    d = asymmetry(l, r)
    a = clamp01((0.34 + 0.26 * hi) * max(k, 0.0)) * (1.0 - d)
    lo = lo + (hi - lo) * a
    e = smoothstep01((hi - 0.5) / 0.3) * d
    hi = hi + (1.0 - hi) * e
    return (hi, lo) if hi_is_l else (lo, hi)


def wide_vs_lid(wide, lid):                  # sub_18017cb10
    return clamp01(wide) * smoothstep01(((1.0 - lid) - 0.5) / 0.5)


def symmetrize(a, b, w):                     # sub_18017cba0
    a, b = clamp01(a), clamp01(b)
    f = (1.0 - smoothstep01(abs(a - b) / 0.6)) * smoothstep01(min(a, b) / 0.2) * w
    if a > b:
        b += (a - b) * f
    elif b > a:
        a += (b - a) * f
    return a, b


class EyeCore:
    """State of faceOscClass for the eye fields; build() returns packet fields 0..19."""

    def __init__(self, calib):
        self.cal = calib
        self.eyes = calib.eyes
        self.lm = [None, None]               # landmarks with filtered pt7 (this+0x408 + eye*0x5c)
        self.expr = [None, None]             # raw blendshape_output[3]
        self.seen = [-math.inf, -math.inf]
        self.pt7 = [[OneEuro(*PT7_FILTER), OneEuro(*PT7_FILTER)] for _ in range(2)]
        z = lambda: [0.0, 0.0]
        self.lid, self.raw_lid, self.squint = z(), z(), z()
        self.wide, self.lower, self.inner, self.outer, self.pinch = z(), z(), z(), z(), z()
        self.sm_state, self.sm_init = z(), [False, False]
        self.wide_raw = z()                  # neutral-rescaled widen before smoothing (+0x115C), pt7 blend input
        self.calibrator = None               # calib.EyeCalibrator; samples features during coarse calibration
        self.lid_filters = [OneEuro(), OneEuro()]
        self.expr_filters = {k: [OneEuro(*EXPR_FILTER), OneEuro(*EXPR_FILTER)]
                             for k in ("squint", "wide", "lower", "inner", "outer", "pinch")}
        self.post_filters = {k: [OneEuro(*POST_FILTER), OneEuro(*POST_FILTER)] for k in ("wide", "lower", "inner")}
        self.gaze = GazeSolver(
            meshes=[e.mesh for e in calib.eyes],
            mesh_valid=tuple(bool(e.mesh) and e.mesh_valid and calib.precise_gaze for e in calib.eyes),
            feature_mode=tuple(e.feature_mode for e in calib.eyes),
            ranges={i: {"yaw": e.range_yaw, "pitch": e.range_pitch} for i, e in enumerate(calib.eyes)},
            precise=calib.precise_gaze)
        self.configure()

    def configure(self):
        """(Re)apply the sliders and amplitudes from self.cal; filter states are kept. Blink
        amplitude, sync rate and the switches are read from self.cal on every packet."""
        c = self.cal
        for f in self.lid_filters:
            f.set_params(*expression_filter_params(c.eye_expression_filter))
        self.gaze.configure(c.eye_gaze_filter, c.eye_gaze_amplitude, c.eye_squint_gaze_compensation)
        self.amp = dict(zip(EYE_AMP_KEYS, amp_slots(c.amps)[len(MOUTH_AMP_NAMES):]))

    def reload_calibration(self):
        """After a calibration flow changed meshes, ranges or the precise flag in self.cal."""
        c, g = self.cal, self.gaze
        g.meshes = [e.mesh for e in c.eyes]
        g.mesh_valid = tuple(bool(e.mesh) and e.mesh_valid and c.precise_gaze for e in c.eyes)
        g.mode = tuple(e.feature_mode for e in c.eyes)
        g.ranges = {i: {"yaw": e.range_yaw, "pitch": e.range_pitch} for i, e in enumerate(c.eyes)}
        g.precise = c.precise_gaze

    def reset_lid_state(self, eye):          # recenter: lid output, raw lid, filters 5/6, blink state
        self.lid[eye] = self.raw_lid[eye] = 0.0
        self.lid_filters[eye] = OneEuro(*expression_filter_params(self.cal.eye_expression_filter))
        self.sm_state[eye], self.sm_init[eye] = 0.0, False

    def reset_expression_filters(self, eye, mask):   # sub_1800b9460 after a new neutral
        for bit, names in ((1, ("wide",)), (2, ("lower", "pinch")), (4, ("inner", "outer"))):
            if mask & bit:
                for n in names:
                    self.expr_filters[n][eye] = OneEuro(*EXPR_FILTER)
                    if n in self.post_filters:
                        self.post_filters[n][eye] = OneEuro(*POST_FILTER)

    # ---------------------------------------------------------------- input
    def on_landmarks(self, eye, landmarks, expression, t):
        """sub_1800b78f0: pt7 is low-passed in place when valid, its filters reset otherwise."""
        lm = [tuple(map(float, p)) for p in landmarks]
        x, y = lm[7]
        if finite(x, y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            lm[7] = (self.pt7[eye][0](x, t), self.pt7[eye][1](y, t))
        else:
            self.pt7[eye] = [OneEuro(*PT7_FILTER), OneEuro(*PT7_FILTER)]
        self.lm[eye] = lm
        self.expr[eye] = [float(v) for v in expression]
        self.seen[eye] = t

    # ---------------------------------------------------------------- per-eye pieces
    def _raw_yaw(self, lm, eye):
        """Gaze feature yaw used for the lid references (sub_1800c7590 with blend off)."""
        if lm is None:
            return math.nan
        f = feature_mode2(lm) if self.eyes[eye].feature_mode == 2 else None
        if f is None:
            f = feature_mode0(lm)
        return f[0] if f else math.nan

    def _interp_refs(self, eye, yaw):        # sub_1800c8050
        refs = self.eyes[eye].lid_refs
        if not refs or not finite(yaw) or any(not finite(y) or not lids_valid(r) for y, r in refs):
            return None
        recs = sorted(refs, key=lambda r: r[0])
        (y0, r0), (y1, r1), (y2, r2) = recs
        if y1 - y0 <= 1e-5 or y2 - y1 <= 1e-5:
            return None
        if y0 >= yaw:
            return r0
        if yaw >= y2:
            return r2
        (ya, ra), (yb, rb) = (recs[0], recs[1]) if y1 >= yaw else (recs[1], recs[2])
        h = yb - ya
        if h <= 1e-5:
            return None
        t = clamp01((yaw - ya) / h)          # sub_1800c7c60
        out = tuple(a + (b - a) * t for a, b in zip(ra, rb))
        return out if lids_valid(out) else None

    def _eye_refs(self, i, yaw):             # sub_1800c8730: that eye's own yaw when available
        own = self._raw_yaw(self.lm[i], i)
        return self._interp_refs(i, own if finite(own) else yaw)

    def _balance(self, eye, yaw):            # sub_1800c8810
        if not self.cal.precise_gaze or not finite(yaw):
            return 1.0
        r0, r1 = self._eye_refs(0, yaw), self._eye_refs(1, yaw)
        if r0 is None or r1 is None:
            return 1.0
        o0, o1 = mean_opening(r0), mean_opening(r1)
        m = 0.5 * (o0 + o1)
        if o0 <= 1e-5 or o1 <= 1e-5 or m <= 1e-5:
            return 1.0
        return clamp((o0 if eye == 0 else o1) / m, *BALANCE_RANGE)

    def _asymmetry(self, eye, yaw):          # sub_1800c94f0
        if not self.cal.precise_gaze or not finite(yaw):
            return 0.0
        r0, r1 = self._eye_refs(0, yaw), self._eye_refs(1, yaw)
        c0 = self.eyes[0].lid_refs[1][1] if self.eyes[0].lid_refs else None
        c1 = self.eyes[1].lid_refs[1][1] if self.eyes[1].lid_refs else None
        if r0 is None or r1 is None or not lids_valid(c0) or not lids_valid(c1):
            return 0.0
        amp = self.cal.eye_blink_close_amplitude
        b0, b1 = lid_close_vs_ref(r0, c0, amp), lid_close_vs_ref(r1, c1, amp)
        d = max((b0 if eye == 0 else b1) - 0.5 * (b0 + b1), 0.0)
        if d <= 1e-5:
            return 0.0
        return clamp(d * yaw_ramp(yaw, yaw) * 0.75, 0.0, ASYM_MAX)

    def _eye_step(self, eye):                # sub_1800b7c90 + sub_1800b83b0
        lm, cfg = self.lm[eye], self.eyes[eye]
        lids = lid_points(lm, eye)
        refs = cfg.center_refs
        yaw = math.nan
        if self.cal.precise_gaze:
            yaw = self._raw_yaw(lm, eye)
            interp = self._interp_refs(eye, yaw) if finite(yaw) else None
            if interp is not None:
                refs = interp
        valid = lids_valid(lids) and lids_valid(refs)
        w = pupil_weights(lm, eye)
        upper, lower = lid_at_pupil(lids, w, True), lid_at_pupil(lids, w, False)
        scale = self._balance(eye, yaw) if valid else 1.0
        offset = self._asymmetry(eye, yaw) if valid else 0.0
        lower_mid = 0.5 * (lm[4][1] + lm[5][1])
        lid = lid_map(lower - upper, cfg.lid_values[0], cfg.lid_values[1])
        s0, s1 = cfg.squint_values[0], cfg.squint_values[1]
        # Deviation: without a squint calibration (value0 == value1) the original divides by
        # zero and its pitch goes non-finite; we report no squint instead.
        self.squint[eye] = 1.0 - (lower_mid - lm[7][1] - s0) / (s1 - s0) if s1 != s0 else 0.0
        if valid:
            lid = lid_close_vs_ref(lids, refs, self.cal.eye_blink_close_amplitude)
            lid = blend_close(lid, scale, offset)
        self.lid[eye] = lid
        self.raw_lid[eye] = lid                                       # this+0xAB0
        n = cfg.neutral
        ch = [neutral_rescale(self.expr[eye][i], n[i]) for i in range(3)]
        self.wide[eye] = ch[0]
        self.lower[eye] = self.pinch[eye] = ch[1]
        self.inner[eye] = self.outer[eye] = ch[2]

    def _hold(self, present):                # sub_1800c8cd0: one eye present -> copy it across
        if present == (True, False):
            src, dst = 0, 1
        elif present == (False, True):
            src, dst = 1, 0
        else:
            return
        for arr in (self.lid, self.squint, self.wide, self.lower, self.inner, self.outer, self.pinch):
            arr[dst] = arr[src]

    def _blink(self, eye, x):                # sub_1800c5860
        x = clamp01(x)
        if not self.sm_init[eye]:
            self.sm_state[eye], self.sm_init[eye] = x, True
            return x
        s = self.sm_state[eye]
        d = x - s
        if d >= 0.0:
            gain = clamp(0.78 + 0.17 * clamp01(d / 0.35) + 0.03 * x, 0.78, 0.98)
        else:
            o = 1.0 - clamp01(0.5 * (x + s) / 0.25)
            gain = clamp(0.30 + 0.36 * clamp01(s) - 0.03 * o, 0.24, 0.72)
        s = clamp01(s + gain * d)
        if x < 0.03 and s < 0.03:
            s = 0.0
        self.sm_state[eye] = s
        return s

    # ---------------------------------------------------------------- one packet
    def build(self, now):
        present = tuple(self.lm[e] is not None and now - self.seen[e] <= EYE_SEEN_S for e in (0, 1))
        if not any(present):
            return None
        for e in (0, 1):
            if present[e]:
                self._eye_step(e)
        self._hold(present)
        for e in (0, 1):                                              # sub_1800bc0c0
            self.lid[e] = clamp01(self.lid_filters[e](self._blink(e, self.lid[e]), now))
        widen = tuple(self.wide)
        self.wide_raw = list(widen)
        lid_close = tuple(max(self.lid[e], self.raw_lid[e]) for e in (0, 1))
        g = self.gaze.step(self.lm[0] if present[0] else None, self.lm[1] if present[1] else None,
                           widen=widen, lid_close=lid_close, squint=tuple(self.squint), now=now)
        cal = self.calibrator
        if cal is not None and cal.coarse is not None:                # sub_1800c0950 while coarse runs
            for e in (0, 1):
                if present[e]:
                    cal.coarse_sample(e, self.lm[e], self.gaze.features[e])
        for name, arr in (("squint", self.squint), ("wide", self.wide), ("lower", self.lower),
                          ("inner", self.inner), ("outer", self.outer), ("pinch", self.pinch)):
            for e in (0, 1):                                          # sub_1800bc250, filters 7..18
                arr[e] = self.expr_filters[name][e](arr[e], now)
        for name, arr in (("wide", self.wide), ("lower", self.lower), ("inner", self.inner)):
            for e in (0, 1):                                          # sub_1800bc760
                arr[e] = clamp01(self.post_filters[name][e](arr[e], now))
        self.pinch, self.outer = list(self.lower), list(self.inner)
        self._hold(present)
        return self._fill(g, present)

    def _fill(self, g, present):             # sub_1800bcb00 + sub_1800bb920
        a = self.amp
        f = [0.0] * 20
        f[0], f[1], f[2], f[3] = g["EyePitch_L"], g["EyePitch_R"], g["EyeYaw_L"], g["EyeYaw_R"]
        f[6], f[7] = clamp01(self.lid[0]), clamp01(self.lid[1])
        att = pitch_attenuation(self.gaze.filtered[0])
        for e in (0, 1):
            f[8 + e] = clamp01(clamp01(squint_deadzone(self.squint[e]) * a["Amp_EyeSquint"]) * att)
            f[10 + e] = clamp01(expr_deadzone(self.wide[e]) * a["Amp_EyeWide"])
            f[12 + e] = clamp01(expr_deadzone(self.lower[e]) * a["Amp_BrowDown"])
            f[14 + e] = clamp01(expr_deadzone(self.inner[e]) * a["Amp_BrowInRaise"])
            f[16 + e] = clamp01(expr_deadzone(self.outer[e]) * a["Amp_BrowInRaise"])
            f[18 + e] = clamp01(self.pinch[e])
        if present[0] == present[1]:                                  # sub_1800c8d70 false
            f[14], f[15] = pair_pull(f[14], f[15], 1.0)
            f[16], f[17] = pair_pull(f[16], f[17], 1.0)
        sw = self.cal.switches
        if sw.get(12):                                                # "silly"
            p = 0.5 * (f[0] + f[1])
            f[0], f[1] = p, -p
            y = 0.5 * (f[2] + f[3])
            f[2], f[3] = y, -y
        if sw.get(7):                                                 # eyePitchFlip
            f[0], f[1] = -f[0], -f[1]
        sync = clamp(self.cal.eye_gaze_sync_rate, 0.0, 100.0) / 100.0
        if sync > 0.0 and not sw.get(12):
            m = 0.5 * (f[2] + f[3])
            f[2], f[3] = f[2] + (m - f[2]) * clamp01(sync), f[3] + (m - f[3]) * clamp01(sync)
        k = 1.0 + 0.85 * yaw_ramp(f[2], f[3])
        if sw.get(9):                                                 # lidSync
            k += 0.65
        both = present == (True, True)
        f[6], f[7] = clamp01(f[6]), clamp01(f[7])                     # sub_18017c680
        w = 1.0 - asymmetry(f[6], f[7])
        if both:
            f[6], f[7] = couple_lids(f[6], f[7], k)
        f[10], f[11] = wide_vs_lid(f[10], f[6]), wide_vs_lid(f[11], f[7])
        if both:
            f[10], f[11] = symmetrize(f[10], f[11], w)
            f[8], f[9] = symmetrize(f[8], f[9], w)
            f[12], f[13] = symmetrize(f[12], f[13], w)
        f[18] = clamp01(f[12] * a["Amp_BrowPinch"])
        f[19] = clamp01(f[13] * a["Amp_BrowPinch"])
        return f


# ------------------------------------------------------------------ mouth half (fields 20..69)
MOUTH_CHANNELS = 30


def mouth_filter_params(filer):              # sub_1800c5ba0 ([Cymple_Mouth] filer, getMouthPara field 0)
    t = clamp01(filer / 0.5)
    return 9.0 - 7.0 * t, 0.12 - 0.09 * t


# Field -> amplitude slot as sub_1800bd5e0 reads this+0x1184 + 4*slot. Slots 14..18 hold the keys
# ShrugLower, MouthSmile, MouthDimple, MouthFrown, MouthStretch (key table order), so Smile is scaled
# by Amp_ShrugLowerAmp, Dimple by Amp_MouthSmileAmp, Frown by Amp_MouthDimpleAmp, Stretch by
# Amp_MouthFrownAmp and ShrugLower by Amp_MouthStretchAmp. Cymple does this; we keep it.
MOUTH_AMP_SLOT = {
    "NoseSneerLeft": 0, "NoseSneerRight": 0, "CheekSquintLeft": 1, "CheekSquintRight": 1,
    "CheekPuffLeft": 2, "CheekPuffRight": 2, "CheekSuckRight": 3, "JawLeft": 4, "JawRight": 4,
    "JawFwd": 5, "JawOpen": 6, "MouthClosed": 7,
    "MouthUpperShiftLeft": 8, "MouthUpperShiftRight": 8, "MouthLowerShiftLeft": 8, "MouthLowerShiftRight": 8,
    "LipSuckUL": 9, "LipSuckUR": 9, "LipSuckDL": 9, "LipSuckDR": 9,
    "MouthFunnelUL": 10, "MouthFunnelUR": 10, "MouthFunnelDL": 10, "MouthFunnelDR": 10,
    "MouthPuckerUpper": 11, "MouthPuckerLower": 11, "LipRaiseUL": 12, "LipRaiseUR": 12,
    "LipDepressDL": 13, "LipDepressDR": 13, "MouthSmileLeft": 14, "MouthSmileRight": 14,
    "MouthDimpleLeft": 15, "MouthDimpleRight": 15, "MouthFrownLeft": 16, "MouthFrownRight": 16,
    "MouthStretchLeft": 17, "MouthStretchRight": 17, "MouthShrugLower": 18, "TongueOut": 19,
    "TongueLeft": 20, "TongueRight": 20, "TongueUp": 21, "TongueDown": 21, "TongueRoll": 22,
}
MOUTH_AMP_DOUBLE = ("JawLeft", "JawRight")   # x 2.0 on top of Amp_JawShiftAmp


def map_mouth(m):
    """sub_1800bee00: 30 filtered channels -> fields 20..69, before amplitudes."""
    jaw, shift, tongue_x, tongue_y = m[3] - m[4], m[8] - m[9], m[25] - m[26], m[27] - m[28]
    lip = abs(1.0 - m[23])                   # sub_180035960 = fabs
    return {
        "NoseSneerLeft": m[15], "NoseSneerRight": m[16], "CheekSquintLeft": m[19], "CheekSquintRight": m[20],
        "CheekPuffLeft": m[0], "CheekPuffRight": m[1], "CheekSuckLeft": m[2], "CheekSuckRight": m[2],
        "JawLeft": jaw, "JawRight": -jaw, "JawFwd": m[5], "JawOpen": m[6], "MouthClosed": m[7],
        "MouthUpperShiftLeft": shift, "MouthUpperShiftRight": -shift,
        "MouthLowerShiftLeft": shift, "MouthLowerShiftRight": -shift,
        "LipSuckUL": m[10], "LipSuckUR": m[10], "LipSuckDL": m[11], "LipSuckDR": m[11],
        "MouthFunnelUL": m[12], "MouthFunnelUR": m[12], "MouthFunnelDL": m[13], "MouthFunnelDR": m[13],
        "MouthPuckerUpper": m[14], "MouthPuckerLower": m[14],
        "LipRaiseUL": lip * m[15], "LipRaiseUR": lip * m[16], "LipDepressDL": lip * m[17], "LipDepressDR": lip * m[18],
        "MouthSmileLeft": m[19], "MouthSmileRight": m[20], "MouthDimpleLeft": m[19], "MouthDimpleRight": m[20],
        "MouthFrownLeft": m[21], "MouthFrownRight": m[22], "MouthStretchLeft": m[21], "MouthStretchRight": m[22],
        "MouthShrugUpper": clamp01(2.0 * m[23]) * 0.5 * (m[15] + m[16]), "MouthShrugLower": m[23],
        "TongueOut": m[24], "TongueLeft": tongue_x, "TongueRight": -tongue_x,
        "TongueUp": tongue_y, "TongueDown": -tongue_y,
        "TongueCurlUp": 0.0, "TongueBendDown": 0.0, "TongueWide": 0.0, "TongueRoll": m[29],
    }


def apply_mouth_amps(f, slots):
    """sub_1800bd5e0: x amplitude, clamp to [0, 1]; CheekSuckLeft copies the right side; ShrugUpper
    is only clamped; fields 66..68 stay 0."""
    out = dict(f)
    for name, slot in MOUTH_AMP_SLOT.items():
        out[name] = clamp01(f[name] * slots[slot] * (2.0 if name in MOUTH_AMP_DOUBLE else 1.0))
    out["CheekSuckLeft"] = out["CheekSuckRight"]
    out["MouthShrugUpper"] = clamp01(f["MouthShrugUpper"])
    return out


class MouthCore:
    """Mouth fields of faceOscClass. Once the first mouth frame arrived (flag +0x12E4, never
    cleared) every packet tick filters the latest 30 values again and refills fields 20..69."""

    def __init__(self, calib):
        self.cal = calib
        self.filters = [OneEuro() for _ in range(MOUTH_CHANNELS)]            # bank +0x12E8
        self.raw = None                                                       # this+0x126C
        self.configure()

    def configure(self):
        """(Re)apply the mouth filter slider and the amplitudes from self.cal."""
        self.slots = amp_slots(self.cal.amps)[:len(MOUTH_AMP_NAMES)]
        for f in self.filters:
            f.set_params(*mouth_filter_params(self.cal.mouth.filter))

    def on_values(self, values):             # sub_1800b7c30 + sub_1800c5ea0
        self.raw = [clamp01(float(v)) if math.isfinite(v) else 0.0 for v in values[:MOUTH_CHANNELS]]

    def build(self, now):
        if self.raw is None:
            return None
        m = [clamp01(f(x, now)) for f, x in zip(self.filters, self.raw)]      # sub_1800bed10 -> +0x11F4
        return apply_mouth_amps(map_mouth(m), self.slots)
