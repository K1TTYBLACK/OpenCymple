import math

import pytest

from cymple_linux import face as F
from cymple_linux.calibration import Calibration


def test_lid_map_endpoints():
    assert F.lid_map(0.16195, 0.16195, 0.0425) == 0.0
    assert F.lid_map(0.0425, 0.16195, 0.0425) == 1.0
    assert abs(F.lid_map(0.5 * (0.16195 + 0.0425), 0.16195, 0.0425) - 0.5) < 1e-9
    assert F.lid_map(float("nan"), 0.16, 0.04) == 0.0


def test_column_close_and_median():
    # amp 0.21: closed = 0.21 * ref, open = 0.8 * ref
    assert F.column_close(0.16, 0.2, 0.21) == pytest.approx(0.0, abs=1e-9)
    assert abs(F.column_close(0.042, 0.2, 0.21) - 1.0) < 1e-9
    assert F.column_close(0.1, 0.0, 0.21) == 0.0
    lids = (0.40, 0.42, 0.60, 0.50)          # outer open, inner nearly closed
    refs = (0.40, 0.40, 0.60, 0.60)
    vals = sorted([F.column_close(0.20, 0.20, 0.21), F.column_close(0.14, 0.20, 0.21),
                   F.column_close(0.08, 0.20, 0.21)])
    assert F.lid_close_vs_ref(lids, refs, 0.21) == pytest.approx(vals[1])
    assert vals[1] == pytest.approx(1 - (0.14 - 0.042) / (0.16 - 0.042))


def test_blend_close():
    assert F.blend_close(0.2, 1.0, 0.0) == pytest.approx(0.2)
    assert F.blend_close(0.2, 1.2, 0.1) == pytest.approx(0.14)
    assert F.blend_close(1.0, 0.5, 0.2) == pytest.approx(1.0)


def test_blink_state_machine():
    core = F.EyeCore(Calibration())
    assert core._blink(0, 0.0) == 0.0                     # first sample passes through
    assert core._blink(0, 1.0) == pytest.approx(0.98)     # closing: gain clamp(0.78+0.17+0.03)
    assert core._blink(0, 0.0) == pytest.approx(0.98 - 0.6528 * 0.98)
    for _ in range(30):
        s = core._blink(0, 0.0)
    assert s == 0.0                                       # snaps to 0 below 0.03


def test_couple_lids():
    assert F.couple_lids(1.0, 0.0, 1.0) == (1.0, 0.0)     # clear wink stays a wink
    hi, lo = F.couple_lids(0.5, 0.4, 1.0)
    assert hi == 0.5 and lo == pytest.approx(0.4 + 0.1 * 0.47)
    assert F.couple_lids(0.4, 0.5, 1.0) == pytest.approx((0.4 + 0.1 * 0.47, 0.5))


def test_lid_points_and_weights():
    lm = [(0.2, 0.5), (0.8, 0.5), (0.3, 0.40), (0.7, 0.41), (0.3, 0.60), (0.7, 0.61), (0.65, 0.5), (0.5, 0.55)]
    assert F.lid_points(lm, 0) == (0.41, 0.40, 0.61, 0.60)
    assert F.lid_points(lm, 1) == (0.40, 0.41, 0.60, 0.61)
    wo, wi = F.pupil_weights(lm, 1)                       # right eye: outer = pt0
    assert (wo, wi) == pytest.approx((0.25, 0.75))
    assert F.pupil_weights(lm, 0) == pytest.approx((0.75, 0.25))


def test_build_with_synthetic_landmarks():
    core = F.EyeCore(Calibration())
    lm = [(0.2, 0.5), (0.8, 0.5), (0.3, 0.40), (0.7, 0.41), (0.3, 0.60), (0.7, 0.61), (0.5, 0.5), (0.5, 0.55)]
    assert core.build(0.0) is None
    for i in range(20):
        t = i * 0.01
        core.on_landmarks(0, lm, (0.9, 0.0, 0.9), t)
        core.on_landmarks(1, lm, (0.9, 0.0, 0.9), t)
        f = core.build(t)
    assert len(f) == 20 and all(math.isfinite(v) for v in f)
    assert all(-1.0 <= v <= 1.0 for v in f[:4]) and all(0.0 <= v <= 1.0 for v in f[4:])
    later = core.build(10.0)                              # eyes not seen for 10 s
    assert later is None


# ---- mouth (sub_1800bee00 / sub_1800bd5e0 / sub_1800c5ba0)

def test_mouth_mapping_table():
    m = [0.0] * 30
    m[3], m[4] = 0.5, 0.2                      # jaw shift
    m[8], m[9] = 0.1, 0.4                      # mouth shift
    m[15], m[16], m[17], m[23] = 0.6, 0.4, 0.3, 0.25
    m[25], m[26], m[27], m[28], m[29] = 0.7, 0.1, 0.0, 0.3, 0.2
    f = F.map_mouth(m)
    assert f["JawLeft"] == pytest.approx(0.3) and f["JawRight"] == pytest.approx(-0.3)
    assert f["MouthUpperShiftRight"] == pytest.approx(0.3) == f["MouthLowerShiftRight"]
    assert f["LipRaiseUL"] == pytest.approx(0.75 * 0.6) and f["LipDepressDL"] == pytest.approx(0.75 * 0.3)
    assert f["MouthShrugUpper"] == pytest.approx(0.5 * 0.5 * (0.6 + 0.4))
    assert f["TongueLeft"] == pytest.approx(0.6) and f["TongueDown"] == pytest.approx(0.3)
    assert f["TongueRoll"] == 0.2 and f["TongueWide"] == 0.0


def test_mouth_amplitude_slots_follow_cymple():
    from cymple_linux.calibration import amp_slots
    amps = {"Amp_ShrugLowerAmp": 0.5, "Amp_MouthSmileAmp": 0.6, "Amp_MouthDimpleAmp": 1.6,
            "Amp_MouthFrownAmp": 2.25, "Amp_MouthStretchAmp": 0.7, "Amp_JawShiftAmp": 1.0,
            "Amp_TongueOutAmp": 9.0, "Amp_EyeWide": 0.3}
    slots = amp_slots(amps)
    assert len(slots) == 28 and slots[19] == 5.0                          # clamped to [0, 5]
    assert slots[26] == 0.3                                                # BrowInRaise falls back to EyeWide
    f = F.apply_mouth_amps(F.map_mouth([0.2] * 30), slots[:23])
    # slots 14..18 are read for Smile, Dimple, Frown, Stretch, ShrugLower (key order differs)
    assert f["MouthSmileLeft"] == pytest.approx(0.2 * 0.5)
    assert f["MouthDimpleLeft"] == pytest.approx(0.2 * 0.6)
    assert f["MouthFrownLeft"] == pytest.approx(0.2 * 1.6)
    assert f["MouthStretchLeft"] == pytest.approx(0.2 * 2.25)
    assert f["MouthShrugLower"] == pytest.approx(0.2 * 0.7)
    assert f["JawLeft"] == 0.0 and f["JawRight"] == 0.0                   # 0.2 - 0.2, x 2
    assert f["TongueOut"] == 1.0                                           # clamped to [0, 1]


def test_mouth_filter_params_and_core():
    assert F.mouth_filter_params(0.0) == pytest.approx((9.0, 0.12))
    assert F.mouth_filter_params(0.25) == pytest.approx((5.5, 0.075))
    assert F.mouth_filter_params(1.0) == pytest.approx((2.0, 0.03))
    core = F.MouthCore(Calibration())
    assert core.build(0.0) is None
    core.on_values([float("nan")] + [1.5] + [0.4] * 28)
    f = core.build(0.0)
    assert f["CheekPuffLeft"] == 0.0 and f["CheekPuffRight"] == 1.0       # NaN -> 0, clamp to 1
    assert f["JawOpen"] == pytest.approx(0.4)


def test_live_reconfigure_keeps_filter_state():
    from cymple_linux.gaze import OneEuro, gaze_filter_params
    f = OneEuro(4.0, 0.06)
    f(0.0, 0.0)
    f(1.0, 0.02)
    x = f.x
    f.set_params(1.8, 0.02)
    assert f.x == x and (f.min_cutoff, f.beta) == (1.8, 0.02)

    cal = Calibration()
    core = F.EyeCore(cal)
    cal.eye_gaze_filter, cal.eye_gaze_amplitude, cal.eye_expression_filter = 0.2, 0.7, 0.5
    cal.amps = {"Amp_EyeWide": 2.0}
    core.configure()
    assert core.gaze.filters[0].min_cutoff == gaze_filter_params(0.2)[0] and core.gaze.amplitude == 0.7
    assert core.lid_filters[0].min_cutoff == pytest.approx(F.expression_filter_params(0.5)[0])
    assert core.amp["Amp_EyeWide"] == 2.0

    mouth = F.MouthCore(cal)
    cal.mouth.filter = 0.5
    cal.amps = {"Amp_ShrugLowerAmp": 3.0}             # slot 14 = what the Smile slider sets
    mouth.configure()
    assert mouth.filters[0].min_cutoff == pytest.approx(2.0) and mouth.slots[14] == 3.0


def test_mouth_roi_resets_when_ui_changes_it():
    import numpy as np
    from cymple_linux.calibration import MouthCalibration
    from cymple_linux.mouth import MouthV1Tracker

    class Model:
        def __init__(self, out):
            self.out = out

        def run(self, x):
            return {"o": self.out}

    quad = np.float32([0.2, 0.2, 0.2, 0.8, 0.8, 0.2, 0.8, 0.8])
    cal = MouthCalibration(auto_roi=True)
    tr = MouthV1Tracker(Model(np.zeros(30, np.float32)), Model(quad), cal)
    frame = np.full((120, 160), 128, np.uint8)
    assert tr.process(frame, now_ms=0).quad is not None
    cal.auto_roi = False                                  # unticked in the window
    obs = tr.process(frame, now_ms=10)
    assert obs.quad is None and obs.rect is not None
    cal.auto_roi = True
    assert tr.process(frame, now_ms=20).quad is not None
    cal.rotation = 90.0                                   # the quad was found on the old rotation
    tr.last_roi_ms = 20                                   # within the 1 s ROI interval
    obs = tr.process(frame, now_ms=30)
    assert obs.quad is not None and tr.last_roi_ms == 30  # re-detected at once
