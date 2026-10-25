"""Calibration flows (calib.py) on synthetic landmarks, plus the relations seen in a saved Cymple INI."""
import math

import pytest

from opencymple import calib as K
from opencymple import calibration as C
from opencymple.calibration import Calibration
from opencymple.face import EyeCore
from opencymple.gaze import feature_mode2, mesh_map


class Writer:
    def __init__(self):
        self.keys = {}

    def set(self, section, key, value):
        self.keys[(section, key)] = value


def eye_landmarks(yaw=0.0, pitch=0.0, opening=0.2, ref_y=0.5):
    """An open eye in the 96x96 crop; the pupil moves against pt7 like the model's (raw yaw ~ -dx)."""
    up, lo = 0.5 - opening / 2, 0.5 + opening / 2
    return [(0.2, 0.5), (0.8, 0.5), (0.4, up), (0.6, up), (0.4, lo), (0.6, lo),
            (0.5 - 0.6 * yaw * 0.15, 0.5 - 0.6 * 0.42 * pitch * 0.15), (0.5, ref_y)]


def feed(core, t, yaw=0.0, pitch=0.0, expr=(0.3, 0.1, 0.2), **kw):
    for eye in (0, 1):
        core.on_landmarks(eye, eye_landmarks(yaw, pitch, **kw), expr, t)


def make_core(precise=True, mesh=True):
    cal = Calibration(precise_gaze=precise)
    for e in cal.eyes:
        e.mesh = [(0.1 * t[0], 0.1 * t[1], t[0], t[1]) for t in K.MESH_TARGETS] if mesh else None
        e.mesh_valid = mesh
        e.squint_values = (0.0, 0.0, 1.0)
    return EyeCore(cal)


def test_pure_helpers():
    assert K.lid_values(0.202438, 0.21) == pytest.approx((0.16195, 0.042512, 0.202438), abs=1e-6)
    assert K.lid_values(0.0, 0.21)[2] == pytest.approx(K.EPS / 0.8 * 0.8)
    lids = (0.4, 0.42, 0.6, 0.61)
    assert K.remap_lids(lids, lids, lids) == pytest.approx(lids)
    moved = K.remap_lids((0.3, 0.32, 0.7, 0.71), lids, tuple(v + 0.05 for v in lids))
    assert moved == pytest.approx((0.35, 0.37, 0.75, 0.76))
    assert K.sanitize_range((0.0, 0.5, 1.0)) == (-1.0, 0.0, 1.0)            # "unset" default
    assert K.sanitize_range((0.3, 0.1, 0.5)) == pytest.approx((0.3, 0.4, 0.5))
    r = K.recenter_range((-0.2, 0.0, 0.3), 0.1, 0.08)
    assert r == pytest.approx((-0.1, 0.1, 0.4))                             # spans kept, mid moved
    assert K.recenter_range((-0.01, 0.0, 0.01), 0.5, 0.08) == pytest.approx((0.42, 0.5, 0.58))
    n, mask = K.neutral_update((0.5, 0.99995, float("nan")), (0.1, 0.2, 0.3))
    assert n == (0.5, 0.2, 0.3) and mask == 1


def test_finalize_range():
    backup = (-0.3, 0.0, 0.3)
    assert K.finalize_range((-0.5, 0.2, 0.6), backup) == ((-0.5, 0.0, 0.6), True)   # old mid kept
    assert K.finalize_range((-0.05, 0.0, 0.5), backup) == ((-0.3, 0.0, 0.5), True)  # short side -> backup
    assert K.finalize_range((-0.05, 0.0, 0.05), backup) == (backup, False)
    assert K.finalize_range((float("nan"), 0.0, 0.5), backup) == (backup, False)


def test_recenter_shifts_mesh_and_sets_references():
    core = make_core()
    w = Writer()
    cal = K.EyeCalibrator(core, w)
    feed(core, 1.0, yaw=0.3, pitch=-0.2, expr=(0.4, 0.05, 0.3))       # off the old centre: the mesh moves
    core.build(1.0)
    out = cal.recenter(1.01)
    assert out[0]["gaze"].startswith("mesh shifted")
    assert set(out) == {0, 1} and all(isinstance(v, dict) for v in out.values())
    for eye, e in enumerate(core.cal.eyes):
        f = feature_mode2(core.lm[eye])
        assert e.mesh[0][:2] == pytest.approx(f)                         # centre point = current feature
        assert mesh_map(*f, e.mesh) == pytest.approx((0.0, 0.0), abs=1e-6)
        assert e.range_yaw[1] == pytest.approx(f[0]) and e.range_pitch[1] == pytest.approx(f[1])
        assert e.neutral == pytest.approx((0.4, 0.05, 0.3))
        assert e.lid_values == pytest.approx((0.16, 0.04, 0.2), abs=1e-6)   # opening 0.2, eye closing 0.2
        assert e.center_refs == e.lid_refs[1][1]
    side = "Left_Eye"
    for key in (side + "gaze_mesh_0_raw_yaw", side + "pupil_x_mid", side + "expression_neutral_widen",
                "vrcftLeftLidvalue2", "squintLeftvalue1", "eyeWideLeftvalue0", side + "upper_outer_y",
                side + "lid_ref_center_yaw"):
        assert ("OSC", key) in w.keys, key


def test_recenter_skips_closed_and_stale_eyes():
    core = make_core()
    cal = K.EyeCalibrator(core, Writer())
    feed(core, 1.0, opening=0.01)                                        # nearly closed
    core.build(1.0)
    assert cal.recenter(1.01) == {0: "eye is closed or half closed", 1: "eye is closed or half closed"}
    assert cal.recenter(5.0) == {0: "no fresh eye input", 1: "no fresh eye input"}


def test_recenter_coarse_mode_moves_range_mid():
    core = make_core(precise=False, mesh=False)
    for e in core.cal.eyes:
        e.range_yaw, e.range_pitch = (-0.2, 0.0, 0.2), (-0.1, 0.0, 0.1)
    cal = K.EyeCalibrator(core, Writer())
    feed(core, 1.0, yaw=0.5)
    core.build(1.0)
    cal.recenter(1.01)
    lm = core.lm[0]
    fx = lm[7][0] - lm[6][0]                                             # mode 0 in coarse mode
    assert core.cal.eyes[0].range_yaw == pytest.approx((fx - 0.2, fx, fx + 0.2))


def test_coarse_flow_grows_ranges():
    core = make_core(precise=False, mesh=False)
    for e in core.cal.eyes:
        e.range_yaw, e.range_pitch = (-0.02, 0.0, 0.02), (-0.02, 0.0, 0.02)
    cal = K.EyeCalibrator(core, Writer())
    core.calibrator = cal
    cal.start_coarse(0.0)
    assert core.cal.eyes[0].range_yaw == (0.0, 0.0, 0.0)
    t = 0.0
    for yaw, pitch in ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)):
        t += 0.1
        feed(core, t, yaw=yaw, pitch=pitch)
        core.build(t)
    out = cal.stop_coarse()
    r = core.cal.eyes[0].range_yaw
    assert out[0]["changed"] and r[0] < -0.08 and r[2] > 0.08 and r[1] == 0.0


def test_mesh_flow_saves_both_eyes():
    core = make_core(mesh=False)
    w = Writer()
    cal = K.EyeCalibrator(core, w)
    cal.start_mesh(overlay=(-0.05, 2.0))
    t = 1.0
    for k, (ty, tp) in enumerate(K.MESH_TARGETS):
        t += 0.5
        feed(core, t, yaw=0.8 * ty, pitch=0.8 * tp)
        core.build(t)
        rep = cal.capture_mesh_point(t)
        assert rep["point"] == k and rep["eyes"] == [0, 1]
    assert rep["done"] and rep["saved"] == {0: True, 1: True} and cal.mesh is None
    e = core.cal.eyes[0]
    assert e.mesh_valid and e.feature_mode == 2 and core.gaze.mesh_valid == (True, True)
    for p in e.mesh:
        assert mesh_map(p[0], p[1], e.mesh) == pytest.approx(p[2:], abs=1e-6)
    assert e.lid_refs[0][0] < e.lid_refs[1][0] < e.lid_refs[2][0] or e.lid_refs[0][0] > e.lid_refs[2][0]
    assert w.keys[("OSC", "Left_Eyegaze_mesh_overlay_ring_scale")] == 2.0
    assert w.keys[("OSC", "Right_Eyegaze_mesh_12_target_pitch")] == pytest.approx(-math.sqrt(0.5))


def test_mesh_point_needs_eye_data():
    core = make_core(mesh=False)
    cal = K.EyeCalibrator(core, Writer())
    cal.start_mesh()
    rep = cal.capture_mesh_point(1.0)
    assert rep["skipped"] == "no reliable eye data" and cal.mesh.step == 0


INI = C.find_ini()


@pytest.mark.skipif(not INI, reason="no Cymple INI")
def test_saved_calibration_relations():
    cal = C.load(INI)
    for e in cal.eyes:
        ry, rp = K.ranges_from_mesh(e.mesh)
        assert ry == pytest.approx(e.range_yaw, abs=1e-6) and rp == pytest.approx(e.range_pitch, abs=1e-6)
        assert e.lid_values == pytest.approx(K.lid_values(e.lid_values[2], cal.eye_blink_close_amplitude), abs=2e-6)
        assert e.wide_values[1] == pytest.approx(1.2 * e.wide_values[0], abs=2e-6)
        assert [v for p in e.mesh for v in p[2:]] == pytest.approx([v for t in K.MESH_TARGETS for v in t], abs=1e-6)
