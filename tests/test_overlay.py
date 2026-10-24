"""Headset calibration overlay: Cymple's geometry, the thumbstick steps and the two-step flow."""
import math
import os
import threading
import types

import pytest

from opencymple_linux import overlay as G
from opencymple_linux.calib import MESH_TARGETS

Y, SCALE = -0.067157, 2.274527                       # the user's stored overlay geometry


def flat_eye(tan=1.0, x=0.0):
    return G.EyeView((x, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), -tan, tan, tan, -tan)


def test_dot_positions_follow_cymples_pixels():
    m = 1.65 * SCALE / 896
    assert G.dot_position(0, Y, SCALE) == (0.0, Y, -2.0)
    assert G.dot_position(5, Y, SCALE) == pytest.approx((330 * m, Y, -2.0))         # outer right
    assert G.dot_position(7, Y, SCALE) == pytest.approx((0.0, Y + 330 * m, -2.0))   # outer top
    assert G.dot_position(13 - 1, Y, SCALE)[0] == pytest.approx(int(330 * math.cos(math.radians(315))) * m)
    assert G.ring_radius(SCALE) == pytest.approx(330 * m)
    assert G.OverlayGeometry(Y, SCALE).ring_degrees() == pytest.approx(math.degrees(math.atan(330 * m / 2)))


def test_stick_step():
    assert G.stick_step(0.19, 0.7, 0.05) == 0.0
    assert G.stick_step(1.0, 0.7, 0.2) == pytest.approx(0.7 * 0.05)                  # dt capped at 50 ms
    assert G.stick_step(-0.6, 0.85, 0.02) == pytest.approx(-0.5 * 0.85 * 0.02)
    assert G.stick_step(float("nan"), 0.7, 0.02) == 0.0


def test_max_scale_keeps_dots_in_view():
    assert G.max_scale(0.0, []) == 6.0
    assert G.max_scale(0.0, [flat_eye(0.05)]) == 0.45
    # tan 1 each way: the outer dots may reach tan 0.88 (94 % of the view) -> R <= 1.76 m at 2 m
    limit = 1.76 / G.ring_radius(1.0)
    assert G.max_scale(0.0, [flat_eye()]) == pytest.approx(limit, abs=2e-3)
    assert G.fits(0.0, limit - 0.01, [flat_eye()]) and not G.fits(0.0, limit + 0.01, [flat_eye()])
    # raised by 0.3 m, the top dot runs out first
    raised = G.max_scale(0.3, [flat_eye()])
    assert raised == pytest.approx((1.76 - 0.3) / G.ring_radius(1.0), abs=1e-2)
    # an eye offset sideways (half the IPD) loses room on the far side
    assert G.max_scale(0.0, [flat_eye(x=0.032), flat_eye(x=-0.032)]) < limit


def test_geometry_adjustments_clamp_like_cymple():
    g = G.OverlayGeometry(0.0, 2.0, [flat_eye()])
    assert g.adjust_height(1.0, 0.05) and g.y == pytest.approx(0.035)
    for _ in range(100):
        g.adjust_height(1.0, 0.05)
    assert g.y == 0.75
    assert g.scale <= G.max_scale(0.75, [flat_eye()]) + 1e-9         # coverage re-capped by the height
    assert not g.adjust_height(1.0, 0.05)
    for _ in range(200):
        g.adjust_coverage(1.0, 0.05)
    assert g.scale == pytest.approx(G.max_scale(0.75, [flat_eye()]))
    for _ in range(200):
        g.adjust_coverage(-1.0, 0.05)
    assert g.scale == 0.45
    assert not g.adjust_coverage(0.1, 0.05)                           # inside the dead zone


def test_saved_targets_stay_canonical():
    """sub_180183060 + sub_180182b60 give back the canonical layout whenever the dots are in view,
    which is why the user's saved targets equal it exactly."""
    for extents in ((1.0, 1.0, 1.0, 1.0), (1.2, 0.9, 1.1, 1.4)):
        t = G.fov_targets(Y, SCALE, extents)
        assert t is not None
        for got, want in zip(t, MESH_TARGETS):
            assert got == pytest.approx(want, abs=1e-6)
    assert G.fov_targets(Y, SCALE, (1.0, 0.0, 1.0, 1.0)) is None


# ---- the flow, with stand-ins for OpenXR and the runtime ---------------------------------
@pytest.fixture(scope="module")
def qt():
    pytest.importorskip("PySide6")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"          # never on the real desktop
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class FakeRuntime:
    def __init__(self):
        self.calib = types.SimpleNamespace(overlay_geometry=(Y, SCALE))
        self.calibrator = types.SimpleNamespace(mesh=None)
        self.geometry, self.cancelled, self.started = None, 0, threading.Event()

    def set_overlay_geometry(self, y, s):
        self.geometry = (y, s)

    def start_mesh(self, overlay=None):
        from opencymple_linux.calib import MeshCapture
        self.calibrator.mesh = MeshCapture(overlay)
        self.started.set()

    def capture_mesh_point(self):
        m = self.calibrator.mesh
        k = m.step
        if k == 3 and not getattr(self, "skipped", False):
            self.skipped = True
            return {"point": k, "eyes": [], "done": False, "saved": {}, "skipped": "no reliable eye data"}
        m.raw[0][k] = m.raw[1][k] = (0.0, 0.0)
        m.step += 1
        done = m.step == len(MESH_TARGETS)
        if done:
            self.calibrator.mesh = None
        return {"point": k, "eyes": [0, 1], "done": done, "saved": {0: True, 1: True} if done else {}}

    def cancel_mesh(self):
        self.cancelled += 1
        self.calibrator.mesh = None

    def eye_readiness(self):
        return (True, True)


class FakeXr:
    def __init__(self):
        self.pulses = []

    def pulse(self, hand, amplitude, seconds):
        self.pulses.append(hand)

    def quad(self, sc, x, y, z, w, h):
        return (sc, x, y, z, w, h)


def press(h, x, inp_hand=1, stick=(0.0, 0.0), dt=0.011, now=0.0):
    trig = [0.0, 0.0]
    trig[inp_hand] = 1.0
    h.handle_input(x, (stick[0], stick[1], tuple(trig), (False, False)), dt, now)
    h.handle_input(x, (0.0, 0.0, (0.0, 0.0), (False, False)), dt, now)


def test_flow_adjust_capture_done(qt):
    from opencymple_linux.ui import xr_overlay as X
    rt, x = FakeRuntime(), FakeXr()
    h = X.HeadsetCalibration(rt)
    assert (h.geometry.y, h.geometry.scale) == pytest.approx((Y, SCALE))
    h.handle_input(x, (1.0, -1.0, (0.0, 0.0), (False, False)), 0.05, 0.0)    # left up: bigger, right down
    assert h.geometry.scale > SCALE and h.geometry.y < Y
    press(h, x, inp_hand=0)                                     # left trigger does not accept step 1
    assert h.state == X.ADJUST
    press(h, x, inp_hand=1)
    assert h.state == X.PREPARING and rt.geometry == (h.geometry.y, h.geometry.scale)
    assert rt.started.wait(2)
    h._prep.join(2)
    h._prepared()
    assert h.state == X.CAPTURE and h.current_point() == 0
    assert rt.calibrator.mesh.overlay == rt.geometry
    lines = h.text_lines()
    assert lines[0][0] == "Point 1 of 13"
    quads = h.layout(x, "main", "glow", "text", 0.0)
    assert [q[0] for q in quads] == ["main", "glow", "text"]
    assert quads[1][1:4] == pytest.approx(G.dot_position(0, h.geometry.y, h.geometry.scale)[:2] + (-1.995,))
    for _ in range(3):
        press(h, x)
    press(h, x, inp_hand=0)                                     # point 3 skipped once, either trigger
    assert h.current_point() == 3 and "not captured" in h.message
    while h.state == X.CAPTURE:
        press(h, x, now=10.0)
    assert h.state == X.DONE and h.saved == {0: True, 1: True} and h.done_at == 10.0 + X.DONE_CLOSE_S
    assert "left and right" in h.status
    assert not rt.cancelled


def test_back_returns_to_step_one(qt):
    from opencymple_linux.ui import xr_overlay as X
    rt, x = FakeRuntime(), FakeXr()
    h = X.HeadsetCalibration(rt)
    press(h, x)
    rt.started.wait(2)
    h._prep.join(2)
    h._prepared()
    h.handle_input(x, (0.0, 0.0, (0.0, 0.0), (False, True)), 0.01, 0.0)       # B
    assert h.state == X.ADJUST and rt.cancelled == 1
    h.handle_input(x, (0.0, 0.0, (0.0, 0.0), (False, True)), 0.01, 0.0)       # still held: no new press
    assert not h._stop.is_set()
    h.handle_input(x, (0.0, 0.0, (0.0, 0.0), (False, False)), 0.01, 0.0)
    h.handle_input(x, (0.0, 0.0, (0.0, 0.0), (True, False)), 0.01, 0.0)       # Y in step 1 closes
    assert h._stop.is_set()


def test_textures_paint(qt):
    from opencymple_linux.ui import xr_overlay as X
    img = X.paint_main(current=4, captured={0, 1, 2, 3})
    assert img.width() == X.MAIN_PX
    px, py = X.dot_pixel(4)
    assert img.pixelColor(int(px), int(py) - 8).red() > 200           # the yellow current dot
    assert img.pixelColor(2, 2).alpha() == 0                          # transparent outside the disc
    assert X.paint_text([("Point 5 of 13", X.TEXT, 46)]).pixelColor(X.TEXT_W // 2, 10).alpha() > 0
    assert X.paint_glow().pixelColor(X.GLOW_PX // 2, X.GLOW_PX // 2).alpha() == 0
