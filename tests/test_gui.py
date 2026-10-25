"""The window and Runtime on a replayed capture: every change reaches the pipeline and our INI,
Cymple's files stay as they were. Needs the local install, the capture and PySide6."""
import argparse
import hashlib
import json
import logging
import os
import shutil
import socket
import time

import pytest

pytest.importorskip("PySide6")
os.environ["QT_QPA_PLATFORM"] = "offscreen"          # never on the real desktop, whatever the session sets

from opencymple_linux import calibration as C                       # noqa: E402
from opencymple_linux import settings as S                          # noqa: E402
from opencymple_linux.gaze import gaze_filter_params                # noqa: E402

HERE = os.path.dirname(__file__)
CAPTURE = os.path.join(HERE, "..", "captures", "live-test-1.cap")
INI = C.find_ini()
MODELS = C.find_model_dir(INI) if INI else None
pytestmark = pytest.mark.skipif(not (INI and MODELS and os.path.exists(CAPTURE)),
                                reason="needs the Cymple install and captures/live-test-1.cap")


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """Runtime on our settings copy, imported from a copy of the real Cymple INI."""
    from opencymple_linux.runtime import Runtime
    cym = tmp_path / "Cymple"
    shutil.copytree(os.path.dirname(INI), cym, ignore=shutil.ignore_patterns("logs", "dataset"))
    monkeypatch.setattr(C, "find_ini", lambda: str(cym / "iniFile.ini"))
    monkeypatch.setenv("OPENCYMPLE_LINUX_CONFIG", str(tmp_path / "cfg"))
    before = _md5(cym / "iniFile.ini")
    rt = Runtime(S.resolve_ini(), MODELS)
    rt.start()
    rt.start_source(replay=CAPTURE, loop=True)
    yield rt
    rt.stop()
    assert _md5(cym / "iniFile.ini") == before


def _pump(app, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def test_window_changes_apply_and_save(runtime):
    from PySide6.QtWidgets import QApplication
    from opencymple_linux.ui.app import MainWindow
    app = QApplication.instance() or QApplication([])
    rt = runtime
    win = MainWindow(rt, argparse.Namespace(no_osc=True, osc_listen=0), S.load_options())
    try:
        _pump(app, 2.0)
        panel = win.eye_panels[0]
        assert panel.raw.image is not None and panel.raw.quad is not None and len(panel.crop.points) == 8
        assert win.mouth_panel.crop.image is not None
        assert "received" in panel.stats.text()

        s = win.eye_sliders["eye_gaze_filter"]
        s.slider.setValue(s.slider.maximum())                     # 1.0
        assert rt.calib.eye_gaze_filter == pytest.approx(1.0)
        assert rt.pipeline.face.gaze.filters[0].min_cutoff == pytest.approx(gaze_filter_params(1.0)[0])

        win._mouth_rect_drawn(0.2, 0.3, 0.6, 0.8)                 # drag on the mouth preview
        m = rt.calib.mouth
        assert not m.auto_roi and not win.auto_roi.isChecked()
        assert (m.upper, m.bottom, m.width, m.x_offset) == pytest.approx((0.3, 0.8, 0.4, 0.4))
        _pump(app, 0.5)
        obs = rt.pipeline.snapshot()["mouth"]
        assert obs.quad is None and obs.rect is not None

        assert not win.afk_toggle.isChecked() and not rt.pipeline.afk.enabled
        assert any(rt.pipeline.snapshot()["mouth_fields"].values())
        win.afk_toggle.setChecked(True)                           # live-test-1 was recorded with the headset off
        assert rt.pipeline.afk.enabled and S.load_options()["afk"]["enabled"]
        _pump(app, 1.5)
        snap = rt.pipeline.snapshot()
        assert rt.pipeline.afk.away and not any(snap["mouth_fields"].values()) and not any(snap["gaze"].values())
        assert not win.afk_status.isHidden()
        win.afk_toggle.setChecked(False)
        _pump(app, 0.5)
        assert not rt.pipeline.afk.away and any(rt.pipeline.snapshot()["mouth_fields"].values())
        assert win.afk_status.isHidden()

        win.amp_sliders[14].spin.setValue(2.0)                    # "Smile" -> slot 14
        assert rt.calib.amps["Amp_ShrugLowerAmp"] == 2.0 and rt.pipeline.mouth_core.slots[14] == 2.0

        rt.calib.eyes[0].roi_quad = None
        win._calibrate_roi((0, 1))
        assert rt.calib.eyes[0].roi_quad is not None and rt.pipeline.workers[0].tracker.roi_quad is not None

        rt.flush()
        ours = C.read_ini(rt.ini)
        assert ours["OSC"]["eyeGazeFilter"] == "1.000000"
        assert ours["Cymple_Mouth"]["v1_auto_roi_calibration"] == "false"
        assert ours["CympleFace_LeftEye"]["RoiQuadValid"] == "true"
        with open(os.path.join(rt.avatar_dir, "avatarParaments", "default.json")) as f:
            assert json.load(f)["OSC"]["Amp_ShrugLowerAmp"] == 2.0
    finally:
        win.close()


def test_osc_output_and_avatar_amplitudes(runtime):
    from opencymple_linux.outputs.avatar import AvatarInfo
    rt = runtime
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(3.0)
    rt.start_osc("127.0.0.1:%d" % sock.getsockname()[1], avatar="none", listen_port=0)
    try:
        assert sock.recvfrom(65536)[0].startswith(b"#bundle")
    finally:
        rt.stop_osc()
        sock.close()

    class Watcher:                                        # stands in for AvatarWatcher
        version, current = 1, AvatarInfo("avtr_test", "Test")

        def stop(self):
            pass

    rt.watcher = Watcher()
    deadline = time.monotonic() + 3
    while rt.avatar_id != "avtr_test" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert rt.avatar_id == "avtr_test" and not rt.amps_own()
    rt.create_own_amps()
    rt.set_amp(0, 3.0)
    rt.flush()
    av = os.path.join(rt.avatar_dir, "avatarParaments")
    with open(os.path.join(av, "avtr_test.json")) as f:
        assert json.load(f)["OSC"]["Amp_NoseSneerAmp"] == 3.0
    assert C.load_avatar_amps(av).get("Amp_NoseSneerAmp", 1.0) != 3.0     # default set untouched
    rt.delete_own_amps()
    assert not rt.amps_own() and rt.amp_values()[0] != 3.0
    rt.watcher = None


def test_calibration_flows_on_replay(runtime):
    from PySide6.QtWidgets import QApplication
    from opencymple_linux import calib as K
    from opencymple_linux.ui.calibration import MeshTargetWindow
    app = QApplication.instance() or QApplication([])
    rt = runtime
    _pump(app, 1.5)

    rep = rt.recenter(wait=True)
    assert set(rep) == {0, 1}
    assert any(isinstance(r, dict) for r in rep.values()), rep
    ours = C.read_ini(rt.ini)["OSC"]
    for eye, r in rep.items():
        if isinstance(r, dict):
            side = ("Left", "Right")[eye]
            e = rt.calib.eyes[eye]
            assert ours["%s_Eyelid_ref_center_upper_outer_y" % side] == "%.6f" % e.lid_refs[1][1][0]
            assert ours["%s_Eyepupil_x_mid" % side] == "%.6f" % e.range_yaw[1]

    rt.set_precise(False)
    assert rt.pipeline.face.gaze.precise is False and rt.pipeline.face.gaze.mesh_valid == (False, False)
    assert rt.start_coarse()
    _pump(app, 1.0)
    rt.calibrator.coarse["until"] = 0.0                  # let the packet timer end it now
    _pump(app, 0.3)
    assert rt.calibrator.coarse is None and rt.calib_status.startswith("coarse calibration finished")
    rt.set_precise(True)

    win = MeshTargetWindow(rt)
    win.show()
    for _ in K.MESH_TARGETS:
        _pump(app, 0.1)
        win._capture()
    assert win.done and rt.calibrator.mesh is None
    saved = [e for e in (0, 1) if rt.calib.eyes[e].mesh_valid]
    assert saved, rt.calib_status
    ours = C.read_ini(rt.ini)["OSC"]
    assert ours["Left_Eyegaze_mesh_layout_version"] == "4" and ours["Left_Eyegaze_feature_mode"] == "2"
    win.close()


def test_headset_flow_on_replay(runtime):
    """The overlay's steps without OpenXR: geometry from the sticks, captures from the trigger, the
    mesh saved with that geometry."""
    from PySide6.QtWidgets import QApplication
    from opencymple_linux.ui import xr_overlay as X
    app = QApplication.instance() or QApplication([])
    rt = runtime
    _pump(app, 1.5)

    class Xr:
        def pulse(self, *_):
            pass
    x = Xr()
    h = X.HeadsetCalibration(rt)
    h.handle_input(x, (1.0, 1.0, (0.0, 0.0), (False, False)), 0.05, 0.0)

    def trigger():
        h.handle_input(x, (0.0, 0.0, (0.0, 1.0), (False, False)), 0.01, 0.0)
        h.handle_input(x, (0.0, 0.0, (0.0, 0.0), (False, False)), 0.01, 0.0)
    trigger()
    h._prep.join(10)
    h._prepared()
    assert h.state == X.CAPTURE
    for _ in range(40):
        if h.state != X.CAPTURE:
            break
        _pump(app, 0.1)
        trigger()
    assert h.state == X.DONE, h.message
    y, s = h.geometry.y, h.geometry.scale
    ours = C.read_ini(rt.ini)["OSC"]
    assert ours["eye_gaze_calibration_overlay_y_offset_meters"] == "%.6f" % y
    assert ours["eye_gaze_calibration_overlay_ring_scale"] == "%.6f" % s
    for eye, ok in h.saved.items():
        if ok:
            side = ("Left", "Right")[eye]
            assert ours["%s_Eyegaze_mesh_overlay_ring_scale" % side] == "%.6f" % s
            assert ours["%s_Eyegaze_mesh_overlay_y_offset_meters" % side] == "%.6f" % y


def test_window_decks_previews_and_log_export(runtime, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QPushButton, QToolButton
    from opencymple_linux import logs
    from opencymple_linux.ui import app as ui
    from opencymple_linux.ui.app import MainWindow
    app = QApplication.instance() or QApplication([])
    win = MainWindow(runtime, argparse.Namespace(no_osc=True, osc_listen=0), S.load_options())
    try:
        _pump(app, 1.0)
        # upper deck: the pages in order, Output among them; lower deck: Logs and Settings
        assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == [
            "Values", "Eyes", "Mouth", "Amplitudes", "Output", "Logs", "Settings"]
        navs = [b.text() for b in win.findChildren(QToolButton, "nav") if not b.isHidden()]
        assert navs == ["Values", "Eyes", "Mouth", "Amplitudes", "Output", "Logs", "Settings"]
        # eyes show the model input only, the mouth the camera feed only
        assert win.eye_panels[0].raw.isHidden() and not win.eye_panels[0].crop.isHidden()
        assert win.mouth_panel.crop.isHidden() and not win.mouth_panel.raw.isHidden()
        # narrow window: the previews become a tab in front of Output
        win._place_previews(True)
        assert win.tabs.indexOf(win.previews_area) == win.tabs.indexOf(win.output_page) - 1
        win._place_previews(False)
        assert {"Import From", "Export To", "Export Logs"} <= {b.text() for b in win.findChildren(QPushButton)}

        logs.attach()
        try:
            logging.getLogger("test_gui").warning("hello from the window test")
            win._refresh_slow(win.rt.pipeline.snapshot())
            assert "hello from the window test" in win.log_view.toPlainText()

            out = tmp_path / "exported.log"

            class SaveDialog:
                @staticmethod
                def getSaveFileName(*_args, **_kwargs):
                    return str(out), ""
            monkeypatch.setattr(ui, "QFileDialog", SaveDialog)
            win._export_logs()
            assert "hello from the window test" in out.read_text(encoding="utf-8")
        finally:
            logging.getLogger().removeHandler(logs.BUFFER)
    finally:
        win.close()
