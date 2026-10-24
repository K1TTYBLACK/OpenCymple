"""OpenCymple as an app: desktop entry, single-instance commands, tray shell, background start.
The shell and background tests need the local install, the capture and PySide6; they never send
OSC (the user's VRChat may be running) and use their own socket and config directories."""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time

import pytest

from opencymple_linux import calibration as C
from opencymple_linux import desktop, instance
from opencymple_linux import opencymple as O

HERE = os.path.dirname(__file__)
CAPTURE = os.path.join(HERE, "..", "captures", "live-test-1.cap")
INI = C.find_ini()
MODELS = C.find_model_dir(INI) if INI else None
needs_install = pytest.mark.skipif(not (INI and MODELS and os.path.exists(CAPTURE)),
                                   reason="needs the Cymple install and captures/live-test-1.cap")


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    for var in ("XDG_RUNTIME_DIR", "XDG_DATA_HOME", "XDG_STATE_HOME"):
        (tmp_path / var).mkdir()
        monkeypatch.setenv(var, str(tmp_path / var))
    monkeypatch.setenv("OPENCYMPLE_LINUX_CONFIG", str(tmp_path / "cfg"))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    return tmp_path


def test_install_and_uninstall(dirs, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(dirs / "home"))            # ~/.local/bin for the command link
    assert O.main(["install"]) == 0
    entry, icon = desktop.entry_path(), desktop.installed_icon_path()
    assert entry.startswith(str(dirs / "XDG_DATA_HOME")) and os.path.isfile(icon)
    text = open(entry).read()
    assert "Name=OpenCymple" in text and "Categories=Utility;X-WiVRn-VR;" in text
    assert "Icon=opencymple" in text and "Exec=%s" % desktop.executable()[0] in text
    assert "[Desktop Action calibrate]" in text and "calibrate precise" in text
    if shutil.which("desktop-file-validate"):
        r = subprocess.run(["desktop-file-validate", entry], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
    link = desktop.link_path()
    assert link.startswith(str(dirs / "home")) and os.path.realpath(link) == os.path.realpath(desktop.executable()[0])
    assert O.main(["install"]) == 0                                       # again: link replaced, no error
    assert O.main(["uninstall"]) == 0
    assert not os.path.exists(entry) and not os.path.exists(icon) and not os.path.lexists(link)
    os.makedirs(os.path.dirname(link), exist_ok=True)
    open(link, "w").close()                                               # someone else's file stays
    O.main(["install"])
    assert not os.path.islink(link)
    O.main(["uninstall"])
    assert os.path.isfile(link)


def test_exec_quoting():
    assert desktop._quote("/usr/bin/opencymple") == "/usr/bin/opencymple"
    assert desktop._quote("/opt/my apps/opencymple") == '"/opt/my apps/opencymple"'
    assert desktop._quote('a"b$c') == '"a\\"b\\$c"'


def test_commands_without_an_app(dirs, capsys):
    assert O.main(["status"]) == 1 and O.main(["quit"]) == 1
    assert "not running" in capsys.readouterr().out
    assert instance.send({"cmd": "ping"}) is None


# ---- the shell in this process -----------------------------------------------------------
def _pump(app, seconds, until=None):
    end = time.monotonic() + seconds
    while time.monotonic() < end and not (until and until()):
        app.processEvents()
        time.sleep(0.01)


def _ask(app, request, timeout=10.0):
    """Send from a thread while the GUI thread pumps events (the shell answers there)."""
    out = {}
    t = threading.Thread(target=lambda: out.update(reply=instance.send(request, timeout=timeout)))
    t.start()
    _pump(app, timeout, until=lambda: not t.is_alive())
    t.join(1)
    return out.get("reply")


@needs_install
def test_shell_commands(dirs, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from opencymple_linux import settings as S
    from opencymple_linux.runtime import Runtime
    from opencymple_linux.ui import shell as SH
    from opencymple_linux.ui import xr_overlay
    app = QApplication.instance() or QApplication([])
    cym = dirs / "Cymple"
    shutil.copytree(os.path.dirname(INI), cym, ignore=shutil.ignore_patterns("logs", "dataset"))
    monkeypatch.setattr(C, "find_ini", lambda: str(cym / "iniFile.ini"))
    monkeypatch.setattr(xr_overlay, "available", lambda: "no headset in tests")
    monkeypatch.setattr(QApplication, "quit", staticmethod(lambda: None))
    rt = Runtime(S.resolve_ini(), MODELS)
    rt.start()
    rt.start_source(replay=CAPTURE, loop=True)
    args = argparse.Namespace(no_osc=True, osc_listen=0, headless=True)
    shell = SH.Shell(rt, args, S.load_options())
    try:
        assert rt.vrchat is None and shell.window is None
        assert _ask(app, {"cmd": "ping"})["ok"]
        _pump(app, 1.5)
        st = _ask(app, {"cmd": "status"})
        assert st["ok"] and len(st["eyes"]) == 2 and any(st["eyes"]), st
        rep = _ask(app, {"cmd": "calibrate", "mode": "precise"})
        assert not rep["ok"] and "no headset in tests" in rep["message"]
        rep = _ask(app, {"cmd": "recenter"}, timeout=15)       # ok unless the replay is mid-blink
        assert rep["message"].startswith("recenter") and rep["ok"] == ("done" in rep["message"]), rep
        rep = _ask(app, {"cmd": "calibrate", "mode": "coarse"})
        assert rep["ok"] and rt.calibrator.coarse is not None and not rt.calib.precise_gaze
        rt.stop_coarse()
        rt.set_precise(True)
        assert _ask(app, {"cmd": "show"})["ok"] and shell.window is not None
        win = shell.window
        _pump(app, 0.5)
        win.close()
        _pump(app, 0.2)
        assert shell.window is None and rt.pipeline is not None
        assert not _ask(app, {"cmd": "nonsense"})["ok"]
        assert _ask(app, {"cmd": "quit"})["ok"]
        _pump(app, 0.5, until=lambda: shell.quitting)
        assert shell.quitting
    finally:
        if not shell.quitting:
            shell.quit()


@needs_install
def test_command_starts_a_background_app(dirs):
    """`opencymple recenter` with nothing running: a headless app starts, recenters and stays."""
    pytest.importorskip("PySide6")
    py = [sys.executable, "-m", "opencymple_linux.opencymple"]
    env = dict(os.environ)
    try:
        r = subprocess.run(py + ["--no-osc", "--replay", CAPTURE, "--loop", "recenter"], env=env,
                           capture_output=True, text=True, timeout=90)
        # the replayed eyes may be mid-blink: either way the background app answered the recenter
        assert "starting OpenCymple in the background" in r.stdout, r.stdout + r.stderr
        assert r.stdout.splitlines()[-1].startswith("recenter:"), r.stdout + r.stderr
        r = subprocess.run(py + ["status"], env=env, capture_output=True, text=True, timeout=20)
        assert r.returncode == 0, r.stdout
    finally:
        r = subprocess.run(py + ["quit"], env=env, capture_output=True, text=True, timeout=20)
    assert r.returncode == 0 and "stopped" in r.stdout
    deadline = time.monotonic() + 15
    while instance.running() and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not instance.running()
