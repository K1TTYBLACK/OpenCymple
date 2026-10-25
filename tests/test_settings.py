"""Our settings: all defaults on a fresh install, written only to our own directory."""
import json
import os

import pytest

from opencymple import calibration as C
from opencymple import settings as S

INI = """[ui]
themeMode = dark

[OSC]
eyeGazeFilter = 0.470000
eyePitchFlip = false

[Cymple_Mouth]
filer = 0.000000
rotation = -180.000000
"""


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCYMPLE_CONFIG", str(tmp_path / "cfg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CYMPLE_MODELS", raising=False)
    return tmp_path / "cfg"


def test_fresh_install_is_all_defaults(cfg):
    ini = S.resolve_ini()
    assert ini == str(cfg / "iniFile.ini") and open(ini).read() == ""
    assert not (cfg / "avatarParaments").exists() and not (cfg / "opencymple.json").exists()
    cal = C.load(ini)
    assert cal.eye_gaze_filter == 0.8 and cal.mouth.filter == 0.25
    assert all(e.roi_quad is None and e.mesh is None for e in cal.eyes)
    assert cal.amps == {}
    assert S.load_options()["osc"]["target"] == "127.0.0.1:9000"
    assert S.OPTIONS_NAME == "opencymple.json"


def test_models_dir_lookup(cfg, tmp_path, monkeypatch):
    assert S.find_model_dir() is None
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "model0.bin").write_bytes(b"x")
    opts = S.load_options()
    opts["models_dir"] = str(tmp_path / "c")
    S.save_options(opts)
    assert S.find_model_dir() == str(tmp_path / "c")
    monkeypatch.setenv("CYMPLE_MODELS", str(tmp_path / "b"))
    assert S.find_model_dir() == str(tmp_path / "b")
    assert S.find_model_dir(str(tmp_path / "a")) == str(tmp_path / "a")
    assert S.find_model_dir(str(tmp_path)) == str(tmp_path / "b")        # no model0.bin there: next candidate
    monkeypatch.delenv("CYMPLE_MODELS")
    monkeypatch.setattr(S, "load_options", lambda: {})
    bundled = tmp_path / "app" / "usr" / "share" / "opencymple" / "model"
    bundled.mkdir(parents=True)
    (bundled / "model0.bin").write_bytes(b"x")
    monkeypatch.setenv("APPDIR", str(tmp_path / "app"))
    assert S.find_model_dir() == str(bundled)


def test_writer_writes_our_ini(cfg):
    ini = S.resolve_ini()
    w = S.IniWriter(ini)
    w.set("OSC", "eyeGazeFilter", 0.5)
    w.set("OSC", "lidSync", True)
    w.set("Cymple_Mouth", "v1_auto_roi_calibration", False)
    assert w.flush()
    osc = C.read_ini(ini)
    assert osc["OSC"]["eyeGazeFilter"] == "0.500000" and osc["OSC"]["lidSync"] == "true"
    assert osc["Cymple_Mouth"]["v1_auto_roi_calibration"] == "false"
    assert C.load(ini).switches[9] is True


def test_amp_sets(cfg, tmp_path):
    cfg = str(cfg)
    S.resolve_ini()
    S.save_amps(cfg, None, {"Amp_JawOpenAmp": 1.5})
    assert not S.has_own_amps(cfg, "avtr_x") and not S.has_own_amps(cfg, None)
    S.save_amps(cfg, "avtr_x", {"Amp_JawOpenAmp": 2.0})
    assert S.has_own_amps(cfg, "avtr_x")
    assert C.load_avatar_amps(os.path.join(cfg, "avatarParaments"), "avtr_x") == {"Amp_JawOpenAmp": 2.0}
    S.delete_amps(cfg, "avtr_x")
    assert C.load_avatar_amps(os.path.join(cfg, "avatarParaments"), "avtr_x") == {"Amp_JawOpenAmp": 1.5}
    with pytest.raises(ValueError):
        S.save_amps(cfg, "../evil", {})


def test_options_merge(cfg):
    opts = S.load_options()
    assert opts["osc"]["target"] == "127.0.0.1:9000"
    opts["osc"]["enabled"] = False
    S.save_options(opts)
    again = S.load_options()
    assert again["osc"] == {"enabled": False, "target": "127.0.0.1:9000", "avatar": "auto", "config_dir": ""}
    assert (cfg / "opencymple.json").exists()
    assert again["afk"] == {"enabled": False}


def test_update_ini_keeps_crlf(tmp_path):
    p = tmp_path / "iniFile.ini"
    p.write_bytes(INI.replace("\n", "\r\n").encode())
    C.update_ini(str(p), "OSC", {"eyeGazeFilter": "0.600000", "newKey": "1"}, backup=False)
    data = p.read_bytes()
    assert data.count(b"\r\n") == data.count(b"\n") and b"eyeGazeFilter = 0.600000\r\n" in data
    assert C.read_ini(str(p))["OSC"]["newKey"] == "1"


def test_bundle_roundtrip(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "avatarParaments").mkdir(parents=True)
    (a / "iniFile.ini").write_bytes(INI.replace("\n", "\r\n").encode())
    (a / "avatarParaments" / "avtr_x.json").write_text('{"OSC": {"Amp_JawOpenAmp": 2.0}}')
    out = tmp_path / "cfg.json"
    S.export_bundle(str(out), str(a / "iniFile.ini"))
    data = json.load(open(out))
    assert data["avatarParaments"]["avtr_x"] == {"OSC": {"Amp_JawOpenAmp": 2.0}}
    assert not any("cymple" in k.lower() for k in data["settings"])
    assert "ini" not in data and isinstance(data["settings"], dict)
    b.mkdir()
    (b / "iniFile.ini").write_text("[OSC]\nold = 1\n")
    S.import_bundle(str(out), str(b))
    assert C.read_ini(str(b / "iniFile.ini")) == C.read_ini(str(a / "iniFile.ini")) | {"OSC": {"old": "1", **C.read_ini(str(a / "iniFile.ini"))["OSC"]}}
    assert json.load(open(b / "avatarParaments" / "avtr_x.json")) == {"OSC": {"Amp_JawOpenAmp": 2.0}}
    assert any(n.startswith("iniFile.ini.bak-") for n in os.listdir(b))
