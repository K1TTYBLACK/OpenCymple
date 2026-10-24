"""Our settings copy: import once, write only our INI, export back with backups."""
import json
import os

import pytest

from cymple_linux import calibration as C
from cymple_linux import settings as S

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
def cymple(tmp_path, monkeypatch):
    d = tmp_path / "Cymple"
    (d / "avatarParaments").mkdir(parents=True)
    (d / "iniFile.ini").write_text(INI)
    (d / "avatarParaments" / "default.json").write_text(json.dumps({"OSC": {"Amp_JawOpenAmp": 1.5}}))
    monkeypatch.setattr(C, "find_ini", lambda: str(d / "iniFile.ini"))
    monkeypatch.setenv("CYMPLE_LINUX_CONFIG", str(tmp_path / "cfg"))
    return d


def test_import_once_then_keep(cymple, tmp_path):
    ini = S.resolve_ini()
    assert ini == str(tmp_path / "cfg" / "iniFile.ini")
    assert open(ini).read() == INI
    assert (tmp_path / "cfg" / "avatarParaments" / "default.json").exists()
    assert S.load_options()["imported_from"] == str(cymple / "iniFile.ini")
    (cymple / "iniFile.ini").write_text(INI.replace("0.470000", "0.900000"))
    assert C.read_ini(S.resolve_ini())["OSC"]["eyeGazeFilter"] == "0.470000"     # not re-imported
    assert S.resolve_ini(own=False) == str(cymple / "iniFile.ini")


def test_writer_changes_only_our_copy(cymple):
    ini = S.resolve_ini()
    w = S.IniWriter(ini)
    w.set("OSC", "eyeGazeFilter", 0.5)
    w.set("OSC", "lidSync", True)
    w.set("Cymple_Mouth", "v1_auto_roi_calibration", False)
    assert w.flush()
    osc = C.read_ini(ini)
    assert osc["OSC"]["eyeGazeFilter"] == "0.500000" and osc["OSC"]["lidSync"] == "true"
    assert osc["Cymple_Mouth"]["v1_auto_roi_calibration"] == "false"
    assert open(cymple / "iniFile.ini").read() == INI
    assert C.load(ini).switches[9] is True


def test_writer_refuses_cymples_ini(cymple):
    w = S.IniWriter(str(cymple / "iniFile.ini"))
    assert w.readonly
    w.set("OSC", "eyeGazeFilter", 0.5)
    assert not w.flush()
    assert open(cymple / "iniFile.ini").read() == INI


def test_amp_sets(cymple, tmp_path):
    cfg = str(tmp_path / "cfg")
    S.resolve_ini()
    assert not S.has_own_amps(cfg, "avtr_x") and not S.has_own_amps(cfg, None)
    S.save_amps(cfg, "avtr_x", {"Amp_JawOpenAmp": 2.0})
    assert S.has_own_amps(cfg, "avtr_x")
    assert C.load_avatar_amps(os.path.join(cfg, "avatarParaments"), "avtr_x") == {"Amp_JawOpenAmp": 2.0}
    S.delete_amps(cfg, "avtr_x")
    assert C.load_avatar_amps(os.path.join(cfg, "avatarParaments"), "avtr_x") == {"Amp_JawOpenAmp": 1.5}
    with pytest.raises(ValueError):
        S.save_amps(cfg, "../evil", {})


def test_export(cymple, tmp_path):
    ini = S.resolve_ini()
    cfg = str(tmp_path / "cfg")
    assert S.export_plan(ini, str(cymple / "iniFile.ini")) == ({}, [])
    w = S.IniWriter(ini)
    w.set("OSC", "eyeGazeFilter", 0.6)
    w.set("Cymple_Mouth", "rotation", 0.0)
    w.set("ui", "themeMode", "light")                    # not a section we export
    w.flush()
    S.save_amps(cfg, None, {"Amp_JawOpenAmp": 3.0})
    keys, files = S.export_plan(ini, str(cymple / "iniFile.ini"))
    assert keys == {"OSC": {"eyeGazeFilter": "0.600000"}, "Cymple_Mouth": {"rotation": "0.000000"}}
    assert files == ["default.json"]
    _, _, backup = S.export_to_cymple(ini, str(cymple / "iniFile.ini"))
    assert open(backup).read() == INI
    theirs = C.read_ini(str(cymple / "iniFile.ini"))
    assert theirs["OSC"]["eyeGazeFilter"] == "0.600000" and theirs["ui"]["themeMode"] == "dark"
    assert json.load(open(cymple / "avatarParaments" / "default.json"))["OSC"]["Amp_JawOpenAmp"] == 3.0
    assert any(n.startswith("default.json.bak-") for n in os.listdir(cymple / "avatarParaments"))
    assert S.export_plan(ini, str(cymple / "iniFile.ini")) == ({}, [])


def test_options_merge(cymple):
    opts = S.load_options()
    assert opts["osc"]["target"] == "127.0.0.1:9000"
    opts["osc"]["enabled"] = False
    S.save_options(opts)
    again = S.load_options()
    assert again["osc"] == {"enabled": False, "target": "127.0.0.1:9000", "avatar": "auto"}


def test_update_ini_keeps_crlf(tmp_path):
    p = tmp_path / "iniFile.ini"
    p.write_bytes(INI.replace("\n", "\r\n").encode())
    C.update_ini(str(p), "OSC", {"eyeGazeFilter": "0.600000", "newKey": "1"}, backup=False)
    data = p.read_bytes()
    assert data.count(b"\r\n") == data.count(b"\n") and b"eyeGazeFilter = 0.600000\r\n" in data
    assert C.read_ini(str(p))["OSC"]["newKey"] == "1"
