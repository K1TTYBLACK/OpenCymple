from opencymple import calibration as C
from opencymple.oracle import MODES, CympleBottle

INI = "[VRCDIRECT]\nEnabled = true\n\n[Face Inference Device]\nDeviceId = 1\n"


def test_ini_overrides_are_restored(tmp_path):
    p = tmp_path / "iniFile.ini"
    p.write_text(INI)
    b = CympleBottle(str(p))
    b.apply_ini(MODES["vrcft"][0])
    ini = C.read_ini(str(p))
    assert ini["VRCDIRECT"]["Enabled"] == "false" and ini["Face Inference Device"]["DeviceId"] == "0"
    b.restore_ini()
    assert p.read_text() == INI and not (tmp_path / "iniFile.ini.oracle-backup").exists()


def test_backup_from_interrupted_run_is_recovered_first(tmp_path):
    p = tmp_path / "iniFile.ini"
    p.write_text(INI)
    CympleBottle(str(p)).apply_ini(MODES["osc"][0])        # "crash": no restore
    b = CympleBottle(str(p))
    b.apply_ini(MODES["vrcft"][0])                           # must recover the original first
    b.restore_ini()
    assert p.read_text() == INI


def test_stop_never_touches_a_cymple_it_did_not_start(tmp_path):
    p = tmp_path / "iniFile.ini"
    p.write_text(INI)
    CympleBottle(str(p)).stop()                              # proc is None: returns without acting
