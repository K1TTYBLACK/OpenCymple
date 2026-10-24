import struct

from opencymple_linux import calibration as C
from opencymple_linux.params import FIELDS, INDEX, decode_vrcft, encode_vrcft

INI = """[Others]
LANGUAGE = ui_EN.qm


[VRCDIRECT]
Enabled = true
SelectedAvatarId = avtr_x


[Face Inference Device]
DeviceId = 1
"""


def test_vrcft_packet_layout():
    values = [i / 100 for i in range(70)]
    pkt = encode_vrcft(3, values)
    assert len(pkt) == 292
    assert struct.unpack_from("<i", pkt)[0] == -3 and pkt[4:8] == b"\x00\x00\x00\x03"
    assert struct.unpack_from("<HH", pkt, 8) == (0, 280)
    assert abs(struct.unpack_from("<f", pkt, 12 + 4 * INDEX["JawOpen"])[0] - values[31]) < 1e-6
    flags, back = decode_vrcft(pkt)
    assert flags == 3 and max(abs(a - b) for a, b in zip(values, back)) < 1e-6
    assert decode_vrcft(pkt[:200]) is None
    assert FIELDS[0] == "EyePitch_L" and FIELDS[69] == "TongueRoll"


def test_update_ini_keeps_formatting(tmp_path):
    p = tmp_path / "iniFile.ini"
    p.write_text(INI)
    bak = C.update_ini(str(p), "VRCDIRECT", {"Enabled": "false", "SendPort": "9000"})
    text = p.read_text()
    assert "Enabled = false" in text and "SelectedAvatarId = avtr_x" in text
    assert text.index("SendPort = 9000") < text.index("[Face Inference Device]")
    assert text.replace("Enabled = false", "Enabled = true").replace("SendPort = 9000\n", "") == INI
    assert open(bak).read() == INI
    C.update_ini(str(p), "New Section", {"A": 1}, backup=False)
    assert C.read_ini(str(p))["New Section"]["A"] == "1"
    assert C.read_ini(str(p))["Face Inference Device"]["DeviceId"] == "1"
