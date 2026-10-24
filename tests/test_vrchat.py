import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from cymple_linux.outputs import vrcft_params as V
from cymple_linux.outputs.avatar import (FixedAvatar, OscQueryClient, find_avatar_config, params_from_oscquery,
                                         read_avatar_config)
from cymple_linux.outputs.osc import message, parse
from cymple_linux.outputs.vrchat import VRChatOutput
from cymple_linux.params import FIELDS

P = V.PREFIX


def fields(**kw):
    f = dict.fromkeys(FIELDS[:20], 0.0)
    f.update(kw)
    return f


def test_relevance_matches_vrcft_regex():
    assert V.name_matches(P + "v2/EyeLeftX", "v2/EyeLeftX")
    assert V.name_matches(P + "FT/v2/EyeLeftX", "v2/EyeLeftX")
    assert V.name_matches(P + "EyeSquintLeft", "EyeSquintLeft")
    assert not V.name_matches(P + "FT/v2/EyeSquintLeft", "EyeSquintLeft")    # legacy name, v2 address
    assert not V.name_matches(P + "v2/EyeLeftXY", "v2/EyeLeftX")
    assert V.binary_index(P + "FT/v2/EyeSquintLeft4", "v2/EyeSquintLeft") == 2
    assert V.binary_index(P + "FT/v2/EyeSquintLeft1", "v2/EyeSquintLeft") == 0
    assert V.binary_index(P + "FT/v2/EyeSquintLeft3", "v2/EyeSquintLeft") is None
    assert V.binary_index(P + "FT/v2/EyeSquintLeft4", "EyeSquintLeft") is None


def test_binary_steps_like_cymple():
    # 3 bits: Cymple's output steps at multiples of 1/7 (corpus1 OSC oracle)
    bits = lambda v, neg=False: tuple(V.binary_bit(v, k, 7, neg) for k in range(3))
    assert bits(0.14) == (False, False, False)
    assert bits(0.15) == (True, False, False)
    assert bits(3 / 7 + 0.01) == (True, True, False)
    assert bits(1.0) == (True, True, True)
    assert bits(-0.5) == (False, False, False)                 # no <name>Negative on the avatar
    assert bits(-0.5, neg=True) == (True, True, False)


def test_resolve_binds_avatar_addresses_and_native_only_without_eye_params():
    avatar = [(P + "FT/v2/EyeLeftX", float), (P + "FT/v2/EyeY", float), (P + "FT/v2/EyeSquintLeft1", bool),
              (P + "FT/v2/EyeSquintLeft2", bool), (P + "FT/v2/EyeSquintLeft4", bool),
              (P + "FT/v2/BrowExpressionLeftNegative", bool), (P + "FT/v2/BrowExpressionLeft1", bool),
              (P + "SomethingElse", float)]
    b = {x.address: x for x in V.resolve(avatar)}
    # gaze params present -> no native gaze; no Eye*Open/Lid params -> native lid stays (VRCFT rule)
    assert set(b) == {a for a, t in avatar if "Something" not in a} | {"/tracking/eye/EyesClosedAmount"}
    u = V.unified_from_fields(fields(EyeYaw_L=0.25, EyePitch_L=-0.5, EyePitch_R=-0.3, EyeSquintLeft=0.6))
    assert b[P + "FT/v2/EyeLeftX"].value(u) == pytest.approx(0.25)
    assert b[P + "FT/v2/EyeY"].value(u) == pytest.approx(-0.4)
    assert [b[P + "FT/v2/EyeSquintLeft%d" % k].value(u) for k in (1, 2, 4)] == [False, False, True]   # int(4.2)=4

    plain = {x.address for x in V.resolve([(P + "VRCEmote", int)])}
    assert plain == {"/tracking/eye/LeftRightPitchYaw", "/tracking/eye/EyesClosedAmount"}
    assert V.resolve(avatar, full_face_tracking=True) == []


def test_default_addresses_without_avatar_info():
    b = {x.address: x for x in V.resolve(None)}
    assert P + "v2/EyeLeftX" in b and P + "LeftEyeLid" in b
    assert all(x.type is not bool for x in b.values())          # bool twins would fight the floats
    u = V.unified_from_fields(fields(EyeYaw_L=1.0, EyePitch_L=1.0, EyeLidCloseLeft=1.0))
    pitch_l, yaw_l, pitch_r, yaw_r = b["/tracking/eye/LeftRightPitchYaw"].value(u)
    assert (pitch_l, yaw_l, pitch_r, yaw_r) == pytest.approx((-45.0, 45.0, 0.0, 0.0))
    assert b["/tracking/eye/EyesClosedAmount"].value(u) == pytest.approx(0.5)


def test_formulas_against_a_cymple_osc_sample():
    # corpus1 OSC oracle, first datagrams: brows R inner = outer = 0.5597, L = 0.0072
    u = V.unified_from_fields(fields(BrowInnerUpRight=0.5597356557846069, BrowOuterUpRight=0.5597356557846069,
                                     BrowInnerUpLeft=0.007223823107779026, BrowOuterUpLeft=0.007223823107779026))
    b = {x.address: x for x in V.resolve(None)}
    assert b[P + "v2/BrowExpression"].value(u) == pytest.approx(0.2834797501564026, abs=1e-7)
    assert b[P + "BrowsInnerUp"].value(u) == pytest.approx(0.5597356557846069, abs=1e-7)
    assert b[P + "v2/BrowUpLeft"].value(u) == pytest.approx(0.007223823573440313, abs=1e-7)
    u = V.unified_from_fields(fields(EyeLidCloseRight=0.9, EyeSquintRight=0.5, EyeWideRight=0.1))
    sq = (1 - 0.1 ** 0.15) * 0.5
    assert b[P + "RightEyeLidExpandedSqueeze"].value(u) == pytest.approx(0.1 * 0.2 + 0.1 * 0.8 - sq, abs=1e-6)
    assert b[P + "v2/EyeLidRight"].value(u) == pytest.approx(0.1 * 0.75 + 0.1 * 0.25, abs=1e-6)


def test_output_sends_changes_with_deadband_and_refresh(tmp_path):
    path = tmp_path / "avtr_x.json"
    path.write_text(json.dumps({"id": "avtr_x", "parameters": [
        {"name": "v2/EyeX", "input": {"address": P + "v2/EyeX", "type": "Float"}},
        {"name": "v2/EyeOpenLeft", "input": {"address": P + "v2/EyeOpenLeft", "type": "Float"}}]}))
    out = VRChatOutput(None, FixedAvatar(read_avatar_config(str(path))), refresh_s=5.0)
    first = dict(out.messages(fields(EyeYaw_L=0.2, EyeYaw_R=0.2), now=0.0))
    assert set(first) == {P + "v2/EyeX", P + "v2/EyeOpenLeft"}
    assert out.messages(fields(EyeYaw_L=0.2, EyeYaw_R=0.2), now=0.01) == []
    assert out.messages(fields(EyeYaw_L=0.20005, EyeYaw_R=0.20005), now=0.02) == []
    assert dict(out.messages(fields(EyeYaw_L=0.2002, EyeYaw_R=0.2002), now=0.03)) == {
        P + "v2/EyeX": (pytest.approx(0.2002),)}
    assert len(out.messages(fields(EyeYaw_L=0.2002, EyeYaw_R=0.2002), now=5.0)) == 2


def test_avatar_config_files(tmp_path):
    d = tmp_path / "usr_1" / "Avatars"
    d.mkdir(parents=True)
    (d / "avtr_a.json").write_text(json.dumps({"id": "avtr_a", "name": "A", "parameters": [
        {"name": "v2/EyeLeftX", "input": {"address": P + "v2/EyeLeftX", "type": "Float"},
         "output": {"address": P + "v2/EyeLeftX", "type": "Float"}},
        {"name": "IsLocal", "output": {"address": P + "IsLocal", "type": "Bool"}}]}), encoding="utf-8-sig")
    path = find_avatar_config("avtr_a", [str(tmp_path)])
    info = read_avatar_config(path)
    assert (info.id, info.name, info.params) == ("avtr_a", "A", [(P + "v2/EyeLeftX", float)])
    assert find_avatar_config("avtr_b", [str(tmp_path)]) is None


def test_osc_strings_round_trip():
    assert parse(message("/avatar/change", "avtr_123")) == [("/avatar/change", ["avtr_123"])]


TREE = {"FULL_PATH": "/avatar/parameters", "CONTENTS": {
    "VRCEmote": {"FULL_PATH": P + "VRCEmote", "TYPE": "i", "VALUE": [0]},
    "FT": {"FULL_PATH": P + "FT", "CONTENTS": {"v2": {"FULL_PATH": P + "FT/v2", "CONTENTS": {
        "EyeLeftX": {"FULL_PATH": P + "FT/v2/EyeLeftX", "TYPE": "f", "VALUE": [0.0]},
        "EyeSquintLeft1": {"FULL_PATH": P + "FT/v2/EyeSquintLeft1", "TYPE": "T", "VALUE": [False]}}}}}}}


def test_oscquery_tree_and_client():
    assert sorted(params_from_oscquery(TREE)) == sorted([
        (P + "VRCEmote", int), (P + "FT/v2/EyeLeftX", float), (P + "FT/v2/EyeSquintLeft1", bool)])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = {"/avatar/change": {"FULL_PATH": "/avatar/change", "TYPE": "s", "VALUE": ["avtr_q"]},
                    "/avatar/parameters": TREE}.get(self.path)
            self.send_response(200 if body else 404)
            self.end_headers()
            if body:
                self.wfile.write(json.dumps(body).encode())

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        info = OscQueryClient("127.0.0.1", server.server_port).avatar()
    finally:
        server.shutdown()
    assert (info.id, info.full_face_tracking, info.source) == ("avtr_q", False, "oscquery")
    assert len(info.params) == 3
    assert [b.address for b in V.resolve(info.params) if b.address.startswith("/tracking/")] == [
        "/tracking/eye/EyesClosedAmount"]


def test_mouth_shapes_from_fields():
    u = V.unified_from_fields(fields(MouthSmileLeft=0.4, LipSuckUL=0.2, LipSuckDL=0.6, TongueWide=0.3,
                                     LipRaiseUR=0.5, JawFwd=0.7, MouthShrugLower=0.1))
    assert u.face_active                                    # mouth fields present
    assert u.s("MouthCornerPullLeft") == u.s("MouthCornerSlantLeft") == 0.4
    assert u.s("LipSuckCornerLeft") == pytest.approx(0.4)   # mean of upper and lower
    assert u.s("TongueFlat") == 0.3 and u.s("MouthUpperDeepenRight") == 0.5
    assert u.s("JawForward") == 0.7 and u.s("MouthRaiserLower") == 0.1
    assert not V.unified_from_fields(dict.fromkeys(FIELDS[:20], 0.0)).face_active


def test_mouth_parameters_and_cymple_legacy_lip():
    b = {x.address[len(P):]: x for x in V.resolve(None)}
    u = V.unified_from_fields(fields(JawOpen=0.5, MouthClosed=0.2, MouthSmileRight=0.1, MouthFrownRight=0.3,
                                     MouthStretchRight=0.4, TongueOut=0.6, JawLeft=0.25,
                                     MouthUpperShiftRight=0.2, MouthLowerShiftRight=0.4))
    assert b["v2/JawX"].value(u) == pytest.approx(-0.25)
    assert b["v2/MouthX"].value(u) == pytest.approx(0.3)
    assert b["v2/SmileSadRight"].value(u) == pytest.approx(0.1 - 0.4)       # smile - max(frown, stretch)
    assert b["JawOpenApe"].value(u) == pytest.approx(0.5 - 0.2)            # Cymple: SRanipal JawOpen = JawOpen
    assert b["TongueSteps"].value(u) == pytest.approx(1.0 - 0.2)           # step1 - step2
    assert b["SmileSadRight"].value(u) == pytest.approx(0.1 - 0.4)
    assert b["MouthX"].value(u) == pytest.approx(0.3)                      # averaged, not max (0.4)


def test_catalog_has_every_vrcft_v2_shape():
    names = {q.name for q in V.CATALOG}
    assert all("v2/" + n in names for n in V.UNIFIED_EXPRESSIONS)
    assert {"v2/MouthSmileLeft", "v2/MouthSadRight", "TongueSteps", "PuffOverturn", "ExpressionTrackingActive"} <= names
