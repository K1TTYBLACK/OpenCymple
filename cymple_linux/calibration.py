"""Read (and selectively update) Cymple's own iniFile.ini so existing calibration carries over.

Defaults for keys the INI may lack come from the faceOscClass constructor (sub_1800b67d0):
eyeExpressionFilter 0.1, eyeGazeFilter 0.8, eyeBlinkCloseAmplitude 0.2, eyeGazeAmplitude 1.0,
eyeSquintGazeCompensation 0.
"""
import glob
import os
import re
import shutil
import time
from dataclasses import dataclass, field

import numpy as np

ROI_KEYS = ("RoiLeftTopX", "RoiLeftTopY", "RoiLeftBottomX", "RoiLeftBottomY",
            "RoiRightTopX", "RoiRightTopY", "RoiRightBottomX", "RoiRightBottomY")
EYE_SECTIONS = ("CympleFace_LeftEye", "CympleFace_RightEye")
SIDES = ("Left", "Right")
LID_KEYS = ("upper_outer_y", "upper_inner_y", "lower_outer_y", "lower_inner_y")
LID_REF_POSITIONS = ("left", "center", "right")
# OSC_EYE_CONFIG_S switch items (cympleFacelib key table at 0x18003b450, index = sw slot)
SWITCHES = {7: "eyePitchFlip", 8: "gazeForceSync", 9: "lidSync", 12: "silly"}
EYE_AMP_KEYS = ("Amp_EyeSquint", "Amp_EyeWide", "Amp_BrowDown", "Amp_BrowInRaise", "Amp_BrowPinch")
# OSC_MOUTH_CONFIG_S: cympleFacelib key table 0x18003b4c0 (slot i <- "Amp_" + name + "Amp"), then the
# eye keys (0x18003b578, no suffix). cympleface copies the 28 floats to this+0x1184.
MOUTH_AMP_NAMES = ("NoseSneer", "CheekSquint", "CheekPuff", "CheekSuck", "JawShift", "JawFwd", "JawOpen",
                   "MouthClose", "MouthShift", "LipRoll", "MouthFunnel", "MouthPucker", "LipRaise", "LipDepress",
                   "ShrugLower", "MouthSmile", "MouthDimple", "MouthFrown", "MouthStretch", "TongueOut",
                   "TongueShift", "TongueUpDown", "TongueRoll")
AMP_KEYS = tuple("Amp_%sAmp" % n for n in MOUTH_AMP_NAMES) + EYE_AMP_KEYS
AMP_FALLBACK = {"Amp_BrowDown": "Amp_BrowPinch", "Amp_BrowInRaise": "Amp_EyeWide",   # sub_18002c910
                "Amp_BrowPinch": "Amp_BrowDown"}
# Calibration / MouthCalibration attribute -> INI key, for the settings the UI changes
OSC_SETTING_KEYS = {"eye_expression_filter": "eyeExpressionFilter", "eye_gaze_filter": "eyeGazeFilter",
                    "eye_blink_close_amplitude": "eyeBlinkCloseAmplitude", "eye_gaze_amplitude": "eyeGazeAmplitude",
                    "eye_squint_gaze_compensation": "eyeSquintGazeCompensation",
                    "eye_gaze_sync_rate": "eyeGazeSyncRate"}
MOUTH_SETTING_KEYS = {"auto_roi": "v1_auto_roi_calibration", "rotation": "rotation", "upper": "upper",
                      "bottom": "bottom", "width": "width", "x_offset": "x_offset", "filter": "filer"}


@dataclass
class EyeCalibration:
    roi_quad: np.ndarray = None              # 4x2 float32 (P0..P3) or None
    mesh: list = None                        # 13 x (raw_yaw, raw_pitch, target_yaw, target_pitch)
    mesh_valid: bool = False
    feature_mode: int = 2
    neutral: tuple = (0.0, 0.0, 0.0)         # widen, brow_down, brow_in_raise
    range_yaw: tuple = (-1.0, 0.0, 1.0)      # legacy (min, mid, max) fallback
    range_pitch: tuple = (-1.0, 0.0, 1.0)
    # OSC_EYE_CONFIG_S items 3/4 (vrcft<Side>Lid) and 5/6 (squint<Side>): value0..value2
    lid_values: tuple = (0.0, 0.0, 0.0)
    squint_values: tuple = (0.0, 0.0, 0.0)
    wide_values: tuple = (0.0, 0.0, 1.0)     # eyeWide<Side>value0..2 (written by recenter)
    overlay_y_offset: float = 0.0            # full-view calibration overlay geometry saved with the mesh
    overlay_ring_scale: float = 1.0
    # <Side>_Eye{upper,lower}_{outer,inner}_y: lid geometry used when no gaze reference applies
    center_refs: tuple = None                # (upper_outer, upper_inner, lower_outer, lower_inner)
    # <Side>_Eyelid_ref_{left,center,right}_*: [(raw_yaw, (uo, ui, lo, li))] x 3
    lid_refs: list = None


@dataclass
class MouthCalibration:
    auto_roi: bool = True
    rotation: float = 0.0
    upper: float = 0.0
    bottom: float = 1.0
    width: float = 1.0
    x_offset: float = 0.5
    filter: float = 0.25


@dataclass
class Calibration:
    path: str = None
    eyes: list = field(default_factory=lambda: [EyeCalibration(), EyeCalibration()])
    mouth: MouthCalibration = field(default_factory=MouthCalibration)
    precise_gaze: bool = True
    eye_gaze_filter: float = 0.8
    eye_expression_filter: float = 0.1
    eye_gaze_amplitude: float = 1.0
    eye_squint_gaze_compensation: float = 0.0
    eye_blink_close_amplitude: float = 0.2
    eye_gaze_sync_rate: float = 0.0
    overlay_geometry: tuple = (0.0, 1.0)                # calibration overlay (y offset m, ring scale)
    switches: dict = field(default_factory=dict)        # sw slot -> bool
    amps: dict = field(default_factory=dict)            # Amp_* -> float (avatarParaments JSON)


def read_ini(path):
    """{section: {key: value}} keeping key case; tolerant of duplicates and odd lines."""
    out, section = {}, ""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                out.setdefault(section, {})
            elif "=" in line and not line.startswith((";", "#")):
                k, v = line.split("=", 1)
                out.setdefault(section, {})[k.strip()] = v.strip()
    return out


def _f(d, key, default):
    try:
        return float(d[key])
    except (KeyError, ValueError):
        return default


def _b(d, key, default):
    v = d.get(key)
    return default if v is None else v.lower() == "true"


def load(path):
    ini = read_ini(path)
    osc = ini.get("OSC", {})
    cal = Calibration(path=path)
    for eye, (section, side) in enumerate(zip(EYE_SECTIONS, SIDES)):
        e = cal.eyes[eye]
        sec = ini.get(section, {})
        if _b(sec, "RoiQuadValid", False) and all(k in sec for k in ROI_KEYS):
            e.roi_quad = np.float32([float(sec[k]) for k in ROI_KEYS]).reshape(4, 2)
        try:
            e.mesh = [tuple(float(osc["%s_Eyegaze_mesh_%d_%s" % (side, i, f)])
                            for f in ("raw_yaw", "raw_pitch", "target_yaw", "target_pitch"))
                      for i in range(13)]
            e.mesh_valid = _b(osc, "%s_Eyegaze_mesh_valid" % side, False)
        except (KeyError, ValueError):
            e.mesh, e.mesh_valid = None, False
        e.feature_mode = int(_f(osc, "%s_Eyegaze_feature_mode" % side, 2))
        e.neutral = tuple(_f(osc, "%s_Eyeexpression_neutral_%s" % (side, k), 0.0)
                          for k in ("widen", "brow_down", "brow_in_raise"))
        # [Inferred] The legacy fallback triples (this+0x140..) look like the pupil_x/y ranges:
        # their mid values equal the centre raw yaw/pitch of the mesh calibration.
        px = [_f(osc, "%s_Eyepupil_x_%s" % (side, k), None) for k in ("min", "mid", "max")]
        py = [_f(osc, "%s_Eyepupil_y_%s" % (side, k), None) for k in ("min", "mid", "max")]
        if None not in px:
            e.range_yaw = tuple(px)
        if None not in py:
            e.range_pitch = tuple(py)
        e.lid_values = tuple(_f(osc, "vrcft%sLidvalue%d" % (side, i), 0.0) for i in range(3))
        e.squint_values = tuple(_f(osc, "squint%svalue%d" % (side, i), 0.0) for i in range(3))
        e.wide_values = tuple(_f(osc, "eyeWide%svalue%d" % (side, i), (0.0, 0.0, 1.0)[i]) for i in range(3))
        e.overlay_y_offset = _f(osc, "%s_Eyegaze_mesh_overlay_y_offset_meters" % side,
                                _f(osc, "eye_gaze_calibration_overlay_y_offset_meters", 0.0))
        e.overlay_ring_scale = _f(osc, "%s_Eyegaze_mesh_overlay_ring_scale" % side,
                                  _f(osc, "eye_gaze_calibration_overlay_ring_scale", 1.0))
        refs = []
        for pos in LID_REF_POSITIONS:
            prefix = "%s_Eyelid_ref_%s_" % (side, pos)
            vals = [_f(osc, prefix + k, None) for k in ("yaw",) + LID_KEYS]
            if None in vals:
                refs = None
                break
            refs.append((vals[0], tuple(vals[1:])))
        e.lid_refs = refs
        center = [_f(osc, "%s_Eye%s" % (side, k), None) for k in LID_KEYS]
        e.center_refs = tuple(center) if None not in center else (refs[1][1] if refs else None)
    cal.switches = {slot: _b(osc, name, False) for slot, name in SWITCHES.items()}
    cal.amps = load_avatar_amps(os.path.join(os.path.dirname(path), "avatarParaments"),
                                ini.get("VRCDIRECT", {}).get("SelectedAvatarId"))
    cal.precise_gaze = _b(osc, "eye_gaze_use_precise_calibration", True)
    cal.overlay_geometry = (_f(osc, "eye_gaze_calibration_overlay_y_offset_meters", 0.0),
                            _f(osc, "eye_gaze_calibration_overlay_ring_scale", 1.0))
    cal.eye_gaze_filter = _f(osc, "eyeGazeFilter", 0.8)
    cal.eye_expression_filter = _f(osc, "eyeExpressionFilter", 0.1)
    cal.eye_gaze_amplitude = _f(osc, "eyeGazeAmplitude", 1.0)
    cal.eye_squint_gaze_compensation = _f(osc, "eyeSquintGazeCompensation", 0.0)
    cal.eye_blink_close_amplitude = _f(osc, "eyeBlinkCloseAmplitude", 0.2)
    cal.eye_gaze_sync_rate = _f(osc, "eyeGazeSyncRate", 0.0)
    m = ini.get("Cymple_Mouth", {})
    cal.mouth = MouthCalibration(
        auto_roi=_b(m, "v1_auto_roi_calibration", True), rotation=_f(m, "rotation", 0.0),
        upper=_f(m, "upper", 0.0), bottom=_f(m, "bottom", 1.0), width=_f(m, "width", 1.0),
        x_offset=_f(m, "x_offset", 0.5), filter=_f(m, "filer", 0.25))
    return cal


def load_avatar_amps(directory, avatar_id=None):
    """Amp_* values from avatarParaments/<avatar>.json, else default.json (missing keys: 1.0)."""
    import json
    for name in ([avatar_id + ".json"] if avatar_id else []) + ["default.json"]:
        p = os.path.join(directory, name)
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            osc = data.get("OSC", data) if isinstance(data, dict) else {}
            return {k: float(v) for k, v in osc.items() if k.startswith("Amp_") and isinstance(v, (int, float))}
    return {}


def amp_slots(amps):
    """The 28 amplitude slots in Cymple's struct order (sub_18002ce20): key, else its fallback
    key, else 1.0; each clamped to [0, 5]."""
    out = []
    for k in AMP_KEYS:
        v = amps.get(k, amps.get(AMP_FALLBACK.get(k, ""), 1.0))
        out.append(min(max(float(v), 0.0), 5.0))
    return out


def find_ini():
    """$CYMPLE_INI, else the first Cymple iniFile.ini found in a Bottles or Wine prefix."""
    env = os.environ.get("CYMPLE_INI")
    if env:
        return env
    patterns = [
        "~/.var/app/com.usebottles.bottles/data/bottles/bottles/*/drive_c/users/*/AppData/Local/Cymple/iniFile.ini",
        "~/.local/share/bottles/bottles/*/drive_c/users/*/AppData/Local/Cymple/iniFile.ini",
        "~/.wine/drive_c/users/*/AppData/Local/Cymple/iniFile.ini",
    ]
    for p in patterns:
        hits = sorted(glob.glob(os.path.expanduser(p)))
        if hits:
            return hits[0]
    return None


def find_model_dir(ini_path=None):
    """$CYMPLE_MODELS, else Program Files/Cymple/model in the same prefix as the INI."""
    env = os.environ.get("CYMPLE_MODELS")
    if env:
        return env
    if ini_path:
        drive_c = os.path.abspath(os.path.join(os.path.dirname(ini_path), *[".."] * 5))
        cand = os.path.join(drive_c, "Program Files", "Cymple", "model")
        if os.path.isdir(cand):
            return cand
    return None


def update_ini(path, section, values, backup=True):
    """Set keys of one section in place, keeping the rest of the file byte for byte. Missing
    keys are appended to the section; a missing section is appended to the file. Returns the
    backup path (or None)."""
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        text = f.read()
    eol = "\r\n" if "\r\n" in text else "\n"             # Cymple (QSettings on Windows) writes CRLF
    text = text.replace("\r\n", "\n")
    m = re.search(r"^\[%s\]\s*$(.*?)(?=^\[|\Z)" % re.escape(section), text, re.M | re.S)
    if m:
        head, body, tail = text[:m.start(1)], m.group(1), text[m.end(1):]
    else:
        head, body, tail = text.rstrip("\n") + "\n\n[%s]\n" % section, "", "\n"
    for k, v in values.items():
        body, n = re.subn(r"^(%s\s*=\s*).*$" % re.escape(k), lambda mm, v=v: mm.group(1) + str(v), body, flags=re.M)
        if n == 0:
            stripped = body.rstrip("\n")
            trailing = body[len(stripped):] or "\n"
            body = stripped + ("\n" if stripped else "") + "%s = %s" % (k, v) + trailing
    bak = None
    if backup:
        bak = "%s.bak-%s" % (path, time.strftime("%Y%m%d-%H%M%S"))
        shutil.copy2(path, bak)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write((head + body + tail).replace("\n", eol))
    return bak


def roi_values(quad):
    values = dict(zip(ROI_KEYS, ("%.6f" % v for v in np.asarray(quad, np.float32).reshape(-1))))
    values["RoiQuadValid"] = "true"
    return values


def write_roi_quad(path, eye, quad, backup=True):
    """Replace the eye's Roi* keys in place (formatting kept), after a timestamped backup."""
    return update_ini(path, EYE_SECTIONS[eye], roi_values(quad), backup=backup)
