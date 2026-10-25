"""OpenCymple's own settings in ~/.config/opencymple/ ($OPENCYMPLE_CONFIG overrides): iniFile.ini
(calibration and tuning), avatarParaments/ (amplitude sets) and opencymple.json (our options).

A fresh install has none of these: every value comes from the defaults in calibration.py and the
files appear as soon as something is changed. The INI keeps Cymple's key format.
"""
import json
import logging
import os
import re
import shutil
import threading
import time

from . import calibration as C

log = logging.getLogger(__name__)

INI_NAME = "iniFile.ini"
AVATAR_DIR = "avatarParaments"
OPTIONS_NAME = "opencymple.json"
DEFAULT_OPTIONS = {"osc": {"enabled": True, "target": "127.0.0.1:9000", "avatar": "auto", "config_dir": ""},
                   "models_dir": "",
                   "afk": {"enabled": False}}


def config_dir():
    env = os.environ.get("OPENCYMPLE_CONFIG")
    if env:
        return env
    return os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "opencymple")


def own_ini(directory=None):
    return os.path.join(directory or config_dir(), INI_NAME)


def _stamp():
    return time.strftime("%Y%m%d-%H%M%S")


def resolve_ini(explicit=None):
    """The INI a command works on: `explicit`, else ours (created empty on first use: all defaults)."""
    if explicit:
        return explicit
    dst = own_ini()
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "w").close()
    return dst


def find_model_dir(explicit=None):
    """Directory with model0.bin ... model6.bin: --models, $CYMPLE_MODELS, "models_dir" in
    opencymple.json, <config dir>/model, else the models bundled in the AppImage."""
    appdir = os.environ.get("APPDIR")
    for cand in (explicit, os.environ.get("CYMPLE_MODELS"), load_options().get("models_dir"),
                 os.path.join(config_dir(), "model"),
                 appdir and os.path.join(appdir, "usr", "share", "opencymple", "model")):
        if cand and os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "model0.bin")):
            return cand
    return None


# ---- our own options --------------------------------------------------------------------------

def load_options(directory=None):
    opts = json.loads(json.dumps(DEFAULT_OPTIONS))
    try:
        with open(os.path.join(directory or config_dir(), OPTIONS_NAME), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return opts
    for k, v in data.items():
        if isinstance(v, dict) and isinstance(opts.get(k), dict):
            opts[k].update(v)
        else:
            opts[k] = v
    return opts


def save_options(opts, directory=None):
    d = directory or config_dir()
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, OPTIONS_NAME + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(opts, f, indent=2)
    os.replace(tmp, os.path.join(d, OPTIONS_NAME))


# ---- INI writes -------------------------------------------------------------------------------

def ini_value(v):
    """Cymple's (QSettings) spelling: true/false, floats with 6 decimals."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return "%.6f" % v
    return str(v)


class IniWriter:
    """Collects key changes and writes them in one go (update_ini keeps the rest of the file)."""

    def __init__(self, path):
        self.path = path
        self.pending = {}
        self.lock = threading.Lock()

    def set(self, section, key, value):
        with self.lock:
            self.pending.setdefault(section, {})[key] = ini_value(value)

    def flush(self):
        with self.lock:
            pending, self.pending = self.pending, {}
            if not pending:
                return False
            for section, values in pending.items():
                C.update_ini(self.path, section, values, backup=False)
            return True


# ---- amplitude sets (avatarParaments/<avatar id>.json, default.json) ---------------------------

def _avatar_file(directory, avatar_id):
    if avatar_id is not None and not re.fullmatch(r"[A-Za-z0-9_\-]+", avatar_id):
        raise ValueError("unexpected avatar id %r" % avatar_id)
    return os.path.join(directory, AVATAR_DIR, (avatar_id or "default") + ".json")


def has_own_amps(directory, avatar_id):
    return bool(avatar_id) and os.path.isfile(_avatar_file(directory, avatar_id))


def save_amps(directory, avatar_id, amps):
    """Write the 28 Amp_* keys into <avatar>.json (default.json for None), keeping other keys."""
    path = _avatar_file(directory, avatar_id)
    data = {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        pass
    osc = data.setdefault("OSC", {}) if isinstance(data, dict) else {}
    for k in C.AMP_KEYS:
        if k in amps:
            osc[k] = round(float(amps[k]), 6)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, sort_keys=True)
    os.replace(tmp, path)
    return path


def delete_amps(directory, avatar_id):
    if avatar_id:
        try:
            os.remove(_avatar_file(directory, avatar_id))
        except FileNotFoundError:
            pass


# ---- export / import of our settings as one file -----------------------------------------------

BUNDLE_FORMAT = "opencymple-config"


# section names in the exported file (the INI keeps its own); keys naming the original app are left out
SECTION_NAMES = {"Cymple_Mouth": "Mouth", "CympleFace_LeftEye": "LeftEye", "CympleFace_RightEye": "RightEye"}
INI_SECTIONS = {v: k for k, v in SECTION_NAMES.items()}


def _typed(v):
    """INI text -> bool/int/float when that converts back to the same text, else the text."""
    if v in ("true", "false"):
        return v == "true"
    for conv in (int, float):
        try:
            if str(conv(v)) == v:
                return conv(v)
        except ValueError:
            pass
    return v


def _ini_text(v):
    return ("true" if v else "false") if isinstance(v, bool) else str(v)


def export_bundle(path, own=None, directory=None):
    """Save our settings and amplitude sets as one JSON file at `path`."""
    own = own or own_ini(directory)
    av_dir = os.path.join(directory or os.path.dirname(own), AVATAR_DIR)
    settings = {SECTION_NAMES.get(sec, sec): {k: _typed(v) for k, v in vals.items() if "cymple" not in k.lower()}
                for sec, vals in C.read_ini(own).items() if sec}
    sets = {}
    if os.path.isdir(av_dir):
        for name in sorted(os.listdir(av_dir)):
            if name.endswith(".json"):
                try:
                    with open(os.path.join(av_dir, name), encoding="utf-8") as f:
                        sets[name[:-5]] = json.load(f)
                except (OSError, ValueError):
                    log.warning("skipping unreadable amplitude set %s", name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"format": BUNDLE_FORMAT, "version": 2, "settings": settings, AVATAR_DIR: sets}, f, indent=1)


def import_bundle(path, directory=None):
    """Apply a file made by export_bundle to our INI and amplitude
    sets, keeping the INI's formatting. Replaced files are kept as .bak-<time> copies."""
    d = directory or config_dir()
    try:
        with open(path, encoding="utf-8") as f:
            bundle = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError("not an OpenCymple config file: %s" % e)
    if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
        raise ValueError("not an OpenCymple config file")
    os.makedirs(d, exist_ok=True)
    stamp = _stamp()
    dst = own_ini(d)
    if os.path.exists(dst):
        shutil.copy2(dst, "%s.bak-%s" % (dst, stamp))
    else:
        open(dst, "w").close()
    for section, values in bundle.get("settings", {}).items():
        section = INI_SECTIONS.get(section, section)
        if section and isinstance(values, dict):
            C.update_ini(dst, section, {k: _ini_text(v) for k, v in values.items()}, backup=False)
    av_dir = os.path.join(d, AVATAR_DIR)
    for name, content in bundle.get(AVATAR_DIR, {}).items():
        if os.path.basename(name) != name or not isinstance(content, dict):
            continue
        os.makedirs(av_dir, exist_ok=True)
        target = os.path.join(av_dir, name + ".json")
        if os.path.exists(target):
            shutil.copy2(target, "%s.bak-%s" % (target, stamp))
        with open(target, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=1)
    log.info("imported settings from %s into %s", path, d)
    return dst
