"""opencymple-linux's own settings: a copy of Cymple's iniFile.ini and avatarParaments/ in
~/.config/opencymple-linux/ ($OPENCYMPLE_LINUX_CONFIG overrides), imported once on first use.

Keys and value formats stay Cymple's, so calibration.load() reads the copy unchanged and
`opencymple-linux config export` can write it back. Tuning here never touches the Windows app's
files; only an explicit export does (with a backup). Options of our own (VRChat OSC target,
...) live in opencymple-linux.json next to the INI.
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
OPTIONS_NAME = "opencymple-linux.json"
EXPORT_SECTIONS = ("OSC", "Cymple_Mouth", "CympleFace_LeftEye", "CympleFace_RightEye")
DEFAULT_OPTIONS = {"osc": {"enabled": True, "target": "127.0.0.1:9000", "avatar": "auto"}}


def config_dir():
    env = os.environ.get("OPENCYMPLE_LINUX_CONFIG")
    if env:
        return env
    return os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "opencymple-linux")


def own_ini(directory=None):
    return os.path.join(directory or config_dir(), INI_NAME)


def _stamp():
    return time.strftime("%Y%m%d-%H%M%S")


def import_from_cymple(source_ini, directory=None, force=False):
    """Copy Cymple's INI and avatarParaments/ into our directory (only once unless `force`;
    a forced import keeps the previous copy as a backup). Returns our INI path."""
    d = directory or config_dir()
    dst = own_ini(d)
    if os.path.exists(dst) and not force:
        return dst
    os.makedirs(d, exist_ok=True)
    if os.path.exists(dst):
        shutil.copy2(dst, "%s.bak-%s" % (dst, _stamp()))
    shutil.copy2(source_ini, dst)
    src_av = os.path.join(os.path.dirname(source_ini), AVATAR_DIR)
    if os.path.isdir(src_av):
        shutil.copytree(src_av, os.path.join(d, AVATAR_DIR), dirs_exist_ok=True)
    opts = load_options(d)
    opts.update(imported_from=os.path.abspath(source_ini), imported_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    save_options(opts, d)
    log.info("imported Cymple settings from %s into %s", source_ini, d)
    return dst


def resolve_ini(explicit=None, own=True):
    """The INI a command works on: `explicit`, else our copy (imported from Cymple's INI the
    first time), or Cymple's own INI with own=False (the oracle commands need that one)."""
    if explicit:
        return explicit
    cymple = C.find_ini()
    if not own:
        return cymple
    if os.path.exists(own_ini()):
        return own_ini()
    return import_from_cymple(cymple) if cymple else None


def is_cymple_ini(path):
    cymple = C.find_ini()
    try:
        return bool(cymple) and os.path.samefile(path, cymple)
    except OSError:
        return False


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
    """Collects key changes and writes them in one go (update_ini keeps the rest of the file).
    Refuses Cymple's own INI: that one changes only through `config export`."""

    def __init__(self, path):
        self.path = path
        self.readonly = is_cymple_ini(path)
        self.pending = {}
        self.lock = threading.Lock()

    def set(self, section, key, value):
        with self.lock:
            self.pending.setdefault(section, {})[key] = ini_value(value)

    def flush(self):
        with self.lock:
            pending, self.pending = self.pending, {}
            if self.readonly or not pending:
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


# ---- export back to the Windows app ------------------------------------------------------------

def export_plan(own, target, directory=None):
    """({section: {key: value}} that differ in our INI, [avatar JSON names that differ])."""
    ours, theirs = C.read_ini(own), C.read_ini(target)
    keys = {}
    for section in EXPORT_SECTIONS:
        diff = {k: v for k, v in ours.get(section, {}).items() if theirs.get(section, {}).get(k) != v}
        if diff:
            keys[section] = diff
    src_av = os.path.join(directory or os.path.dirname(own), AVATAR_DIR)
    dst_av = os.path.join(os.path.dirname(target), AVATAR_DIR)
    files = []
    if os.path.isdir(src_av):
        for name in sorted(os.listdir(src_av)):
            if not name.endswith(".json"):
                continue
            a, b = os.path.join(src_av, name), os.path.join(dst_av, name)
            with open(a, "rb") as f:
                mine = f.read()
            try:
                with open(b, "rb") as f:
                    same = f.read() == mine
            except OSError:
                same = False
            if not same:
                files.append(name)
    return keys, files


def export_to_cymple(own, target, directory=None):
    """Write our changed keys and amplitude files into Cymple's settings. The INI is backed up
    once, every replaced JSON next to itself. Returns (keys, files, ini backup path)."""
    keys, files = export_plan(own, target, directory)
    backup = None
    for i, (section, values) in enumerate(keys.items()):
        b = C.update_ini(target, section, values, backup=(i == 0))
        backup = backup or b
    src_av = os.path.join(directory or os.path.dirname(own), AVATAR_DIR)
    dst_av = os.path.join(os.path.dirname(target), AVATAR_DIR)
    for name in files:
        dst = os.path.join(dst_av, name)
        os.makedirs(dst_av, exist_ok=True)
        if os.path.exists(dst):
            shutil.copy2(dst, "%s.bak-%s" % (dst, _stamp()))
        shutil.copy2(os.path.join(src_av, name), dst)
    return keys, files, backup
