"""Which parameters the avatar VRChat is wearing right now has.

Like VRCFaceTracking: OSCQuery first (mDNS "_oscjson._tcp" service "VRChat-Client-*", HTTP JSON
tree), VRChat's per-avatar OSC config files otherwise. We poll the OSCQuery /avatar/change node
once a second and re-read /avatar/parameters when the id changes. Without OSCQuery we listen for
/avatar/change on VRChat's OSC output port (9001) and read
LocalLow/VRChat/VRChat/OSC/usr_*/Avatars/<id>.json from VRChat's Proton prefix.
"""
import glob
import json
import logging
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .osc import parse

log = logging.getLogger(__name__)

VRCHAT_APPID = 438100
OSCQUERY_SERVICE = "_oscjson._tcp.local."
OSC_TYPES = {"f": float, "i": int, "T": bool, "F": bool}
CONFIG_TYPES = {"Float": float, "Int": int, "Bool": bool}


@dataclass
class AvatarInfo:
    id: str
    name: str
    params: list = field(default_factory=list)      # [(address, type)]
    full_face_tracking: bool = False
    source: str = ""


# ---- OSCQuery ---------------------------------------------------------------------------------

def params_from_oscquery(node):
    """Leaves of an OSCQuery /avatar/parameters tree -> [(FULL_PATH, type)]."""
    out = []
    for child in (node.get("CONTENTS") or {}).values():
        if child.get("CONTENTS"):
            out += params_from_oscquery(child)
        elif child.get("TYPE") and child["TYPE"][0] in OSC_TYPES:
            out.append((child["FULL_PATH"], OSC_TYPES[child["TYPE"][0]]))
    return out


class OscQueryClient:
    def __init__(self, host, port, timeout=1.0):
        self.base = "http://%s:%d" % (host, port)
        self.timeout = timeout

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    def avatar_id(self):
        node = self.get("/avatar/change")
        value = node.get("VALUE") or [None]
        if node.get("FULL_PATH") != "/avatar/change" or not isinstance(value[0], str) or not value[0].strip():
            raise ValueError("invalid /avatar/change node")
        return value[0]

    def avatar(self):
        """Read the avatar in parts (VRChat's whole-tree reply can carry a wrong length when
        the avatar name has non-ASCII characters) and make sure it did not change meanwhile."""
        for _ in range(2):
            avatar_id = self.avatar_id()
            params = self.get("/avatar/parameters")
            if params.get("FULL_PATH") != "/avatar/parameters":
                raise ValueError("invalid /avatar/parameters node")
            try:
                ft = bool((self.get("/avatar/avatarfullfacetracking").get("VALUE") or [False])[0])
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                ft = False
            if self.avatar_id() == avatar_id:
                return AvatarInfo(avatar_id, avatar_id, params_from_oscquery(params), ft, "oscquery")
        raise ValueError("avatar changed while reading it")


class VRChatBrowser:
    """mDNS browser collecting VRChat's OSCQuery endpoints. Needs the zeroconf package."""

    def __init__(self):
        from zeroconf import ServiceBrowser, Zeroconf
        self.zc = Zeroconf()
        self.names = set()
        self.lock = threading.Lock()
        self.browser = ServiceBrowser(self.zc, OSCQUERY_SERVICE, handlers=[self._on_change])

    def _on_change(self, zeroconf, service_type, name, state_change):
        if not name.startswith("VRChat-Client"):
            return
        with self.lock:
            if state_change.name == "Removed":
                self.names.discard(name)
            else:
                self.names.add(name)

    def endpoints(self):
        """[(host, port)] of the VRChat clients seen (resolved on the caller's thread)."""
        with self.lock:
            names = list(self.names)
        out = []
        for name in names:
            info = self.zc.get_service_info(OSCQUERY_SERVICE, name, timeout=1000)
            if info is None or not info.port:
                continue
            addrs = info.parsed_addresses() or ["127.0.0.1"]
            host = "127.0.0.1" if "127.0.0.1" in addrs else addrs[0]
            out.append((host, info.port))
        return out

    def close(self):
        self.zc.close()


# ---- config files -----------------------------------------------------------------------------

def read_avatar_config(path):
    with open(path, encoding="utf-8-sig") as f:
        d = json.load(f)
    params = []
    for p in d.get("parameters") or []:
        inp = p.get("input")
        if inp and inp.get("address") and inp.get("type") in CONFIG_TYPES:
            params.append((inp["address"], CONFIG_TYPES[inp["type"]]))
    return AvatarInfo(d.get("id", ""), d.get("name", ""), params, bool(d.get("fullFaceTracking", False)),
                      "config " + os.path.basename(path))


def _steam_libraries():
    roots = [os.path.expanduser(p) for p in ("~/.local/share/Steam", "~/.steam/steam",
                                             "~/.var/app/com.valvesoftware.Steam/.local/share/Steam")]
    libs = []
    for root in roots:
        libs.append(root)
        try:
            with open(os.path.join(root, "steamapps", "libraryfolders.vdf"), encoding="utf-8") as f:
                libs += re.findall(r'"path"\s+"([^"]+)"', f.read())
        except OSError:
            pass
    seen, out = set(), []
    for lib in libs:
        real = os.path.realpath(lib)
        if real not in seen and os.path.isdir(real):
            seen.add(real)
            out.append(real)
    return out


def osc_config_dirs():
    """VRChat's LocalLow/VRChat/VRChat/OSC folders in its Proton prefix (every Steam library)."""
    out = []
    for lib in _steam_libraries():
        d = os.path.join(lib, "steamapps", "compatdata", str(VRCHAT_APPID), "pfx", "drive_c", "users",
                         "steamuser", "AppData", "LocalLow", "VRChat", "VRChat", "OSC")
        if os.path.isdir(d):
            out.append(d)
    return out


def find_avatar_config(avatar_id, dirs):
    paths = [p for d in dirs for p in glob.glob(os.path.join(d, "usr_*", "Avatars", avatar_id + ".json"))]
    return max(paths, key=os.path.getmtime) if paths else None


class FixedAvatar:
    """One avatar's parameter list that never changes (from --osc-avatar FILE)."""
    version = 1

    def __init__(self, info):
        self.current = info


# ---- watcher ----------------------------------------------------------------------------------

class AvatarWatcher(threading.Thread):
    """Keeps `current` (AvatarInfo or None) up to date; `version` changes with it."""

    def __init__(self, oscquery=True, listen_port=9001, config_dirs=None, poll_s=1.0):
        super().__init__(name="avatar-watcher", daemon=True)
        self.current, self.version = None, 0
        self.poll_s = poll_s
        self.config_dirs = osc_config_dirs() if config_dirs is None else config_dirs
        self.stop_event = threading.Event()
        self.client, self.browser, self.sock = None, None, None
        if oscquery:
            try:
                self.browser = VRChatBrowser()
            except Exception as e:                 # ImportError, or no multicast socket
                log.warning("OSCQuery discovery unavailable (%s); using VRChat's avatar config files", e)
        if listen_port:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.bind(("127.0.0.1", listen_port))
                self.sock.settimeout(poll_s)
            except OSError as e:
                log.warning("cannot listen for /avatar/change on UDP %d (%s)", listen_port, e)
                self.sock = None

    def _set(self, info):
        if info is None or self.current is None or (info.id, info.params) != (self.current.id, self.current.params):
            self.current = info
            self.version += 1
            if info is not None:
                log.info("VRChat avatar %s (%s): %d parameters", info.name or info.id, info.source, len(info.params))

    def _poll_oscquery(self):
        if self.client is None and self.browser is not None:
            for host, port in self.browser.endpoints():
                self.client = OscQueryClient(host, port)
                log.info("VRChat OSCQuery at %s:%d", host, port)
                break
        if self.client is None:
            return
        try:
            avatar_id = self.client.avatar_id()
            if self.current is None or self.current.source != "oscquery" or avatar_id != self.current.id:
                info = self.client.avatar()
                path = find_avatar_config(info.id, self.config_dirs)      # OSCQuery has no avatar name
                if path is not None:
                    try:
                        info.name = read_avatar_config(path).name or info.name
                    except (OSError, ValueError):
                        pass
                self._set(info)
        except (urllib.error.HTTPError, ValueError) as e:
            log.debug("VRChat OSCQuery not ready (%s); retrying", e)    # e.g. 404 while an avatar loads
        except OSError as e:
            log.info("VRChat OSCQuery lost (%s)", e)
            self.client = None

    def _on_avatar_change(self, avatar_id):
        if self.client is not None:
            self._poll_oscquery()
            return
        path = find_avatar_config(avatar_id, self.config_dirs)
        if path is None:
            log.warning("no OSC config file for avatar %s in %s", avatar_id, self.config_dirs)
            return
        try:
            self._set(read_avatar_config(path))
        except (OSError, ValueError) as e:
            log.warning("cannot read %s: %s", path, e)

    def run(self):
        next_poll = 0.0
        while not self.stop_event.is_set():
            if time.monotonic() >= next_poll:
                self._poll_oscquery()
                next_poll = time.monotonic() + self.poll_s
            if self.sock is None:
                self.stop_event.wait(max(0.0, next_poll - time.monotonic()))
                continue
            try:
                data, _ = self.sock.recvfrom(65536)
                for addr, args in parse(data):
                    if addr == "/avatar/change" and args and isinstance(args[0], str):
                        self._on_avatar_change(args[0])
            except socket.timeout:
                pass
            except (OSError, ValueError, IndexError, UnicodeDecodeError):
                pass

    def stop(self):
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout=2.0)
        if self.browser is not None:
            self.browser.close()
        if self.sock is not None:
            self.sock.close()
