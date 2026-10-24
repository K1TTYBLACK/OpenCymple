"""Windows Cymple as a test oracle: replay a capture into it, record what it outputs.

`record` temporarily edits Cymple's INI (backup restored afterwards):
  [VRCDIRECT] Enabled = false   -> the 292-byte VRCFT packet to 127.0.0.1:22999 is only sent
                                   while direct OSC is off (sub_1800d18d0 tests this flag)
  [Face Inference Device] DeviceId = 0   -> CPU inference (sub_18005c5c0), comparable numerics
then starts Cymple through bottles-cli, replays the capture to 127.0.0.1:23303 in real time and
logs every 22999 / 23304 datagram as JSON lines, with timestamps mapped back to capture time.

The headset relay may keep streaming meanwhile: an Absorber binds 23303 on every LAN address.
Linux delivers a datagram to the socket bound to its exact destination address before a
wildcard one, so live frames (sent to the LAN address) land in the absorber, and only the
loopback replay reaches Cymple's 0.0.0.0 socket. Cymple therefore never sees the live
source, which would otherwise own the devices (sub_180090f80).

`compare` runs our pipeline deterministically over the same capture and pairs each oracle
packet with our state at the same capture time.
"""
import json
import logging
import math
import os
import shutil
import socket
import subprocess
import threading
import time

import numpy as np

from . import calibration as C
from . import protocol as P
from .params import FIELDS, decode_vrcft
from .receiver import Ingest, ipv4_broadcast_addresses, read_capture

log = logging.getLogger(__name__)

BOTTLES = ["flatpak", "run", "--command=bottles-cli", "com.usebottles.bottles"]
VRCFT_PORT, PIXEL_GAZE_PORT, OSC_PORT = 22999, 23304, 9000
# mode -> (INI overrides, ports to record). "osc" makes Cymple send every parameter it knows
# (ForceRelevant) to 127.0.0.1:9000, as it would to VRChat.
MODES = {
    "vrcft": ({"VRCDIRECT": {"Enabled": "false"}, "Face Inference Device": {"DeviceId": "0"}},
              (VRCFT_PORT, PIXEL_GAZE_PORT)),
    "osc": ({"VRCDIRECT": {"Enabled": "true", "ForceRelevant": "true", "SendIp": "127.0.0.1",
                           "SendPort": str(OSC_PORT)},
             "Face Inference Device": {"DeviceId": "0"}},
            (OSC_PORT, PIXEL_GAZE_PORT)),
}


def udp_port_bound(port, wildcard_only=False):
    """True if a socket on this host is bound to UDP `port` (reads /proc/net/udp); with
    wildcard_only, only a 0.0.0.0 / :: bind counts (Cymple's, not the absorber's)."""
    for table in ("/proc/net/udp", "/proc/net/udp6"):
        try:
            with open(table) as f:
                next(f)
                for line in f:
                    addr, p = line.split()[1].rsplit(":", 1)
                    if int(p, 16) == port and (not wildcard_only or int(addr, 16) == 0):
                        return True
        except OSError:
            continue
    return False


class Absorber(threading.Thread):
    """Hold UDP `port` on every LAN address and discard what arrives there (see module doc)."""

    def __init__(self, port=P.INGEST_PORT):
        super().__init__(name="absorber", daemon=True)
        self.socks = []
        for _, addr, _ in ipv4_broadcast_addresses():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((addr, port))
            s.setblocking(False)
            self.socks.append(s)
        self.count = 0
        self.stop_event = threading.Event()

    def run(self):
        import select
        while not self.stop_event.is_set():
            ready, _, _ = select.select(self.socks, [], [], 0.1)
            for s in ready:
                try:
                    while True:
                        s.recv(65535)
                        self.count += 1
                except BlockingIOError:
                    pass
        for s in self.socks:
            s.close()


def cymple_pids():
    """PIDs of running appCymple.exe / VrcftModuleProbe.exe Wine processes."""
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % d, "rb") as f:
                cmd = f.read()
        except OSError:
            continue
        if b"appCymple.exe" in cmd or b"VrcftModuleProbe.exe" in cmd:
            pids.append(int(d))
    return pids


class CympleBottle:
    def __init__(self, ini_path, bottle="Cymple", program="Cymple", log_path=None):
        self.ini_path, self.bottle, self.program = ini_path, bottle, program
        self.log_path = log_path
        self.proc = None
        self.backup = ini_path + ".oracle-backup"

    def recover_ini(self):
        """A backup left behind means an earlier run died before restoring: restore it now."""
        if os.path.exists(self.backup):
            log.warning("restoring %s from a backup left by an interrupted oracle run", self.ini_path)
            self.restore_ini()

    def apply_ini(self, overrides):
        self.recover_ini()
        shutil.copy2(self.ini_path, self.backup)
        for section, values in overrides.items():
            C.update_ini(self.ini_path, section, values, backup=False)

    def restore_ini(self):
        if os.path.exists(self.backup):
            shutil.copy2(self.backup, self.ini_path)
            os.remove(self.backup)

    def start(self, timeout=120.0):
        if udp_port_bound(P.INGEST_PORT, wildcard_only=True) or cymple_pids():
            raise RuntimeError("Cymple or cymple-linux is already running (UDP %d); close it first" % P.INGEST_PORT)
        out = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(BOTTLES + ["run", "-b", self.bottle, "-p", self.program],
                                     stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if udp_port_bound(P.INGEST_PORT, wildcard_only=True):
                return
            time.sleep(0.5)
        raise RuntimeError("Cymple did not bind UDP %d within %.0f s" % (P.INGEST_PORT, timeout))

    def stop(self, grace=15.0):
        """Ask Bottles to stop the bottle; Cymple ignores that end-session request, so after
        `grace` seconds its processes get SIGTERM (then SIGKILL). Never raises. Does nothing
        unless this object started Cymple, so a Cymple you opened yourself is never touched."""
        if self.proc is None:
            return
        stopper = subprocess.Popen(BOTTLES + ["stop", "-b", self.bottle], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and cymple_pids():
            time.sleep(0.5)
        for sig in (15, 9):
            pids = cymple_pids()
            if not pids:
                break
            log.info("Cymple did not exit on request; sending signal %d to %s", sig, pids)
            for pid in pids:
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass
            end = time.monotonic() + 10.0
            while time.monotonic() < end and cymple_pids():
                time.sleep(0.5)
        for p in (stopper, self.proc):
            if p is None:
                continue
            try:
                p.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                p.terminate()
        if cymple_pids():
            log.error("Cymple is still running (%s); stop it manually", cymple_pids())


class Sink(threading.Thread):
    """Collect datagrams on 127.0.0.1:<port> as (wall_ns, data)."""

    def __init__(self, port):
        super().__init__(name="sink-%d" % port, daemon=True)
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.1)
        self.items = []
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            self.items.append((time.time_ns(), data))
        self.sock.close()


def replay_to(path, addr=("127.0.0.1", P.INGEST_PORT), speed=1.0, stop=None):
    """Send every datagram of a capture to `addr`, paced like the original. Returns
    (wall_ns at replay start, capture ns of the first datagram)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t0_wall, t0_mono, first = time.time_ns(), time.monotonic(), None
    for ts, _src, data in read_capture(path):
        if stop is not None and stop.is_set():
            break
        first = ts if first is None else first
        delay = t0_mono + (ts - first) / 1e9 / speed - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        sock.sendto(data, addr)
    sock.close()
    return t0_wall, first


def record(capture, out_path, ini_path, mode="vrcft", settle_s=25.0, tail_s=2.0, cymple_log=None):
    from .outputs.osc import parse as parse_osc
    overrides, ports = MODES[mode]
    absorber = Absorber()
    absorber.start()
    bottle = CympleBottle(ini_path, log_path=cymple_log)
    sinks = [Sink(p) for p in ports]
    for s in sinks:
        s.start()
    t0_wall = first = None
    n = 0
    try:
        bottle.apply_ini(overrides)
        log.info("starting Cymple in Bottles...")
        bottle.start()
        log.info("Cymple is listening; letting it settle for %.0f s", settle_s)
        time.sleep(settle_s)
        log.info("replaying %s", capture)
        t0_wall, first = replay_to(capture)
        time.sleep(tail_s)
    finally:
        # order matters: save what we have before anything that can hang or fail
        for s in sinks:
            s.stop_event.set()
        if t0_wall is not None:
            n = _write_golden(out_path, capture, mode, t0_wall, first, sinks, parse_osc)
            log.info("%d oracle datagrams written to %s", n, out_path)
        try:
            bottle.stop()
        finally:
            bottle.restore_ini()
            absorber.stop_event.set()
    if absorber.count:
        log.info("absorbed %d live datagrams from the headset relay", absorber.count)
    return n


def _write_golden(out_path, capture, mode, t0_wall, first, sinks, parse_osc):
    n = 0
    with open(out_path, "w") as f:
        f.write(json.dumps({"capture": os.path.basename(capture), "mode": mode, "replay_start_ns": t0_wall,
                            "capture_t0_ns": first, "fields": FIELDS}) + "\n")
        for s in sinks:
            for wall, data in list(s.items):
                rec = {"port": s.port, "t_capture_ns": wall - t0_wall + first}
                pkt = decode_vrcft(data) if s.port == VRCFT_PORT else None
                if pkt is not None:
                    rec["flags"], rec["values"] = pkt
                elif s.port == OSC_PORT:
                    try:
                        rec["osc"] = parse_osc(data)
                    except (ValueError, IndexError, UnicodeDecodeError):
                        rec["raw"] = data.hex()
                else:
                    rec["raw"] = data.hex()
                f.write(json.dumps(rec) + "\n")
                n += 1
    return n


def load_golden(path):
    with open(path) as f:
        header = json.loads(next(f))
        rows = [json.loads(line) for line in f]
    pkts = sorted((r for r in rows if r["port"] == VRCFT_PORT and "values" in r), key=lambda r: r["t_capture_ns"])
    return header, pkts


def compare(capture, golden_path, pipeline, produce):
    """Run `pipeline` deterministically over `capture`; at each oracle packet time call
    `produce(pipeline, t_seconds)` -> {field: value} for the fields we implement (the
    original's packet timer fired then, so our packet builder runs at the same instants).
    Returns per-field stats."""
    _, pkts = load_golden(golden_path)
    pairs = {name: ([], []) for name in FIELDS}
    ingest = Ingest(pipeline.process_sync)
    k = 0
    for ts, src, data in read_capture(capture):
        while k < len(pkts) and pkts[k]["t_capture_ns"] <= ts:
            ours = produce(pipeline, pkts[k]["t_capture_ns"] / 1e9) or {}
            for name, v in ours.items():
                pairs[name][0].append(v)
                pairs[name][1].append(pkts[k]["values"][FIELDS.index(name)])
            k += 1
        ingest.handle(data, src, ts / 1e9)
    stats = {}
    for name, (a, b) in pairs.items():
        if not a:
            continue
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        err = np.abs(a - b)
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else math.nan
        stats[name] = {"n": int(len(a)), "mean_abs": float(err.mean()), "p95_abs": float(np.percentile(err, 95)),
                       "max_abs": float(err.max()), "corr": corr,
                       "oracle_range": [float(b.min()), float(b.max())]}
    return {"packets": len(pkts), "fields": stats}


# ---- direct-OSC oracle (mode "osc") ---------------------------------------------------------------

# Where the eye state is read back from Cymple's own OSC stream (addresses present on corpus1).
OSC_STATE = {"EyeYaw_L": "FT/v2/EyeLeftX", "EyeYaw_R": "FT/v2/EyeRightX",
             "EyePitch_L": "v2/EyeLeftY", "EyePitch_R": "v2/EyeRightY"}
for _side in ("Left", "Right"):
    OSC_STATE["EyeWide" + _side] = "v2/EyeWide" + _side
    OSC_STATE["EyeSquint" + _side] = "FT/v2/EyeSquint" + _side
    for _b in ("BrowLowerer", "BrowInnerUp", "BrowOuterUp", "BrowPinch"):
        OSC_STATE[_b + _side] = "v2/%s%s" % (_b, _side)
    _s, _u, _l = _side[0], "U" + _side[0], "D" + _side[0]
    OSC_STATE.update({
        "CheekPuff" + _side: "v2/CheekPuff" + _side, "CheekSuck" + _side: "v2/CheekSuck" + _side,
        "CheekSquint" + _side: "v2/CheekSquint" + _side, "Jaw" + _side: "v2/Jaw" + _side,
        "LipSuck" + _u: "v2/LipSuckUpper" + _side, "LipSuck" + _l: "v2/LipSuckLower" + _side,
        "MouthFunnel" + _u: "v2/LipFunnelUpper" + _side, "MouthFunnel" + _l: "v2/LipFunnelLower" + _side,
        "LipRaise" + _u: "v2/MouthUpperUp" + _side, "LipDepress" + _l: "v2/MouthLowerDown" + _side,
        "NoseSneer" + _side: "v2/NoseSneer" + _side, "MouthUpperShift" + _side: "v2/MouthUpper" + _side,
        "MouthLowerShift" + _side: "v2/MouthLower" + _side, "MouthSmile" + _side: "v2/MouthCornerPull" + _side,
        "MouthDimple" + _side: "v2/MouthDimple" + _side, "MouthFrown" + _side: "v2/MouthFrown" + _side,
        "MouthStretch" + _side: "FT/v2/MouthStretch" + _side, "Tongue" + _side: "v2/Tongue" + _side})
OSC_STATE.update({
    "JawOpen": "FT/v2/JawOpen", "MouthClosed": "FT/v2/MouthClosed", "JawFwd": "FT/v2/JawForward",
    "MouthPuckerUpper": "v2/LipPuckerUpperLeft", "MouthPuckerLower": "v2/LipPuckerLowerLeft",
    "MouthShrugUpper": "FT/v2/MouthRaiserUpper", "MouthShrugLower": "FT/v2/MouthRaiserLower",
    "TongueOut": "FT/v2/TongueOut", "TongueUp": "v2/TongueUp", "TongueDown": "v2/TongueDown",
    "TongueRoll": "v2/TongueRoll", "TongueCurlUp": "v2/TongueCurlUp", "TongueBendDown": "v2/TongueBendDown",
    "TongueWide": "v2/TongueFlat"})


def load_golden_osc(path):
    """[(t_capture_ns, [(address, args)])] sorted by time, and {address: type}."""
    rows, types = [], {}
    with open(path) as f:
        next(f)
        for line in f:
            r = json.loads(line)
            if "osc" in r:
                rows.append((r["t_capture_ns"], r["osc"]))
                for a, args in r["osc"]:
                    types[a] = bool if isinstance(args[0], bool) else float
    rows.sort(key=lambda r: r[0])
    return rows, types


def osc_formula_check(golden_path):
    """Check our parameter formulas against Cymple itself: rebuild its eye and mouth state from
    the OSC values it sent (latest value per address, one 10 ms tick at a time) and recompute every
    address we bind. Float differences up to Cymple's 1e-4 send deadband are expected (the
    source value may be stale by that much; pow(openness, 0.15) in the Squeeze parameters
    magnifies it near closed eyes; <name>Negative is skipped while <name> is within it of 0). Returns {address: {...}} and the number of ticks."""
    from .outputs.vrcft_params import PREFIX, resolve, unified_from_fields

    rows, types = load_golden_osc(golden_path)
    ticks, cur, t0 = [], [], None
    for t, msgs in rows:
        if cur and t - t0 > 3e6:
            ticks.append(cur)
            cur = []
        if not cur:
            t0 = t
        cur += msgs
    if cur:
        ticks.append(cur)
    bound = {b.address: b for b in resolve(list(types.items()))}
    state, err = {}, {}
    for msgs in ticks:
        for a, args in msgs:
            state[a] = args[0]
        f = {k: state.get(PREFIX + v, 0.0) for k, v in OSC_STATE.items()}
        for side in ("Left", "Right"):
            f["EyeLidClose" + side] = 1.0 - state.get(PREFIX + "v2/EyeOpen" + side, 1.0)
            f["Eye_Pupil_" + side] = 0.0
        u = unified_from_fields(f, face_active=True)
        for a, args in msgs:
            b = bound.get(a)
            if a.endswith("Negative") and abs(state.get(a[:-len("Negative")], 1.0)) < 2e-4:
                b = None                     # sign of a value inside the deadband: not recoverable
            if b is not None:
                v = b.value(u)
                err.setdefault(a, []).append(float(v != args[0]) if b.type is bool else
                                             max(abs(x - y) for x, y in zip(v, args)) if b.type is tuple else
                                             abs(v - args[0]))
    out = {}
    for a, e in err.items():
        e = np.asarray(e)
        out[a] = {"type": bound[a].type.__name__, "n": int(len(e)), "mean": float(e.mean()), "max": float(e.max())}
    return out, len(ticks)
