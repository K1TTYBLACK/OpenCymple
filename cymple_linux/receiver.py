"""Camera stream receiver: UDP socket + discovery beacon, fragment reassembly, capture replay.

`Ingest` is the transport-independent core (protocol decode, per-stream reassembly as in
imgReassClass / sub_1800a5a00, device registry, statistics). `UdpSource` feeds it from the
network, `ReplaySource` from a capture file written by `--record` (or by cymple_udp_probe.py;
the CYMPCAP1 format is shared).
"""
import fcntl
import logging
import socket
import struct
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field

from . import protocol as P

log = logging.getLogger(__name__)

CAPTURE_MAGIC = b"CYMPCAP1"
CAPTURE_RECORD = struct.Struct("<Q4sHH")     # wall-clock ns, IPv4, source port, datagram length
MAX_SLOTS = 4                                # in-flight frames per stream
SLOT_TIMEOUT_S = 1.0
SIOCGIFADDR, SIOCGIFBRDADDR = 0x8915, 0x8919


@dataclass
class Frame:
    stream: int
    seq: int
    jpeg: bytes
    source: str
    t: float                                 # time.monotonic() when the last fragment arrived


@dataclass
class StreamStats:
    frames: int = 0
    bytes: int = 0
    seq_missing: int = 0
    incomplete: int = 0
    source: str = None
    times: deque = field(default_factory=lambda: deque(maxlen=100))

    def fps(self, now=None):
        t = self.times
        if len(t) < 2:
            return 0.0
        now = t[-1] if now is None else now
        if now - t[-1] > 1.0:                # stream stalled
            return 0.0
        return (len(t) - 1) / (t[-1] - t[0]) if t[-1] > t[0] else 0.0


class _Slot:
    __slots__ = ("total", "buf", "ranges", "t_first")

    def __init__(self, total, now):
        self.total, self.t_first = total, now
        self.buf = bytearray(total)
        self.ranges = []                     # sorted, merged [start, end)

    def add(self, offset, chunk):
        end = offset + len(chunk)
        self.buf[offset:end] = chunk
        merged = []
        for s, e in sorted(self.ranges + [[offset, end]]):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        self.ranges = merged
        return len(merged) == 1 and merged[0] == [0, self.total]


class _Reassembler:
    """One stream. Fragments with the same seq are merged by byte range; a changed total for a
    seq resets that slot; a change of source address resets the stream (sub_1800a5a00)."""

    def __init__(self):
        self.source = None
        self.slots = {}
        self.last_seq = None

    def add(self, frag, source, now, stats):
        if self.source != source:
            self.slots.clear()
            self.source = source
            self.last_seq = None
        if frag.offset == 0 and len(frag.chunk) == frag.total and frag.seq not in self.slots:
            return frag.chunk                # whole frame in one datagram (the observed case)
        for seq in [s for s, sl in self.slots.items() if now - sl.t_first > SLOT_TIMEOUT_S]:
            del self.slots[seq]
            stats.incomplete += 1
        slot = self.slots.get(frag.seq)
        if slot is not None and slot.total != frag.total:
            slot = None
        if slot is None:
            if len(self.slots) >= MAX_SLOTS:
                del self.slots[min(self.slots, key=lambda s: self.slots[s].t_first)]
                stats.incomplete += 1
            slot = self.slots[frag.seq] = _Slot(frag.total, now)
        if slot.add(frag.offset, frag.chunk):
            del self.slots[frag.seq]
            return bytes(slot.buf)
        return None


class Ingest:
    """Decode datagrams and emit complete JPEG frames through `on_frame(Frame)`.

    Each device kind belongs to one source IP (sub_180090f80): a type-5 fragment from another
    IP is dropped while the current owner was heard within the last 5 s; after that the new IP
    takes over. Type-1 fragments are processed first and then claim the device."""

    def __init__(self, on_frame):
        self.on_frame = on_frame
        self.lock = threading.Lock()
        self.streams = {}
        self.stats = {}
        self.errors = Counter()
        self.devices = {}                    # (ip, hardware) -> DeviceVersion
        self.headsets = {}                   # ip -> HeadsetInfo
        self.sources = set()
        self.owners = {}                     # device kind -> (ip, last seen)

    def _claim(self, stream, ip, now):
        kind = P.STREAM_DEVICE.get(stream)
        owner = self.owners.get(kind)
        if owner and owner[0] != ip and now - owner[1] < P.SOURCE_HOLD_S:
            return False
        if owner and owner[0] != ip:
            log.info("stream %d now comes from %s (was %s)", stream, ip, owner[0])
        self.owners[kind] = (ip, now)
        return True

    def handle(self, data, src, now):
        ip = src[0]
        try:
            msg = P.parse_datagram(data)
        except P.ProtocolError as e:
            with self.lock:
                self.errors[str(e)] += 1
            return
        if msg is None:
            return
        if isinstance(msg, P.DeviceVersion):
            with self.lock:
                if self.devices.get((ip, msg.hardware)) != msg:
                    log.info("device %s: %s firmware %d", ip, msg.name, msg.firmware)
                self.devices[(ip, msg.hardware)] = msg
                self.sources.add(ip)
            return
        if isinstance(msg, P.HeadsetInfo):
            with self.lock:
                if self.headsets.get(ip) != msg:
                    log.info("headset %s: %s %s (%s)", ip, msg.manufacturer, msg.model, msg.device)
                self.headsets[ip] = msg
                self.sources.add(ip)
            return
        with self.lock:
            if msg.ptype == P.TYPE_FRAGMENT_V5 and not self._claim(msg.stream, ip, now):
                self.errors["source_conflict"] += 1
                return
            if msg.ptype == P.TYPE_FRAGMENT_V1:
                self._claim(msg.stream, ip, now)
            self.sources.add(ip)
            st = self.stats.setdefault(msg.stream, StreamStats())
            jpeg = self.streams.setdefault(msg.stream, _Reassembler()).add(msg, ip, now, st)
            if jpeg is None:
                return
            r = self.streams[msg.stream]
            if r.last_seq is not None:
                st.seq_missing += (msg.seq - r.last_seq - 1) % 256
            r.last_seq = msg.seq
            st.frames += 1
            st.bytes += len(jpeg)
            st.source = ip
            st.times.append(now)
        self.on_frame(Frame(msg.stream, msg.seq, jpeg, ip, now))

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            return {
                "streams": {s: {"fps": round(st.fps(now), 1), "frames": st.frames,
                                "missing": st.seq_missing, "incomplete": st.incomplete,
                                "source": st.source} for s, st in sorted(self.stats.items())},
                "errors": dict(self.errors),
                "devices": {"%s/%s" % (ip, d.name): d.firmware for (ip, _), d in self.devices.items()},
                "headsets": {ip: "%s %s" % (h.manufacturer, h.model) for ip, h in self.headsets.items()},
            }


def ipv4_broadcast_addresses():
    """(interface, address, broadcast) for every non-loopback IPv4 interface."""
    out = []
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for _, name in socket.if_nameindex():
            req = struct.pack("256s", name.encode()[:15])
            try:
                addr = socket.inet_ntoa(fcntl.ioctl(probe.fileno(), SIOCGIFADDR, req)[20:24])
                brd = socket.inet_ntoa(fcntl.ioctl(probe.fileno(), SIOCGIFBRDADDR, req)[20:24])
            except OSError:
                continue
            if not addr.startswith("127.") and brd != "0.0.0.0":
                out.append((name, addr, brd))
    finally:
        probe.close()
    return out


class UdpSource(threading.Thread):
    """Own UDP 23303, announce this PC on 23302 once per `beacon_interval`, feed `ingest`."""

    def __init__(self, ingest, bind="0.0.0.0", port=P.INGEST_PORT, beacon=True, beacon_to=(),
                 beacon_interval=1.0, record_path=None):
        super().__init__(name="udp-ingest", daemon=True)
        self.ingest = ingest
        self.beacon, self.beacon_interval = beacon, beacon_interval
        self.extra_targets = list(beacon_to)
        self.stop_event = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        try:
            self.sock.bind((bind, port))
        except OSError as e:
            self.sock.close()
            raise OSError("cannot bind UDP %s:%d (%s); is Cymple or another tracker running?"
                          % (bind, port, e)) from e
        self.sock.settimeout(0.05)
        self.record = None
        if record_path:
            self.record = open(record_path, "wb")
            self.record.write(CAPTURE_MAGIC)

    def beacon_targets(self):
        targets = [brd for _, _, brd in ipv4_broadcast_addresses()] + self.extra_targets
        with self.ingest.lock:
            targets += sorted(self.ingest.sources)
        return list(dict.fromkeys(targets))

    def send_beacons(self):
        for addr in self.beacon_targets():
            try:
                self.sock.sendto(P.BEACON, (addr, P.BEACON_PORT))
            except OSError as e:
                log.debug("beacon to %s failed: %s", addr, e)

    def run(self):
        next_beacon = 0.0
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                if self.beacon and now >= next_beacon:
                    self.send_beacons()
                    next_beacon = now + self.beacon_interval
                try:
                    data, src = self.sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    if self.stop_event.is_set():
                        break
                    raise
                if self.record:
                    self.record.write(CAPTURE_RECORD.pack(time.time_ns(), socket.inet_aton(src[0]),
                                                          src[1], len(data)) + data)
                self.ingest.handle(data, src, time.monotonic())
        finally:
            self.sock.close()
            if self.record:
                self.record.close()

    def stop(self):
        self.stop_event.set()


def read_capture(path):
    """Yield (wall_ns, (ip, port), datagram) from a CYMPCAP1 file."""
    with open(path, "rb") as f:
        if f.read(len(CAPTURE_MAGIC)) != CAPTURE_MAGIC:
            raise ValueError("%s is not a Cymple capture (missing CYMPCAP1 header)" % path)
        while True:
            h = f.read(CAPTURE_RECORD.size)
            if len(h) < CAPTURE_RECORD.size:
                return
            ts, ip, port, n = CAPTURE_RECORD.unpack(h)
            data = f.read(n)
            if len(data) < n:
                return
            yield ts, (socket.inet_ntoa(ip), port), data


class ReplaySource(threading.Thread):
    """Feed a capture into `ingest`, paced at `speed` x real time (0 = as fast as possible)."""

    def __init__(self, ingest, path, speed=1.0, loop=False):
        super().__init__(name="replay-ingest", daemon=True)
        self.ingest, self.path, self.speed, self.loop = ingest, path, speed, loop
        self.stop_event = threading.Event()
        self.finished = threading.Event()

    def run(self):
        try:
            while not self.stop_event.is_set():
                t0 = time.monotonic()
                first = None
                for ts, src, data in read_capture(self.path):
                    if self.stop_event.is_set():
                        return
                    first = ts if first is None else first
                    if self.speed > 0:
                        delay = t0 + (ts - first) / 1e9 / self.speed - time.monotonic()
                        if delay > 0:
                            self.stop_event.wait(delay)
                    self.ingest.handle(data, src, time.monotonic())
                if not self.loop:
                    return
        finally:
            self.finished.set()

    def stop(self):
        self.stop_event.set()
