#!/usr/bin/env python3
"""Cymple UDP probe: capture and reassemble eye/mouth camera frames on Linux.

Protocol recovered from cympleface.dll (Cymple 4.18.0, ProtocolIngress / imgReassClass).
All integers are little-endian.

  every datagram   u16 type, u16 payload_len        (datagram size == payload_len + 4)
  type 0           headset info JSON (<= 2048 bytes)
  type 1           JPEG fragment, 16-byte header:
                     @4 u32 offset, @8 u32 total, @12 u8 stream, @13 u8 seq, @14..15 unused
  type 5           JPEG fragment, 12-byte header:
                     @4 u16 offset, @6 u16 total, @8 u16 chunk_len, @10 u8 stream, @11 u8 seq
  type 6           device version, 28 bytes:
                     @4 u32 0xAA55AA55, @8 u32 hardware, @12 u32 firmware, @24 u32 0x55AA55AA

Fragments with the same (stream, seq) are merged by byte range; a frame is complete when
[0, total) is covered. The PC announces itself by sending 00 00 00 00 from its 23303 socket
to UDP 23302 (every IPv4 broadcast address plus every known device). Devices may not
stream until they hear that beacon, so this probe sends it too (disable with --no-beacon).

Close Cymple before running: only one program should own UDP 23303.
"""
import argparse
import fcntl
import json
import os
import select
import socket
import struct
import sys
import time
from collections import Counter, deque

BEACON = b"\x00\x00\x00\x00"
MAX_STREAMS = 5          # the original rejects stream ids >= 5
MAX_SLOTS = 4            # in-flight frames kept per stream
SLOT_TIMEOUT_S = 1.0
KNOWN_HARDWARE = {3: "MOUTH_SINGLE_V1", 10: "EYE_V3"}   # only ids seen in Cymple logs
CAPTURE_MAGIC = b"CYMPCAP1"
SIOCGIFADDR = 0x8915
SIOCGIFBRDADDR = 0x8919


def ipv4_broadcast_addresses():
    """Broadcast address of every non-loopback IPv4 interface (what the original targets)."""
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
            if addr.startswith("127.") or brd == "0.0.0.0":
                continue
            out.append((name, addr, brd))
    finally:
        probe.close()
    return out


def jpeg_info(data):
    """Return (ok, info) from a light JPEG marker walk; stops at SOS."""
    info = {"markers": []}
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return False, {"error": "no SOI", "head": data[:8].hex()}
    tail = data[-16:]
    info["eoi"] = b"\xff\xd9" in tail
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            info["error"] = "expected marker at %d, got 0x%02x" % (i, data[i])
            return False, info
        while i < len(data) and data[i] == 0xFF:
            i += 1
        if i >= len(data):
            break
        m = data[i]
        i += 1
        if len(info["markers"]) < 16:
            info["markers"].append("%02X" % m)
        if m == 0x01 or 0xD0 <= m <= 0xD7:
            continue
        if m == 0xDA:
            break
        seg_len = struct.unpack_from(">H", data, i)[0]
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            _, h, w, nc = struct.unpack_from(">BHHB", data, i + 2)
            info.update(width=w, height=h, components=nc, sof="%02X" % m)
        i += seg_len
    ok = "width" in info and info["eoi"]
    if not ok and "error" not in info:
        info["error"] = "missing SOF" if "width" not in info else "missing EOI"
    return ok, info


class Slot:
    __slots__ = ("seq", "total", "buf", "ranges", "t_first", "frags")

    def __init__(self, seq, total, now):
        self.seq, self.total, self.t_first, self.frags = seq, total, now, 0
        self.buf = bytearray(total)
        self.ranges = []          # sorted, merged [start, end)

    def add(self, offset, chunk):
        end = offset + len(chunk)
        self.buf[offset:end] = chunk
        self.frags += 1
        merged = []
        for s, e in sorted(self.ranges + [[offset, end]]):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        self.ranges = merged

    def complete(self):
        return len(self.ranges) == 1 and self.ranges[0] == [0, self.total]


class StreamState:
    def __init__(self, sid):
        self.sid = sid
        self.source = None
        self.slots = {}
        self.recent_done = deque(maxlen=8)   # seqs just completed, to spot late duplicates
        self.stats = Counter()
        self.frag_sizes = Counter()
        self.last_seq = None
        self.last_info = {}
        self.first_markers = None
        self.frame_times = []
        self.saved = 0
        self.saved_bad = 0
        self.bytes_ok = 0


class Probe:
    def __init__(self, args):
        self.args = args
        self.out = args.out
        os.makedirs(self.out, exist_ok=True)
        self.streams = {}
        self.types = Counter()
        self.header_oddities = Counter()
        self.unknown_logged = Counter()
        self.devices = {}
        self.headsets = {}
        self.sources = set()
        self.packets_logged = 0
        self.pktlog = open(os.path.join(self.out, "packets.log"), "w")
        self.t0 = time.monotonic()

    # ---- helpers ---------------------------------------------------------
    def log_packet(self, src, data, note=""):
        if self.packets_logged < self.args.log_packets:
            self.packets_logged += 1
            self.pktlog.write("%9.3f %-21s len=%-5d %s %s\n" % (
                time.monotonic() - self.t0, "%s:%d" % src, len(data), data[:32].hex(" "), note))

    def stream(self, sid):
        if sid not in self.streams:
            self.streams[sid] = StreamState(sid)
        return self.streams[sid]

    # ---- datagram dispatch ----------------------------------------------
    def handle(self, data, src, now):
        self.sources.add(src[0])
        if len(data) < 4:
            self.types["short"] += 1
            self.log_packet(src, data, "SHORT")
            return
        ptype, plen = struct.unpack_from("<HH", data, 0)
        self.types[ptype] += 1
        if len(data) != plen + 4:
            self.header_oddities["outer_len_mismatch"] += 1
            self.log_packet(src, data, "OUTER-LEN-MISMATCH (payload_len=%d)" % plen)
            return
        if ptype == 0:
            self.on_headset(data, src)
        elif ptype == 1:
            self.on_frag_v1(data, src, now)
        elif ptype == 5:
            self.on_frag_v5(data, src, now)
        elif ptype == 6:
            self.on_version(data, src)
        else:
            if self.unknown_logged[ptype] < 20:
                self.unknown_logged[ptype] += 1
                self.pktlog.write("UNKNOWN type=%d from %s len=%d: %s\n" % (
                    ptype, src[0], len(data), data[:64].hex(" ")))
            self.log_packet(src, data, "type=%d" % ptype)

    def on_headset(self, data, src):
        self.log_packet(src, data, "HEADSET")
        try:
            info = json.loads(data[4:].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            self.header_oddities["headset_json_invalid"] += 1
            print("[headset] invalid JSON from %s: %s" % (src[0], e))
            return
        if self.headsets.get(src[0]) != info:
            print("[headset] %s: %s" % (src[0], json.dumps(info)))
        self.headsets[src[0]] = info

    def on_version(self, data, src):
        self.log_packet(src, data, "VERSION")
        if len(data) != 28:
            self.header_oddities["version_size"] += 1
            return
        m1, hw, fw = struct.unpack_from("<III", data, 4)
        m2 = struct.unpack_from("<I", data, 24)[0]
        rec = {"hardware": hw, "hardware_name": KNOWN_HARDWARE.get(hw), "firmware": fw,
               "markers_ok": m1 == 0xAA55AA55 and m2 == 0x55AA55AA,
               "reserved_hex": data[16:24].hex()}
        key = (src[0], hw)
        if self.devices.get(key) != rec:
            print("[version] %s: hardware=%d (%s) firmware=%d markers_ok=%s" % (
                src[0], hw, rec["hardware_name"] or "?", fw, rec["markers_ok"]))
        self.devices[key] = rec

    def on_frag_v1(self, data, src, now):
        if len(data) < 16:
            self.header_oddities["v1_short"] += 1
            return
        offset, total, sid, seq = struct.unpack_from("<IIBB", data, 4)
        tail = data[14:16]
        if tail != b"\x00\x00":
            self.header_oddities["v1_bytes14_15=" + tail.hex()] += 1
        self.log_packet(src, data, "FRAG1 s=%d seq=%d off=%d tot=%d" % (sid, seq, offset, total))
        self.on_fragment(1, sid, seq, offset, total, data[16:], src, now)

    def on_frag_v5(self, data, src, now):
        if len(data) < 12:
            self.header_oddities["v5_short"] += 1
            return
        offset, total, clen, sid, seq = struct.unpack_from("<HHHBB", data, 4)
        if clen + 12 != len(data):
            self.header_oddities["v5_chunk_len_mismatch"] += 1
            self.log_packet(src, data, "FRAG5-CHUNKLEN-MISMATCH")
            return
        self.log_packet(src, data, "FRAG5 s=%d seq=%d off=%d tot=%d" % (sid, seq, offset, total))
        self.on_fragment(5, sid, seq, offset, total, data[12:], src, now)

    # ---- reassembly ------------------------------------------------------
    def on_fragment(self, ftype, sid, seq, offset, total, chunk, src, now):
        if sid >= MAX_STREAMS:
            self.header_oddities["stream_id_out_of_range=%d" % sid] += 1
            return
        clen = len(chunk)
        if clen == 0 or total == 0 or total > 0xFFFF or offset >= total or clen > total - offset:
            self.header_oddities["frag_bounds_invalid"] += 1
            return
        st = self.stream(sid)
        st.stats["frags_type%d" % ftype] += 1
        st.frag_sizes[clen] += 1
        if st.source != src[0]:
            if st.source is not None:
                st.stats["source_switches"] += 1
                st.slots.clear()
            st.source = src[0]
        for key in [k for k, s in st.slots.items() if now - s.t_first > SLOT_TIMEOUT_S]:
            del st.slots[key]
            st.stats["incomplete_timeout"] += 1
        slot = st.slots.get(seq)
        if slot is not None and slot.total != total:
            st.stats["total_changed_same_seq"] += 1
            slot = None
        if slot is None and seq in st.recent_done:
            # Cymple itself would open a new partial frame here; count it instead.
            st.stats["late_fragment_after_complete"] += 1
            return
        if slot is None:
            if len(st.slots) >= MAX_SLOTS:
                oldest = min(st.slots, key=lambda k: st.slots[k].t_first)
                del st.slots[oldest]
                st.stats["incomplete_evicted"] += 1
            slot = st.slots[seq] = Slot(seq, total, now)
        slot.add(offset, chunk)
        if slot.complete():
            del st.slots[seq]
            st.recent_done.append(seq)
            self.on_frame(st, slot, now)

    def on_frame(self, st, slot, now):
        frame = bytes(slot.buf)
        ok, info = jpeg_info(frame)
        if st.last_seq is not None:
            gap = (slot.seq - st.last_seq) % 256
            if gap != 1:
                st.stats["seq_gaps"] += 1
                st.stats["seq_missing"] += max(gap - 1, 0)
        st.last_seq = slot.seq
        st.stats["frags_per_frame_sum"] += slot.frags
        if ok:
            st.stats["frames_ok"] += 1
            st.bytes_ok += len(frame)
            st.last_info = {k: info[k] for k in ("width", "height", "components", "sof")}
            if st.first_markers is None:
                st.first_markers = info["markers"]
            st.frame_times.append(now)
            st.frame_times = st.frame_times[-120:]
            n = st.stats["frames_ok"]
            if st.saved < self.args.max_saved and (
                    n <= self.args.save_first or n % self.args.save_every == 0):
                st.saved += 1
                self.save(st, frame, "frame_%06d_seq%03d.jpg" % (n, slot.seq))
        else:
            st.stats["frames_bad_jpeg"] += 1
            st.stats["bad:" + info.get("error", "?").split(" at ")[0]] += 1
            if st.saved_bad < 10:
                st.saved_bad += 1
                self.save(st, frame, "bad_%06d_seq%03d.bin" % (st.stats["frames_bad_jpeg"], slot.seq))

    def save(self, st, frame, name):
        d = os.path.join(self.out, "stream%d_%s" % (st.sid, (st.source or "unknown").replace(":", "_")))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name), "wb") as f:
            f.write(frame)

    # ---- reporting -------------------------------------------------------
    def fps(self, st):
        t = st.frame_times
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0

    def status_line(self):
        parts = ["t=%5.1fs types=%s" % (time.monotonic() - self.t0, dict(self.types))]
        for sid in sorted(self.streams):
            st = self.streams[sid]
            dims = "%sx%s" % (st.last_info.get("width", "?"), st.last_info.get("height", "?"))
            parts.append("s%d[%s ok=%d bad=%d lost=%d %.1ffps %s]" % (
                sid, st.source, st.stats["frames_ok"], st.stats["frames_bad_jpeg"],
                st.stats["incomplete_evicted"] + st.stats["incomplete_timeout"], self.fps(st), dims))
        return " ".join(parts)

    def summary(self):
        streams = {}
        for sid, st in sorted(self.streams.items()):
            ok = st.stats["frames_ok"]
            done = ok + st.stats["frames_bad_jpeg"]
            streams[str(sid)] = {
                "source": st.source,
                "stats": dict(st.stats),
                "fps_recent": round(self.fps(st), 2),
                "avg_frame_bytes": round(st.bytes_ok / ok) if ok else None,
                "avg_frags_per_frame": round(st.stats["frags_per_frame_sum"] / done, 2) if done else None,
                "jpeg": st.last_info,
                "first_frame_markers": st.first_markers,
                "fragment_sizes_top": st.frag_sizes.most_common(5),
                "saved_frames": st.saved,
            }
        return {
            "duration_s": round(time.monotonic() - self.t0, 1),
            "packet_types": {str(k): v for k, v in self.types.items()},
            "header_oddities": dict(self.header_oddities),
            "sources": sorted(self.sources),
            "headsets": self.headsets,
            "devices": {"%s/hw%d" % k: v for k, v in self.devices.items()},
            "streams": streams,
        }

    def close(self):
        for st in self.streams.values():
            st.stats["incomplete_at_exit"] += len(st.slots)
            st.slots.clear()
        self.pktlog.close()
        s = self.summary()
        with open(os.path.join(self.out, "summary.json"), "w") as f:
            json.dump(s, f, indent=2)
        return s


def send_beacons(sock, targets, devices):
    sent = 0
    for addr in list(targets) + sorted(devices):
        try:
            sock.sendto(BEACON, (addr, 23302))
            sent += 1
        except OSError as e:
            print("[beacon] send to %s failed: %s" % (addr, e), file=sys.stderr)
    return sent


def run_live(args, probe):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    try:
        sock.bind((args.bind, args.port))
    except OSError as e:
        sys.exit("cannot bind UDP %s:%d (%s). Is Cymple still running?" % (args.bind, args.port, e))
    rec = None
    if args.record:
        rec = open(args.record, "wb")
        rec.write(CAPTURE_MAGIC)
    targets = []
    if not args.no_beacon:
        for name, addr, brd in ipv4_broadcast_addresses():
            print("[beacon] %s %s -> %s:23302" % (name, addr, brd))
            targets.append(brd)
        targets += args.beacon_to
        if not targets:
            print("[beacon] no IPv4 broadcast address found; use --beacon-to", file=sys.stderr)
    print("listening on %s:%d, writing to %s (Ctrl+C to stop)" % (args.bind, args.port, args.out))
    next_beacon = next_status = time.monotonic()
    deadline = time.monotonic() + args.duration if args.duration else None
    try:
        while deadline is None or time.monotonic() < deadline:
            now = time.monotonic()
            if targets and now >= next_beacon:
                send_beacons(sock, targets, probe.sources)
                next_beacon = now + args.beacon_interval
            if now >= next_status:
                print(probe.status_line(), flush=True)
                next_status = now + args.status_interval
            r, _, _ = select.select([sock], [], [], 0.05)
            while r:
                data, src = sock.recvfrom(65535)
                t = time.monotonic()
                if rec:
                    rec.write(struct.pack("<Q4sHH", time.time_ns(), socket.inet_aton(src[0]),
                                          src[1], len(data)) + data)
                probe.handle(data, src, t)
                r, _, _ = select.select([sock], [], [], 0)
    except KeyboardInterrupt:
        pass
    finally:
        if rec:
            rec.close()
        sock.close()


def run_replay(args, probe):
    with open(args.replay, "rb") as f:
        if f.read(len(CAPTURE_MAGIC)) != CAPTURE_MAGIC:
            sys.exit("%s is not a cymple capture" % args.replay)
        hdr = struct.calcsize("<Q4sHH")
        first_ns = None
        while True:
            h = f.read(hdr)
            if len(h) < hdr:
                break
            ts, ip, port, n = struct.unpack("<Q4sHH", h)
            data = f.read(n)
            first_ns = first_ns or ts
            probe.handle(data, (socket.inet_ntoa(ip), port), (ts - first_ns) / 1e9)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=23303)
    ap.add_argument("--out", default=time.strftime("cymple-capture-%Y%m%d-%H%M%S"))
    ap.add_argument("--duration", type=float, default=0, help="seconds to run (0 = until Ctrl+C)")
    ap.add_argument("--no-beacon", action="store_true", help="do not announce this PC on UDP 23302")
    ap.add_argument("--beacon-to", action="append", default=[], metavar="IP",
                    help="extra beacon target (e.g. the headset IP); repeatable")
    ap.add_argument("--beacon-interval", type=float, default=1.0)
    ap.add_argument("--status-interval", type=float, default=2.0)
    ap.add_argument("--save-first", type=int, default=5, help="save the first N frames per stream")
    ap.add_argument("--save-every", type=int, default=60, help="then every Nth frame")
    ap.add_argument("--max-saved", type=int, default=40, help="cap on saved frames per stream")
    ap.add_argument("--log-packets", type=int, default=300, help="header lines to write to packets.log")
    ap.add_argument("--record", metavar="FILE", help="also save every raw datagram to FILE")
    ap.add_argument("--replay", metavar="FILE", help="process a --record file instead of the network")
    args = ap.parse_args()

    probe = Probe(args)
    if args.replay:
        run_replay(args, probe)
    else:
        run_live(args, probe)
    s = probe.close()
    print(probe.status_line())
    print(json.dumps(s, indent=2))
    print("output: %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
