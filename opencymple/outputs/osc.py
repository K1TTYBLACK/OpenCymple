"""Minimal OSC 1.0 encoder and UDP sender, bundling the way Cymple does (sub_180105b70 /
sub_18013c230): "#bundle\\0" + immediate timetag (00..01) + [int32 BE size][message] elements,
at most 4096 bytes per bundle; a lone message is sent without a bundle."""
import socket
import struct

BUNDLE_HEAD = b"#bundle\x00" + b"\x00\x00\x00\x00\x00\x00\x00\x01"
MAX_BUNDLE = 4096


def _pad(b):
    return b + b"\x00" * (4 - len(b) % 4)


def _string(s):
    return _pad(s.encode("utf-8"))


def message(address, *args):
    """Encode one message. bool -> T/F, int -> i (int32), str -> s, float -> f (float32 BE)."""
    tags, data = ",", b""
    for a in args:
        if isinstance(a, bool):
            tags += "T" if a else "F"
        elif isinstance(a, int):
            tags += "i"
            data += struct.pack(">i", a)
        elif isinstance(a, str):
            tags += "s"
            data += _string(a)
        else:
            tags += "f"
            data += struct.pack(">f", float(a))
    return _string(address) + _string(tags) + data


def bundles(messages, limit=MAX_BUNDLE):
    """Pack encoded messages into datagrams: bundles of <= limit bytes, or the bare message
    when only one is left for a datagram."""
    out, cur = [], []
    size = len(BUNDLE_HEAD)
    for m in messages:
        if cur and size + 4 + len(m) > limit:
            out.append(cur)
            cur, size = [], len(BUNDLE_HEAD)
        cur.append(m)
        size += 4 + len(m)
    if cur:
        out.append(cur)
    return [group[0] if len(group) == 1 else
            BUNDLE_HEAD + b"".join(struct.pack(">i", len(m)) + m for m in group) for group in out]


def parse(datagram):
    """Decode a message or bundle into [(address, [args])] (for tests and the oracle)."""
    if datagram.startswith(b"#bundle\x00"):
        out, i = [], 16
        while i + 4 <= len(datagram):
            (n,) = struct.unpack_from(">i", datagram, i)
            out += parse(datagram[i + 4:i + 4 + n])
            i += 4 + n
        return out

    def read_str(i):
        end = datagram.index(b"\x00", i)
        return datagram[i:end].decode("utf-8"), (end // 4 + 1) * 4

    addr, i = read_str(0)
    tags, i = read_str(i)
    args = []
    for t in tags[1:]:
        if t == "f":
            args.append(struct.unpack_from(">f", datagram, i)[0])
            i += 4
        elif t == "i":
            args.append(struct.unpack_from(">i", datagram, i)[0])
            i += 4
        elif t in "TF":
            args.append(t == "T")
        elif t == "s":
            text, i = read_str(i)
            args.append(text)
        else:
            raise ValueError("unsupported OSC type tag %r" % t)
    return [(addr, args)]


class OscSender:
    def __init__(self, host="127.0.0.1", port=9000):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, messages):
        for datagram in bundles(messages):
            try:
                self.sock.sendto(datagram, self.addr)
            except OSError:
                pass

    def close(self):
        self.sock.close()
