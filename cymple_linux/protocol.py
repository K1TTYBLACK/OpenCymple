"""Cymple camera-board wire protocol: UDP ingest on 23303, discovery beacon to 23302.

Recovered from cympleface.dll 4.18.0 (ProtocolIngress::decodeMsg sub_18009f8f0, fragment
parser sub_1800a6cc0, beacon sub_1800a1000). All integers are little-endian.

  every datagram   u16 type, u16 payload_len            (size must equal payload_len + 4)
  type 0           headset info JSON (<= 2048 B): kind == "headset_info", schema == 1
  type 1           JPEG fragment, 16-byte header: @4 u32 offset, @8 u32 total, @12 u8 stream, @13 u8 seq
  type 5           JPEG fragment, 12-byte header: @4 u16 offset, @6 u16 total, @8 u16 chunk_len
                   (== size - 12), @10 u8 stream, @11 u8 seq
  type 6           device version, 28 B: @4 0xAA55AA55, @8 u32 hardware, @12 u32 firmware, @24 0x55AA55AA

The PC must send 00 00 00 00 from its 23303 socket to UDP 23302 (every IPv4 broadcast address
plus every known device); boards stay silent until they hear it.
"""
import json
import struct
from dataclasses import dataclass

INGEST_PORT = 23303
BEACON_PORT = 23302
BEACON = b"\x00\x00\x00\x00"
MAX_STREAMS = 5                  # sub_1800a6cc0 rejects stream >= 5
MAX_TOTAL = 0xFFFF
MAX_HEADSET_JSON = 0x800

TYPE_HEADSET, TYPE_FRAGMENT_V1, TYPE_FRAGMENT_V5, TYPE_VERSION = 0, 1, 5, 6

# sub_1800b14e0 maps stream id -> device kind; sub_1800b1e30 names the kinds
# ("left eye tracker", "right eye tracker", "face tracker", "face tracker V2").
STREAM_DEVICE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
STREAM_KIND = {0: "left_eye", 1: "right_eye", 2: "mouth_v1", 3: "mouth_v2", 4: "mouth_v2"}
SOURCE_HOLD_S = 5.0              # sub_180090f80: a device kind sticks to its source IP for 5000 ms
EYE_STREAMS = {0: 0, 1: 1}       # stream id -> eye index (0 = left, 1 = right)
MOUTH_V1_STREAM = 2

HARDWARE_NAMES = {3: "MOUTH_SINGLE_V1", 10: "EYE_V3"}   # only the ids seen in Cymple logs


class ProtocolError(ValueError):
    """Malformed datagram. `str(e)` is a short reason usable as a counter key."""


@dataclass(frozen=True)
class Fragment:
    ptype: int
    stream: int
    seq: int
    offset: int
    total: int
    chunk: bytes


@dataclass(frozen=True)
class HeadsetInfo:
    model: str
    manufacturer: str
    device: str
    product: str


@dataclass(frozen=True)
class DeviceVersion:
    hardware: int
    firmware: int

    @property
    def name(self):
        return HARDWARE_NAMES.get(self.hardware, "hw%d" % self.hardware)


def _fragment(ptype, stream, seq, offset, total, chunk):
    if stream >= MAX_STREAMS:
        raise ProtocolError("stream_out_of_range")
    n = len(chunk)
    if n == 0 or total == 0 or total > MAX_TOTAL or offset > total or n > total - offset:
        raise ProtocolError("fragment_bounds")
    return Fragment(ptype, stream, seq, offset, total, bytes(chunk))


def parse_datagram(data):
    """Return a Fragment, HeadsetInfo or DeviceVersion; None for datagrams the original ignores
    (empty type 0, i.e. a beacon). Raises ProtocolError for anything the original rejects."""
    if len(data) < 4:
        raise ProtocolError("short")
    ptype, plen = struct.unpack_from("<HH", data, 0)
    if len(data) != plen + 4:
        raise ProtocolError("length_mismatch")
    if ptype == TYPE_FRAGMENT_V5:
        if len(data) < 12:
            raise ProtocolError("v5_short")
        offset, total, clen, stream, seq = struct.unpack_from("<HHHBB", data, 4)
        if clen != len(data) - 12:
            raise ProtocolError("v5_chunk_len")
        return _fragment(ptype, stream, seq, offset, total, memoryview(data)[12:])
    if ptype == TYPE_FRAGMENT_V1:
        if len(data) < 16:
            raise ProtocolError("v1_short")
        offset, total, stream, seq = struct.unpack_from("<IIBB", data, 4)
        return _fragment(ptype, stream, seq, offset, total, memoryview(data)[16:])
    if ptype == TYPE_VERSION:
        if len(data) != 28:
            raise ProtocolError("version_size")
        m1, hw, fw = struct.unpack_from("<III", data, 4)
        (m2,) = struct.unpack_from("<I", data, 24)
        if m1 != 0xAA55AA55 or m2 != 0x55AA55AA:
            raise ProtocolError("version_marker")
        return DeviceVersion(hw, fw)
    if ptype == TYPE_HEADSET:
        if plen == 0:
            return None
        if plen > MAX_HEADSET_JSON:
            raise ProtocolError("headset_too_large")
        try:
            obj = json.loads(bytes(data[4:]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ProtocolError("headset_json") from None
        if not isinstance(obj, dict) or obj.get("kind") != "headset_info" or obj.get("schema") != 1:
            raise ProtocolError("headset_schema")
        return HeadsetInfo(*(str(obj.get(k, "")) for k in ("model", "manufacturer", "device", "product")))
    raise ProtocolError("unknown_type_%d" % ptype)


def build_fragment_v5(stream, seq, jpeg, offset=0, total=None):
    """Encode one type-5 fragment (used by tests and the synthetic sender)."""
    total = len(jpeg) + offset if total is None else total
    return struct.pack("<HHHHHBB", TYPE_FRAGMENT_V5, len(jpeg) + 8, offset, total, len(jpeg),
                       stream, seq) + bytes(jpeg)


def build_version(hardware, firmware):
    return struct.pack("<HHIII8sI", TYPE_VERSION, 24, 0xAA55AA55, hardware, firmware, b"\0" * 8, 0x55AA55AA)
