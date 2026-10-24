import json
import struct

import pytest

from cymple_linux import protocol as P
from cymple_linux.receiver import Ingest


def test_parse_v5_fragment():
    frag = P.parse_datagram(P.build_fragment_v5(2, 17, b"\xff\xd8abc\xff\xd9"))
    assert (frag.ptype, frag.stream, frag.seq, frag.offset, frag.total) == (5, 2, 17, 0, 7)
    assert frag.chunk == b"\xff\xd8abc\xff\xd9"


def test_parse_v1_fragment():
    chunk = b"0123456789"
    data = struct.pack("<HHIIBBH", 1, 12 + len(chunk), 5, 40, 3, 9, 0) + chunk
    frag = P.parse_datagram(data)
    assert (frag.ptype, frag.stream, frag.seq, frag.offset, frag.total, frag.chunk) == (1, 3, 9, 5, 40, chunk)


@pytest.mark.parametrize("data,reason", [
    (b"\x05\x00", "short"),
    (P.build_fragment_v5(0, 0, b"abc")[:-1], "length_mismatch"),
    (P.build_fragment_v5(5, 0, b"abc"), "stream_out_of_range"),
    (P.build_fragment_v5(0, 0, b"abc", offset=10, total=11), "fragment_bounds"),
    (struct.pack("<HH", 9, 0), "unknown_type_9"),
])
def test_rejects(data, reason):
    with pytest.raises(P.ProtocolError, match=reason):
        P.parse_datagram(data)


def test_v5_chunk_len_must_match():
    data = bytearray(P.build_fragment_v5(0, 0, b"abcd"))
    struct.pack_into("<H", data, 8, 3)
    with pytest.raises(P.ProtocolError, match="v5_chunk_len"):
        P.parse_datagram(bytes(data))


def test_version_and_headset():
    v = P.parse_datagram(P.build_version(10, 6))
    assert (v.hardware, v.firmware, v.name) == (10, 6, "EYE_V3")
    body = json.dumps({"kind": "headset_info", "schema": 1, "model": "Quest 3", "manufacturer": "Oculus",
                       "device": "eureka", "product": "eureka"}).encode()
    h = P.parse_datagram(struct.pack("<HH", 0, len(body)) + body)
    assert h.model == "Quest 3" and h.device == "eureka"
    bad = json.dumps({"kind": "headset_info", "schema": 2}).encode()
    with pytest.raises(P.ProtocolError, match="headset_schema"):
        P.parse_datagram(struct.pack("<HH", 0, len(bad)) + bad)
    assert P.parse_datagram(P.BEACON) is None


def _collect():
    frames = []
    return frames, Ingest(frames.append)


def test_reassembly_out_of_order_and_overlap():
    frames, ing = _collect()
    jpeg = bytes(range(256)) * 20
    parts = [(1000, 3000), (0, 1200), (2900, len(jpeg)), (3000, 4000)]
    for off, end in parts:
        ing.handle(P.build_fragment_v5(1, 42, jpeg[off:end], offset=off, total=len(jpeg)), ("10.0.0.2", 5000), 0.0)
    assert len(frames) == 1 and frames[0].jpeg == jpeg and frames[0].stream == 1


def test_source_change_resets_partial_frame():
    frames, ing = _collect()
    jpeg = b"x" * 100
    ing.handle(P.build_fragment_v5(0, 1, jpeg[:50], 0, 100), ("10.0.0.2", 1), 0.0)
    ing.handle(P.build_fragment_v5(0, 1, jpeg[50:], 50, 100), ("10.0.0.3", 1), 0.0)
    assert frames == []


def test_seq_gaps():
    frames, ing = _collect()
    for seq in (254, 255, 1):                 # wraps; seq 0 missing
        ing.handle(P.build_fragment_v5(0, seq, b"abc"), ("10.0.0.2", 1), 0.0)
    assert len(frames) == 3 and ing.stats[0].seq_missing == 1


def test_device_sticks_to_source_for_five_seconds():
    frames, ing = _collect()
    ing.handle(P.build_fragment_v5(0, 0, b"abc"), ("10.0.0.2", 1), 0.0)
    ing.handle(P.build_fragment_v5(0, 1, b"abc"), ("127.0.0.1", 1), 4.9)    # owner still fresh
    ing.handle(P.build_fragment_v5(1, 1, b"abc"), ("127.0.0.1", 1), 4.9)    # other device: free
    ing.handle(P.build_fragment_v5(0, 2, b"abc"), ("127.0.0.1", 1), 5.0)    # owner stale: take over
    assert [(f.stream, f.source) for f in frames] == [(0, "10.0.0.2"), (1, "127.0.0.1"), (0, "127.0.0.1")]
    assert ing.errors["source_conflict"] == 1
