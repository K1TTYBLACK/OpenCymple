import struct

from cymple_linux.outputs.osc import BUNDLE_HEAD, bundles, message, parse


def test_message_encoding_matches_osc_spec():
    m = message("/avatar/parameters/v2/EyeX", 0.5)
    assert m == b"/avatar/parameters/v2/EyeX\x00\x00" + b",f\x00\x00" + struct.pack(">f", 0.5)
    assert len(message("/a", True, 3, 1.0)) % 4 == 0
    assert parse(message("/a", True, False, 3, 1.5)) == [("/a", [True, False, 3, 1.5])]


def test_single_message_is_sent_bare_and_bundles_respect_limit():
    one = message("/x", 1.0)
    assert bundles([one]) == [one]
    msgs = [message("/avatar/parameters/v2/Param%03d" % i, i / 100) for i in range(200)]
    out = bundles(msgs)
    assert all(len(d) <= 4096 for d in out) and len(out) > 1
    assert all(d.startswith(BUNDLE_HEAD) for d in out[:-1])
    decoded = [a for d in out for a in parse(d)]
    assert [a for a, _ in decoded] == ["/avatar/parameters/v2/Param%03d" % i for i in range(200)]
