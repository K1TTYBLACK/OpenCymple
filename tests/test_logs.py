"""The log lines behind the window's Logs tab: what a view has not seen yet, the capacity limit, the text
Export Logs writes."""
import logging

from opencymple import logs


def _logger(name, buf):
    lg = logging.getLogger(name)
    lg.handlers = [buf]
    lg.propagate = False
    lg.setLevel(logging.DEBUG)
    return lg


def test_since_gives_only_the_new_lines():
    buf = logs.LogBuffer()
    lg = _logger("test_logs.since", buf)
    lg.info("first")
    lines, seen = buf.since(0)
    assert len(lines) == 1 and lines[0].endswith("test_logs.since: first")
    lg.warning("second")
    lines, seen = buf.since(seen)
    assert len(lines) == 1 and lines[0].endswith("W test_logs.since: second")
    assert buf.since(seen) == ([], 2)


def test_capacity_drops_the_oldest_lines():
    buf = logs.LogBuffer(capacity=3)
    lg = _logger("test_logs.capacity", buf)
    for i in range(5):
        lg.info("line %d", i)
    lines, seen = buf.since(0)
    assert seen == 5
    assert [x.rsplit(" ", 1)[-1] for x in lines] == ["2", "3", "4"]
    assert buf.text() == "\n".join(lines) + "\n"


def test_attach_adds_the_buffer_once():
    root = logging.getLogger()
    try:
        logs.attach()
        logs.attach()
        assert root.handlers.count(logs.BUFFER) == 1
    finally:
        root.removeHandler(logs.BUFFER)
