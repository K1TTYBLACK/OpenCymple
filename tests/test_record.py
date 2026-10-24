import json
import threading

from opencymple_linux.record import SCRIPT, Beeper, run_script


def test_script_labels_unique():
    names = [n for n, _, _ in SCRIPT]
    assert len(names) == len(set(names))


def test_run_script_writes_ordered_labels(tmp_path):
    path = tmp_path / "x.labels.json"
    beeper = Beeper(enabled=False)
    try:
        run_script(str(path), beeper, threading.Event(), script=SCRIPT[:3], time_scale=0.01, capture="x.cap")
    finally:
        beeper.close()
    labels = json.loads(path.read_text())
    segs = labels["segments"]
    assert labels["capture"] == "x.cap" and [s["name"] for s in segs] == [n for n, _, _ in SCRIPT[:3]]
    assert all(s["complete"] and s["t_end_ns"] > s["t_start_ns"] for s in segs)
    assert all(a["t_end_ns"] <= b["t_start_ns"] for a, b in zip(segs, segs[1:]))


def test_stop_keeps_partial_labels(tmp_path):
    path = tmp_path / "y.labels.json"
    stop = threading.Event()
    beeper = Beeper(enabled=False)
    try:
        run_script(str(path), beeper, stop, script=SCRIPT[:3], time_scale=0.01,
                   on_segment=lambda name: stop.set())
    finally:
        beeper.close()
    segs = json.loads(path.read_text())["segments"]
    assert len(segs) == 1 and not segs[0]["complete"]
