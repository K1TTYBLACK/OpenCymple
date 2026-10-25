"""AFK detection (afk.py) on synthetic eye observations, and the pipeline's reset while away."""
from types import SimpleNamespace

import numpy as np

from opencymple_linux import afk as A
from opencymple_linux.calibration import Calibration
from opencymple_linux.params import FIELDS
from opencymple_linux.pipeline import Pipeline

LIT, DARK = np.full((96, 96), 130, np.uint8), np.full((96, 96), 12, np.uint8)


def _obs(eye, crop, t):
    return SimpleNamespace(eye=eye, crop=crop, t=t)


def _run(det, frames, until, step=0.01):
    """frames(t) -> [(eye, crop)] for the frames arriving at t; returns [(t, away)] per tick."""
    out, t = [], 0.0
    while t < until:
        for eye, crop in frames(t):
            det.on_eye(_obs(eye, crop, t))
        out.append((t, det.update(t)))
        t = round(t + step, 6)
    return out


def _first(ticks, away):
    return next(t for t, a in ticks if a == away)


def test_enters_after_dark_and_leaves_after_lit():
    det = A.AfkDetector()
    det.enabled = True
    off, on = 2.0, 6.0
    ticks = _run(det, lambda t: [(0, DARK if off <= t < on else LIT), (1, DARK if off <= t < on else LIT)], 9.0)
    assert not any(a for t, a in ticks if t < off + A.ENTER_S - 0.02)
    entered = _first(ticks, True)
    assert abs(entered - (off + A.ENTER_S)) < 0.05
    left = next(t for t, a in ticks if t > on and not a)
    assert abs(left - (on + A.LEAVE_S)) < 0.05


def test_one_dark_eye_and_short_dark_runs_do_not_trigger():
    det = A.AfkDetector()
    det.enabled = True
    blink = lambda t: (t % 3.0) >= 2.4                         # 0.6 s of dark frames every 3 s
    ticks = _run(det, lambda t: [(0, DARK), (1, DARK if blink(t) else LIT)], 10.0)
    assert not any(a for _, a in ticks)


def test_no_frames_counts_as_away_and_disabled_never_does():
    det = A.AfkDetector()
    det.enabled = True
    assert not det.update(0.0)                                 # nothing seen yet: nothing to reset
    det.on_eye(_obs(0, LIT, 0.0))
    assert not det.update(0.5) and det.update(1.5)             # the frames stopped
    det.enabled = False
    assert not det.update(2.0) and not det.away


def test_pipeline_sends_defaults_while_away():
    p = Pipeline({}, Calibration())
    p.mouth_core.on_values([0.8] * 30)
    p.afk.enabled = True
    p.afk.on_eye(_obs(0, LIT, 0.0))
    p.afk.on_eye(_obs(0, DARK, 0.1))
    assert any(p.build_packet(0.5).values())                   # mouth values while the eye is only just dark
    packet = p.build_packet(1.5)
    assert list(packet) == list(FIELDS) and not any(packet.values())
    snap = p.snapshot()
    assert not any(snap["gaze"].values()) and not any(snap["mouth_fields"].values())
    p.afk.enabled = False
    assert any(p.build_packet(1.6).values())
