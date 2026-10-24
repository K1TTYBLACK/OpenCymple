"""Guided recording of a labelled test corpus (raw camera datagrams + segment labels).

While you wear the headset, the recorder walks through a fixed script of gaze directions and
facial expressions. Each prompt appears in the terminal (readable in VR through WayVR) and is
announced by beeps: one high beep = do it now, one low beep = relax, three beeps = finished.
The capture is CYMPCAP1 (replayable with `run --replay`); labels go to <name>.labels.json with
the same wall-clock nanosecond timestamps as the capture records.
"""
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave

# (label, prompt, seconds)
SCRIPT = [
    ("neutral", "Relax your face, look straight ahead", 5),
    ("gaze_center", "Look straight ahead", 3),
    ("gaze_left", "Look LEFT (eyes only)", 3),
    ("gaze_right", "Look RIGHT", 3),
    ("gaze_up", "Look UP", 3),
    ("gaze_down", "Look DOWN", 3),
    ("gaze_up_left", "Look UP-LEFT", 2.5),
    ("gaze_up_right", "Look UP-RIGHT", 2.5),
    ("gaze_down_left", "Look DOWN-LEFT", 2.5),
    ("gaze_down_right", "Look DOWN-RIGHT", 2.5),
    ("gaze_sweep", "Slowly circle your eyes around", 6),
    ("blink", "Blink normally several times", 5),
    ("eyes_closed", "Close both eyes", 3),
    ("wink_left", "Close only your LEFT eye", 3),
    ("wink_right", "Close only your RIGHT eye", 3),
    ("squint", "Squint", 3),
    ("widen", "Open your eyes WIDE", 3),
    ("brow_down", "Frown with your brows (brows down)", 3),
    ("brow_up", "Raise your eyebrows", 3),
    ("jaw_open", "Open your mouth wide", 3),
    ("mouth_closed_jaw_open", "Drop your jaw, keep lips closed", 3),
    ("smile", "Smile", 3),
    ("frown", "Sad mouth (corners down)", 3),
    ("pucker", "Pucker (kiss)", 3),
    ("funnel", "Funnel lips (say 'oo')", 3),
    ("cheek_puff", "Puff your cheeks", 3),
    ("cheek_suck", "Suck your cheeks in", 3),
    ("jaw_left", "Move your jaw LEFT", 3),
    ("jaw_right", "Move your jaw RIGHT", 3),
    ("jaw_forward", "Push your jaw FORWARD", 3),
    ("mouth_left", "Move your lips LEFT", 3),
    ("mouth_right", "Move your lips RIGHT", 3),
    ("lips_in", "Roll both lips in", 3),
    ("upper_lip_raise", "Raise your upper lip (sneer)", 3),
    ("tongue_out", "Stick your tongue OUT", 3),
    ("tongue_left", "Tongue out, to the LEFT", 3),
    ("tongue_right", "Tongue out, to the RIGHT", 3),
    ("tongue_up", "Tongue out, UP", 3),
    ("tongue_down", "Tongue out, DOWN", 3),
    ("talk", "Talk or count out loud", 6),
    ("neutral_end", "Relax, look straight ahead", 4),
]
RELAX_S = 1.5


def _tone_file(path, freqs, ms=140, gap_ms=90, rate=22050):
    frames = bytearray()
    for i, f in enumerate(freqs):
        n = int(rate * ms / 1000)
        for k in range(n):
            env = min(1.0, k / 200, (n - k) / 200)
            frames += struct.pack("<h", int(12000 * env * math.sin(2 * math.pi * f * k / rate)))
        if i < len(freqs) - 1:
            frames += b"\0\0" * int(rate * gap_ms / 1000)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


class Beeper:
    """Short cue tones through PipeWire/PulseAudio; silent if no player exists or sound is off."""

    def __init__(self, enabled=True):
        self.player = (shutil.which("pw-play") or shutil.which("paplay")) if enabled else None
        self.dir = tempfile.mkdtemp(prefix="opencymple-beeps-")
        self.files = {}
        for name, freqs in (("go", [880]), ("relax", [440]), ("done", [660, 660, 880])):
            self.files[name] = os.path.join(self.dir, name + ".wav")
            _tone_file(self.files[name], freqs)

    def play(self, name):
        if self.player:
            subprocess.Popen([self.player, self.files[name]], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def _banner(text, sub=""):
    width = max(40, len(text) + 8)
    bar = "=" * width
    sys.stdout.write("\n%s\n   %s\n%s%s\n" % (bar, text, bar, ("\n   " + sub) if sub else ""))
    sys.stdout.flush()


def wait_for_streams(ingest, streams, min_fps, timeout, stop):
    """Block until every stream in `streams` reports >= min_fps; returns the missing ones."""
    deadline = time.monotonic() + timeout
    missing = list(streams)
    while time.monotonic() < deadline and not stop.is_set():
        snap = ingest.snapshot()["streams"]
        missing = [s for s in streams if snap.get(s, {}).get("fps", 0.0) < min_fps]
        if not missing:
            return []
        stop.wait(0.5)
    return missing


def run_script(labels_path, beeper, stop, script=SCRIPT, time_scale=1.0, capture=None, on_segment=None):
    """Walk the script, writing labels as it goes (so an interrupted run keeps what it has)."""
    labels = {"capture": capture, "time_scale": time_scale, "segments": []}

    def save():
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=1)

    total = sum(d for _, _, d in script) + RELAX_S * len(script)
    _banner("Recording starts in 3 seconds", "about %d s in total; follow the beeps" % (total * time_scale))
    stop.wait(3 * time_scale)
    for i, (name, prompt, dur) in enumerate(script):
        if stop.is_set():
            break
        _banner("[%d/%d] %s" % (i + 1, len(script), prompt), "%.1f s" % dur)
        beeper.play("go")
        seg = {"name": name, "prompt": prompt, "t_start_ns": time.time_ns()}
        if on_segment:
            on_segment(name)
        stop.wait(dur * time_scale)
        seg["t_end_ns"] = time.time_ns()
        seg["complete"] = not stop.is_set()
        labels["segments"].append(seg)
        save()
        if stop.is_set():
            break
        beeper.play("relax")
        print("   relax...", flush=True)
        stop.wait(RELAX_S * time_scale)
    beeper.play("done")
    _banner("Done" if not stop.is_set() else "Stopped early")
    save()
    return labels
