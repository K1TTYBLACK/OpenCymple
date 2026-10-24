"""Receiver -> per-stream inference workers -> fused tracking state.

Each camera stream gets its own worker thread with a one-frame mailbox: when inference falls
behind, the newest frame replaces the waiting one instead of queueing (lowest latency, as the
original's per-stream inference threads do). ONNX Runtime releases the GIL while running,
so the workers overlap.
"""
import logging
import threading
import time
from collections import deque

import numpy as np

from . import protocol as P
from .eye import EYE_NAMES, EyeTracker
from .face import EyeCore, MouthCore
from .imgproc import decode_gray
from .mouth import MouthV1Tracker
from .params import FIELDS

log = logging.getLogger(__name__)
PACKET_INTERVAL_S = 0.010        # faceOscClass packet timer: 10 ms, Qt::PreciseTimer


class Mailbox:
    """Latest-frame slot for one stream."""

    def __init__(self):
        self.cond = threading.Condition()
        self.frame = None
        self.dropped = 0

    def put(self, frame):
        with self.cond:
            if self.frame is not None:
                self.dropped += 1
            self.frame = frame
            self.cond.notify()

    def take(self, timeout):
        with self.cond:
            if self.frame is None:
                self.cond.wait(timeout)
            frame, self.frame = self.frame, None
            return frame


class Worker(threading.Thread):
    def __init__(self, name, process, publish):
        super().__init__(name=name, daemon=True)
        self.mailbox = Mailbox()
        self.process, self.publish = process, publish
        self.stop_event = threading.Event()
        self.done_times = deque(maxlen=100)
        self.infer_ms = deque(maxlen=100)
        self.latency_ms = deque(maxlen=100)  # frame complete -> result published
        self.decode_errors = 0
        self.errors = 0
        self.last_frame = None

    def run(self):
        while not self.stop_event.is_set():
            frame = self.mailbox.take(0.1)
            if frame is not None:
                self.handle(frame)

    def handle(self, frame):
        gray = decode_gray(frame.jpeg)
        if gray is None:
            self.decode_errors += 1
            return
        self.last_frame = gray
        try:
            obs = self.process(gray, frame)
        except Exception:
            self.errors += 1
            log.exception("%s: inference failed", self.name)
            return
        if obs is None:
            return
        done = time.monotonic()
        self.done_times.append(done)
        self.infer_ms.append(obs.infer_ms)
        self.latency_ms.append((done - frame.t) * 1000.0)
        self.publish(obs)

    def stats(self):
        t = self.done_times
        fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0
        if t and time.monotonic() - t[-1] > 1.0:
            fps = 0.0
        mean = lambda d: round(float(np.mean(d)), 2) if d else None
        return {"fps": round(fps, 1), "infer_ms": mean(self.infer_ms), "latency_ms": mean(self.latency_ms),
                "dropped": self.mailbox.dropped, "decode_errors": self.decode_errors, "errors": self.errors}


class Pipeline:
    def __init__(self, models, calib):
        self.calib = calib
        self.lock = threading.Lock()
        self.eyes = [None, None]             # latest EyeObservation per eye
        self.mouth = None                    # latest MouthObservation
        self.eye_fields = None               # latest packet fields 0..19 as {name: value}
        self.mouth_fields = None             # latest packet fields 20..69
        self.unhandled_streams = set()
        self.face = EyeCore(calib)
        self.mouth_core = MouthCore(calib)
        self.outputs = []                    # objects with update(fields, now), run on the packet timer
        self.ticker = None
        self.workers = {}
        if "eye_landmark" in models:
            for stream, eye in P.EYE_STREAMS.items():
                tracker = EyeTracker(models["eye_landmark"], eye, calib.eyes[eye].roi_quad)
                self.workers[stream] = Worker(
                    "eye-%s" % EYE_NAMES[eye],
                    lambda gray, f, tr=tracker: tr.process(gray, f.t, f.seq), self._publish_eye)
                self.workers[stream].tracker = tracker
        if "mouth_v1" in models:
            tracker = MouthV1Tracker(models["mouth_v1"], models.get("mouth_roi"), calib.mouth)
            self.workers[P.MOUTH_V1_STREAM] = Worker(
                "mouth-v1", lambda gray, f, tr=tracker: tr.process(gray, f.t, f.seq, now_ms=int(f.t * 1000)),
                self._publish_mouth)
            self.workers[P.MOUTH_V1_STREAM].tracker = tracker

    # ---- frame input (called from the receiver thread) -----------------------------------
    def _worker_for(self, frame):
        w = self.workers.get(frame.stream)
        if w is None and frame.stream not in self.unhandled_streams:
            self.unhandled_streams.add(frame.stream)
            log.warning("stream %d (%s) has no pipeline yet; ignoring it", frame.stream,
                        P.STREAM_KIND.get(frame.stream, "unknown"))
        return w

    def on_frame(self, frame):
        w = self._worker_for(frame)
        if w is not None:
            w.mailbox.put(frame)

    def process_sync(self, frame):
        """Deterministic mode: run this frame now, on the caller's thread, with its own
        timestamp driving every filter (used for replays compared against the oracle)."""
        w = self._worker_for(frame)
        if w is not None:
            w.handle(frame)

    # ---- results -------------------------------------------------------------------------
    def _publish_eye(self, obs):
        with self.lock:
            self.eyes[obs.eye] = obs
            self.face.on_landmarks(obs.eye, obs.landmarks, obs.expression, obs.t)

    def _publish_mouth(self, obs):
        with self.lock:
            self.mouth = obs
            self.mouth_core.on_values(obs.values)

    def build_packet(self, now):
        """One run of the original's 10 ms packet timer (sub_1800c2820): {field: value} with the eye
        fields when an eye was seen recently and the mouth fields once mouth data arrived, or None."""
        with self.lock:
            eye = self.face.build(now)
            mouth = self.mouth_core.build(now)
            if eye is None and mouth is None:
                return None
            packet = {}
            if eye is not None:
                self.eye_fields = dict(zip(FIELDS[:20], eye))
                packet.update(self.eye_fields)
            if mouth is not None:
                self.mouth_fields = mouth
                packet.update(mouth)
            return packet

    def _tick(self):
        next_t = time.monotonic()
        while not self.ticker_stop.is_set():
            next_t += PACKET_INTERVAL_S
            now = time.monotonic()
            fields = self.build_packet(now)
            if fields is not None:
                for out in tuple(self.outputs):      # the UI may add or remove outputs meanwhile
                    try:
                        out.update(fields, now)
                    except Exception:
                        log.exception("output %s failed", type(out).__name__)
            delay = next_t - time.monotonic()
            if delay > 0:
                self.ticker_stop.wait(delay)
            else:
                next_t = time.monotonic()

    # ---- live changes from the UI ----------------------------------------------------------
    def reconfigure(self):
        """Apply changed sliders, switches and amplitudes in self.calib (filter states are kept)."""
        with self.lock:
            self.face.configure()
            self.mouth_core.configure()

    def set_eye_roi(self, eye, quad):
        self.calib.eyes[eye].roi_quad = quad
        for stream, e in P.EYE_STREAMS.items():
            if e == eye and stream in self.workers:
                self.workers[stream].tracker.set_roi(quad)

    # ---- lifecycle -----------------------------------------------------------------------
    def start(self, ticker=True):
        for w in self.workers.values():
            w.start()
        if ticker:
            self.ticker_stop = threading.Event()
            self.ticker = threading.Thread(target=self._tick, name="packet-timer", daemon=True)
            self.ticker.start()

    def stop(self):
        if self.ticker is not None:
            self.ticker_stop.set()
            self.ticker.join(timeout=1.0)
        for w in self.workers.values():
            w.stop_event.set()
        for w in self.workers.values():
            if w.is_alive():
                w.join(timeout=1.0)

    def snapshot(self):
        with self.lock:
            return {
                "eyes": list(self.eyes),
                "mouth": self.mouth,
                "gaze": dict(self.eye_fields) if self.eye_fields else None,
                "mouth_fields": dict(self.mouth_fields) if self.mouth_fields else None,
                "workers": {w.name: w.stats() for w in self.workers.values()},
            }
