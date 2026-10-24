"""What `run` and `gui` share: settings, models, pipeline, camera source and the VRChat output,
plus the live changes the window makes (sliders, switches, amplitudes, eye ROI, calibration)."""
import logging
import os
import threading
import time

from . import calibration as C
from . import protocol as P
from . import settings as S

log = logging.getLogger(__name__)


class AmpFollower:
    """Pipeline output that switches the amplitude set when VRChat changes avatar: that avatar's
    own avatarParaments/<id>.json, else default.json. [Inferred] from Cymple's amplitude page
    ("changes only affect the current model; deleting goes back to the default parameters")."""

    def __init__(self, runtime):
        self.rt = runtime
        self._version = None

    def update(self, fields, now):
        w = self.rt.watcher
        if w is None or w.version == self._version:
            return
        self._version = w.version
        if w.current is not None and w.current.id != self.rt.avatar_id:
            self.rt.select_avatar(w.current.id)


class CalibrationTicker:
    """Pipeline output that ends the coarse range calibration after its 5 s (Cymple's QTimer)."""

    def __init__(self, runtime):
        self.rt = runtime

    def update(self, fields, now):
        co = self.rt.calibrator.coarse
        if co is not None and now >= co["until"]:
            self.rt.stop_coarse(auto=True)


class Runtime:
    ROI_SETTLE_S = 0.6       # our choice: let the pt7 low-pass settle on the new crop before recentering

    def __init__(self, ini, model_dir, provider="auto", threads=1, mouth=True):
        from .models import load_models, warmup
        from .pipeline import Pipeline
        from .receiver import Ingest

        self.ini, self.model_dir, self.provider, self.threads = ini, model_dir, provider, threads
        self.avatar_dir = os.path.dirname(os.path.abspath(ini))
        self.writer = S.IniWriter(ini)
        self.calib = C.load(ini)
        self.avatar_id = C.read_ini(ini).get("VRCDIRECT", {}).get("SelectedAvatarId") or None
        self.amps_dirty = False
        self.amp_lock = threading.RLock()                  # the packet timer switches sets on avatar change
        names = ["eye_landmark"] + (["mouth_roi", "mouth_v1"] if mouth else [])
        self.models = load_models(model_dir, names, provider, threads)
        for m in self.models.values():
            warmup(m)
        for eye, e in enumerate(self.calib.eyes):
            log.info("%s eye: roi=%s mesh=%s feature_mode=%d", ("left", "right")[eye],
                     "stored" if e.roi_quad is not None else "none (whole frame)",
                     "valid" if e.mesh_valid else "missing", e.feature_mode)
        self.pipeline = Pipeline(self.models, self.calib)
        self.ingest = Ingest(self.pipeline.on_frame)
        self.src = self.watcher = self.vrchat = self.sender = None
        self.source_desc = None
        self.roi_model = None
        from .calib import EyeCalibrator
        self.calibrator = EyeCalibrator(self.pipeline.face, self.writer)
        self.pipeline.face.calibrator = self.calibrator
        self.calib_status = ""                             # last calibration message, for the window
        self.calib_busy = False
        self.pipeline.outputs.append(AmpFollower(self))
        self.pipeline.outputs.append(CalibrationTicker(self))

    # ---- lifecycle -----------------------------------------------------------------------
    def start_source(self, replay=None, speed=1.0, loop=False, bind="0.0.0.0", port=P.INGEST_PORT,
                     beacon=True, beacon_to=(), record=None):
        from .receiver import ReplaySource, UdpSource
        if replay:
            self.src = ReplaySource(self.ingest, replay, speed=speed, loop=loop)
            self.source_desc = "replay %s" % os.path.basename(replay)
        else:
            self.src = UdpSource(self.ingest, bind=bind, port=port, beacon=beacon, beacon_to=beacon_to,
                                 record_path=record)
            self.source_desc = "UDP %s:%d%s" % (bind, port, "" if beacon else ", no beacon")
        self.src.start()
        return self.src

    def start(self):
        self.pipeline.start()

    def stop(self):
        if self.src is not None:
            self.src.stop()
        self.pipeline.stop()
        self.stop_osc()
        self.flush()

    # ---- VRChat OSC (Mode A) -------------------------------------------------------------
    def start_osc(self, target="127.0.0.1:9000", avatar="auto", listen_port=9001):
        from .outputs.avatar import AvatarWatcher, FixedAvatar, read_avatar_config
        from .outputs.osc import OscSender
        from .outputs.vrchat import VRChatOutput

        self.stop_osc()
        host, _, port = target.rpartition(":")
        self.sender = OscSender(host or "127.0.0.1", int(port))
        if avatar == "auto":
            self.watcher = AvatarWatcher(oscquery=True, listen_port=listen_port)
            self.watcher.start()
            avatars = self.watcher
        elif avatar == "none":
            avatars = None
        else:
            avatars = FixedAvatar(read_avatar_config(avatar))
        self.vrchat = VRChatOutput(self.sender, avatars)
        self.pipeline.outputs.append(self.vrchat)
        log.info("VRChat OSC output to %s:%s (avatar parameters: %s)", host or "127.0.0.1", port, avatar)
        return self.vrchat

    def stop_osc(self):
        if self.vrchat is not None and self.vrchat in self.pipeline.outputs:
            self.pipeline.outputs.remove(self.vrchat)
        if self.watcher is not None:
            self.watcher.stop()
        if self.sender is not None:
            self.sender.close()
        self.vrchat = self.watcher = self.sender = None

    # ---- live settings (all of them saved to our INI by flush()) --------------------------
    def set_eye_setting(self, attr, value):
        """One of calibration.OSC_SETTING_KEYS (the eye sliders)."""
        setattr(self.calib, attr, float(value))
        self.pipeline.reconfigure()
        self.writer.set("OSC", C.OSC_SETTING_KEYS[attr], float(value))

    def set_switch(self, slot, on):
        self.calib.switches[slot] = bool(on)
        self.writer.set("OSC", C.SWITCHES[slot], bool(on))

    def set_mouth_setting(self, attr, value):
        """One of calibration.MOUTH_SETTING_KEYS; the tracker reads them on every frame."""
        value = bool(value) if attr == "auto_roi" else float(value)
        setattr(self.calib.mouth, attr, value)
        if attr == "filter":
            self.pipeline.reconfigure()
        self.writer.set("Cymple_Mouth", C.MOUTH_SETTING_KEYS[attr], value)

    def calibrate_eye_roi(self, eyes=(0, 1)):
        """model0 on each eye's latest frame (the user looks straight ahead), as Cymple's ROI
        calibration; a valid quad replaces the stored one at once. Returns {eye: quad or None}."""
        from .eye import calibrate_roi
        from .models import load_models
        if self.roi_model is None:
            self.roi_model = load_models(self.model_dir, ["eye_roi"], self.provider, self.threads)["eye_roi"]
        out = {}
        for stream, eye in P.EYE_STREAMS.items():
            if eye not in eyes:
                continue
            w = self.pipeline.workers.get(stream)
            frame = w.last_frame if w is not None else None
            quad = calibrate_roi(self.roi_model, frame, eye) if frame is not None else None
            out[eye] = quad
            if quad is not None:
                self.pipeline.set_eye_roi(eye, quad)
                for k, v in C.roi_values(quad).items():
                    self.writer.set(C.EYE_SECTIONS[eye], k, v)
        return out

    # ---- gaze calibration (calib.py) ----------------------------------------------------
    def _status(self, msg):
        self.calib_status = msg
        log.info("calibration: %s", msg)

    def _refresh_roi_for_calibration(self):
        """Cymple runs the eye ROI calibration before recenter and range calibration; returns the eyes
        whose crop changed."""
        try:
            changed = [e for e, q in self.calibrate_eye_roi().items() if q is not None]
        except Exception as e:                   # no ROI model: keep the stored crops
            log.warning("ROI refresh before calibration failed: %s", e)
            return []
        return changed

    def _wait_fresh(self, since, timeout=1.5):
        """Until every eye that had landmarks has new ones from frames after `since`."""
        face = self.pipeline.face
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self.pipeline.lock:
                if all(face.lm[e] is None or face.seen[e] > since for e in (0, 1)):
                    return True
            time.sleep(0.02)
        return False

    def recenter(self, wait=True):
        """Both eyes: ROI refresh, then the recenter on landmarks from the new crop. Runs in a thread
        unless wait; the outcome goes to calib_status (and the INI)."""
        if self.calib_busy:
            return None
        self.calib_busy = True
        result = {}

        def work():
            try:
                self._status("recenter: refreshing the eye crop...")
                if self._refresh_roi_for_calibration():
                    time.sleep(self.ROI_SETTLE_S)
                self._wait_fresh(time.monotonic())
                with self.pipeline.lock:
                    result.update(self.calibrator.recenter(time.monotonic()))
                self.flush()
                names = ("left", "right")
                self._status("recenter: " + "; ".join(
                    "%s %s" % (names[e], r if isinstance(r, str) else "done (%s)" % r.get("gaze", ""))
                    for e, r in sorted(result.items())))
            except Exception as e:
                log.exception("recenter failed")
                self._status("recenter failed: %s" % e)
            finally:
                self.calib_busy = False

        if wait:
            work()
            return result
        threading.Thread(target=work, name="recenter", daemon=True).start()
        return None

    def set_precise(self, on):
        """Precise (13-point mesh) or coarse (min/mid/max ranges) gaze calibration; switching stops a
        running calibration (sub_1800bff00)."""
        self.stop_coarse()
        with self.pipeline.lock:
            self.calibrator.cancel_mesh()
            self.calib.precise_gaze = bool(on)
            self.pipeline.face.reload_calibration()
        self.writer.set("OSC", "eye_gaze_use_precise_calibration", bool(on))
        self._status("gaze calibration mode: %s" % ("precise" if on else "coarse"))

    def start_coarse(self):
        if self.calib_busy or self.calibrator.coarse is not None:
            return False
        self._refresh_roi_for_calibration()
        with self.pipeline.lock:
            self.calibrator.start_coarse(time.monotonic())
        self._status("coarse calibration: look at the centre, then around in a full circle")
        return True

    def stop_coarse(self, auto=False):
        with self.pipeline.lock:
            out = self.calibrator.stop_coarse()
        if out is None:
            return None
        self.flush()
        self._status("coarse calibration %s: %s" % ("finished" if auto else "stopped", ", ".join(
            "%s %s" % (("left", "right")[e], "updated" if r["changed"] else "unchanged") for e, r in out.items())))
        return out

    def coarse_remaining(self):
        co = self.calibrator.coarse
        return None if co is None else max(0.0, co["until"] - time.monotonic())

    def start_mesh(self, overlay=None):
        """overlay: (y offset m, ring scale) of the headset overlay to save with the mesh."""
        self._refresh_roi_for_calibration()
        with self.pipeline.lock:
            m = self.calibrator.start_mesh(overlay)
        self._status("13-point calibration: point 0 of 12")
        return m

    def capture_mesh_point(self):
        with self.pipeline.lock:
            rep = self.calibrator.capture_mesh_point(time.monotonic())
        if rep is None:
            return None
        if rep.get("done"):
            self.flush()
            saved = rep["saved"]
            self._status("13-point calibration saved for %s" % (", ".join(
                ("left", "right")[e] for e, ok in saved.items() if ok) or "no eye"))
        elif "skipped" in rep:
            self._status("point %d not captured: %s" % (rep["point"], rep["skipped"]))
        else:
            self._status("point %d captured (%s)" % (rep["point"], ", ".join(("left", "right")[e] for e in rep["eyes"])))
        return rep

    def eye_readiness(self):
        """(left, right): True when a 13-point capture would use that eye now."""
        now = time.monotonic()
        with self.pipeline.lock:
            return tuple(self.calibrator.eye_ready(e, now) for e in (0, 1))

    def eyes_tracked(self):
        """(left, right): True when that eye has fresh landmarks."""
        now = time.monotonic()
        with self.pipeline.lock:
            return tuple(self.calibrator._fresh(e, now) for e in (0, 1))

    def set_overlay_geometry(self, y_offset, ring_scale):
        """The headset overlay's height and coverage (updateEyeGazeCalibrationOverlay*)."""
        self.calib.overlay_geometry = (float(y_offset), float(ring_scale))
        self.writer.set("OSC", "eye_gaze_calibration_overlay_y_offset_meters", float(y_offset))
        self.writer.set("OSC", "eye_gaze_calibration_overlay_ring_scale", float(ring_scale))
        self.flush()

    def cancel_mesh(self):
        with self.pipeline.lock:
            active = self.calibrator.mesh is not None
            self.calibrator.cancel_mesh()
        if active:
            self.flush()                         # centre / lid references from points already taken stay
            self._status("13-point calibration stopped; the previous mesh stays active")

    # ---- amplitudes ----------------------------------------------------------------------
    def amp_values(self):
        """The 28 slots as the amplitude page shows them (slot i = Cymple's slider i)."""
        return C.amp_slots(self.calib.amps)

    def amps_own(self):
        """True when the current avatar has its own amplitude set."""
        return S.has_own_amps(self.avatar_dir, self.avatar_id)

    def select_avatar(self, avatar_id):
        with self.amp_lock:
            self.flush()                                   # pending edits belong to the old set
            self.avatar_id = avatar_id
            self.calib.amps = C.load_avatar_amps(os.path.join(self.avatar_dir, S.AVATAR_DIR), avatar_id)
            self.pipeline.reconfigure()
        log.info("amplitudes: %s", "own set of %s" % avatar_id if self.amps_own() else "default set")

    def set_amp(self, slot, value):
        with self.amp_lock:
            self.calib.amps[C.AMP_KEYS[slot]] = min(max(float(value), 0.0), 5.0)
            self.pipeline.reconfigure()
            self.amps_dirty = True

    def create_own_amps(self):
        """Give the current avatar its own set, starting from the values in use."""
        with self.amp_lock:
            if self.avatar_id and not self.writer.readonly:
                self.flush()
                S.save_amps(self.avatar_dir, self.avatar_id, dict(zip(C.AMP_KEYS, self.amp_values())))

    def delete_own_amps(self):
        with self.amp_lock:
            if self.avatar_id and not self.writer.readonly:
                S.delete_amps(self.avatar_dir, self.avatar_id)
                self.amps_dirty = False
                self.select_avatar(self.avatar_id)

    def flush(self):
        """Write pending changes: INI keys, and the amplitude set in use."""
        self.writer.flush()
        with self.amp_lock:
            if self.amps_dirty and not self.writer.readonly:
                S.save_amps(self.avatar_dir, self.avatar_id if self.amps_own() else None,
                            dict(zip(C.AMP_KEYS, self.amp_values())))
            self.amps_dirty = False

    # ---- status --------------------------------------------------------------------------
    def status_line(self):
        rx = self.ingest.snapshot()
        snap = self.pipeline.snapshot()
        parts = []
        for stream, st in rx["streams"].items():
            w = self.pipeline.workers.get(stream)
            ws = w.stats() if w else None
            name = w.name if w else "s%d(ignored)" % stream
            if ws:
                parts.append("%s rx %4.1f inf %4.1f fps %s ms drop %d" % (
                    name, st["fps"], ws["fps"], "%.2f" % ws["infer_ms"] if ws["infer_ms"] else "-", ws["dropped"]))
            else:
                parts.append("%s rx %4.1f" % (name, st["fps"]))
        g = snap["gaze"]
        if g:
            parts.append("gaze yaw %+.2f/%+.2f pitch %+.2f" % (g["EyeYaw_L"], g["EyeYaw_R"], g["EyePitch_L"]))
        m = snap["mouth_fields"]
        if m:
            parts.append("jaw %.2f smile %.2f/%.2f" % (m["JawOpen"], m["MouthSmileLeft"], m["MouthSmileRight"]))
        if rx["errors"]:
            parts.append("errors %s" % rx["errors"])
        line = " | ".join(parts) if parts else "waiting for camera frames..."
        if self.vrchat is not None:
            st = self.vrchat.status()
            line += " | osc %s (%s) %d params, %d sent" % (st["avatar"] or "-", st["source"].split()[0],
                                                          st["parameters"], st["sent"])
        return line
