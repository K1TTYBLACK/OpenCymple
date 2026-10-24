"""`cymple-linux gui`: the pipeline of `run` plus a window in the same process.

Left: per-stream previews (raw frame with the ROI, model crop with landmarks, rates) and the
gaze plot. Right: the 70 values, Cymple's eye and mouth settings, the 28 amplitudes and the
VRChat output. Every change applies at once and is saved to our settings copy (settings.py)
after a short pause; Cymple's own files change only through Export.
"""
import logging
import os
import re
import signal
import sys
import time

from PySide6.QtCore import QByteArray, Qt, QTimer
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
                               QRadioButton, QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from .. import calibration as C
from .. import imgproc
from .. import protocol as P
from .. import settings as S
from ..params import FIELDS
from ..runtime import Runtime
from .widgets import LANDMARK_COLORS, GazeView, ImageView, SliderRow, ValueBars, dim, gray_qimage
from .xr_overlay import HeadsetLauncher

log = logging.getLogger(__name__)

# Cymple's eye page (Cympleeye QML): attribute, title, from, to, step, decimals, description
EYE_SLIDERS = (
    ("eye_expression_filter", "Eyelid smoothing", 0.0, 0.5, 0.01, 2,
     "Higher values make the eyelid open/close output smoother."),
    ("eye_gaze_filter", "Gaze smoothing", 0.2, 1.0, 0.01, 2, "Higher values make eye movement smoother."),
    ("eye_blink_close_amplitude", "Eye closing", 0.1, 0.3, 0.01, 2,
     "Higher values close the eyes more easily; raise it when the eyes do not close fully."),
    ("eye_gaze_amplitude", "Gaze range", 0.5, 1.0, 0.01, 2, "Lower values make eye movement smaller."),
    ("eye_squint_gaze_compensation", "Squint compensation", 0.0, 1.0, 0.01, 2,
     "Raise it when squinting makes the eyes look up."),
    ("eye_gaze_sync_rate", "Eye sync", 0.0, 100.0, 1.0, 0,
     "0 = no extra syncing, 100 = both eyes move together. Silly mode takes priority."),
)
# EyeOsc QML: switch slot, title, description
EYE_SWITCHES = ((7, "Flip vertical gaze", "Inverts the up/down eye output."),
                (9, "Sync eyelids", "Keeps both eyelid outputs together."),
                (12, "Silly mode", "The eyes move in opposite directions."))
# Amplitude page: slider i drives slot i (labels as Cymple's page has them), eye sliders first
AMP_LABELS = ("Nose sneer", "Cheek raise", "Cheek puff", "Cheek suck", "Jaw left/right", "Jaw forward",
              "Jaw open", "Mouth close", "Mouth left/right", "Lip suck", "Lip funnel", "Lip pucker",
              "Upper lip up", "Lower lip down", "Smile", "Dimple", "Frown", "Stretch", "Lower lip up",
              "Tongue out", "Tongue left/right", "Tongue up/down", "Tongue roll", "Eye squint", "Eye wide",
              "Brow down", "Brow inner up", "Brow pinch")
AMP_ORDER = (23, 24, 25, 26, 27) + tuple(range(23))
INACTIVE_FIELDS = ("Eye_Pupil_Left", "Eye_Pupil_Right", "TongueCurlUp", "TongueBendDown", "TongueWide")
TARGET_RE = re.compile(r"^[\w.\-]+:\d{1,5}$")
FAST_MS, SLOW_EVERY, SAVE_DELAY_MS = 33, 8, 600


def _section(title):
    label = QLabel(title)
    label.setStyleSheet("font-weight: bold; font-size: 115%; margin-top: 8px")
    return label


def _note(text):
    label = QLabel(text)
    label.setWordWrap(True)
    return dim(label)


def _scroll(widget):
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    return area


class StreamPanel(QGroupBox):
    def __init__(self, title, editable=False):
        super().__init__(title)
        self.raw, self.crop = ImageView("camera"), ImageView("model input")
        self.raw.editable = editable
        self.stats = QLabel("waiting for frames")
        self.stats.setWordWrap(True)
        self.last_raw = self.last_obs = None
        row = QHBoxLayout()
        row.addWidget(self.raw, 3)
        row.addWidget(self.crop, 2)
        box = QVBoxLayout(self)
        box.addLayout(row, 1)
        box.addWidget(self.stats)


class MainWindow(QMainWindow):
    """shell: the tray app (ui/shell.py) when the window is one of its views. It then owns the
    VRChat output and the headset calibration, and closing the window only hides OpenCymple."""

    def __init__(self, rt, args, options, shell=None):
        super().__init__()
        self.rt, self.args, self.opts, self.shell = rt, args, options, shell
        self.headset = shell.headset if shell is not None else HeadsetLauncher(rt)
        self.setWindowTitle("OpenCymple")
        self.ticks = 0
        self.sent_prev = (time.monotonic(), 0)
        self.amp_view = None                     # (avatar id, own set) the amplitude page shows
        self.save_timer = QTimer(self, singleShot=True, interval=SAVE_DELAY_MS)
        self.save_timer.timeout.connect(self._save)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_previews())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_values_tab(), "Values")
        self.tabs.addTab(_scroll(self._build_eye_tab()), "Eyes")
        self.tabs.addTab(_scroll(self._build_mouth_tab()), "Mouth")
        self.tabs.addTab(_scroll(self._build_amp_tab()), "Amplitudes")
        self.tabs.addTab(_scroll(self._build_output_tab()), "Output")
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        self.setCentralWidget(split)
        self.status = QLabel()
        self.statusBar().addWidget(self.status, 1)

        geometry = self.opts.get("window", {}).get("geometry")
        if geometry:
            self.restoreGeometry(QByteArray.fromBase64(geometry.encode()))
        else:
            self.resize(1500, 900)
        if shell is None:                        # the tray app started the output already
            self._apply_osc()
        self.timer = QTimer(self, interval=FAST_MS)
        self.timer.timeout.connect(self._refresh)
        self.timer.start()

    # ---- layout --------------------------------------------------------------------------
    def _build_previews(self):
        w = QWidget()
        grid = QGridLayout(w)
        self.eye_panels = {}
        for stream, eye in P.EYE_STREAMS.items():
            if stream in self.rt.pipeline.workers:
                self.eye_panels[stream] = StreamPanel(("Left eye", "Right eye")[eye])
                grid.addWidget(self.eye_panels[stream], 0, eye)
        self.mouth_panel = None
        if P.MOUTH_V1_STREAM in self.rt.pipeline.workers:
            self.mouth_panel = StreamPanel("Mouth", editable=True)
            self.mouth_panel.raw.rectDrawn.connect(self._mouth_rect_drawn)
            grid.addWidget(self.mouth_panel, 1, 0)
        box = QGroupBox("Gaze and eyelids")
        self.gaze = GazeView()
        QVBoxLayout(box).addWidget(self.gaze)
        grid.addWidget(box, 1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        return w

    def _build_values_tab(self):
        self.bars = ValueBars(FIELDS, signed=FIELDS[:4], inactive=INACTIVE_FIELDS)
        w = QWidget()
        box = QVBoxLayout(w)
        box.addWidget(_note("The 70 values Cymple sends (eyes blue, mouth green), after smoothing and amplitudes. "
                            "Grey ones are never filled by Cymple either."))
        box.addWidget(self.bars)
        box.addStretch(1)
        return _scroll(w)

    def _build_eye_tab(self):
        cal = self.rt.calib
        w = QWidget()
        box = QVBoxLayout(w)
        box.addWidget(_section("Eye settings"))
        self.eye_sliders = {}
        for attr, title, lo, hi, step, dec, desc in EYE_SLIDERS:
            s = SliderRow(title, lo, hi, step, dec, desc)
            s.set_value(getattr(cal, attr))
            s.valueChanged.connect(lambda v, a=attr: self._changed(self.rt.set_eye_setting, a, v))
            self.eye_sliders[attr] = s
            box.addWidget(s)
        box.addWidget(_section("Eye switches"))
        for slot, title, desc in EYE_SWITCHES:
            cb = QCheckBox(title)
            cb.setChecked(bool(cal.switches.get(slot)))
            cb.toggled.connect(lambda on, s=slot: self._changed(self.rt.set_switch, s, on))
            box.addWidget(cb)
            box.addWidget(_note(desc))
        box.addWidget(_section("Eye crop (ROI)"))
        box.addWidget(_note("Look straight ahead, then calibrate. Cymple's ROI model finds each eye in the camera "
                            "image; the crop is stored and used from then on."))
        row = QHBoxLayout()
        for label, eyes in (("Calibrate both", (0, 1)), ("Left only", (0,)), ("Right only", (1,))):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, e=eyes: self._calibrate_roi(e))
            row.addWidget(b)
        row.addStretch(1)
        box.addLayout(row)
        self.roi_status = QLabel()
        box.addWidget(self.roi_status)
        box.addWidget(_section("Gaze calibration"))
        box.addWidget(_note("Recenter: look straight ahead with relaxed eyes. It updates the eye centre, the "
                            "eyelid, squint and widen references and the expression neutral; in precise mode it "
                            "also moves the 13-point calibration to the new centre."))
        row = QHBoxLayout()
        self.precise_radio = QRadioButton("Precise (13 points)")
        self.coarse_radio = QRadioButton("Coarse (5 s look-around)")
        (self.precise_radio if cal.precise_gaze else self.coarse_radio).setChecked(True)
        self.precise_radio.toggled.connect(lambda on: self._changed(lambda _k, v: self.rt.set_precise(v), None, on))
        row.addWidget(self.precise_radio)
        row.addWidget(self.coarse_radio)
        row.addStretch(1)
        box.addLayout(row)
        box.addWidget(_note("Precise shows 13 dots to look at, one capture each (Cymple's full-view calibration). "
                            "In the headset: left stick sets the size, right stick the height, right trigger "
                            "starts; then look at each yellow dot and pull the trigger. On the desktop: full "
                            "screen, Space per dot. Coarse records how far the eyes move while you look around "
                            "for 5 seconds."))
        row = QHBoxLayout()
        recenter = QPushButton("Recenter both eyes")
        recenter.clicked.connect(lambda: self.rt.recenter(wait=False))
        self.headset_button = QPushButton("Calibrate in headset")
        self.headset_button.clicked.connect(self._headset_button)
        self.calib_button = QPushButton("Start calibration")
        self.calib_button.clicked.connect(self._calibration_button)
        row.addWidget(recenter)
        row.addWidget(self.headset_button)
        row.addWidget(self.calib_button)
        row.addStretch(1)
        box.addLayout(row)
        self.calib_status = QLabel()
        self.calib_status.setWordWrap(True)
        box.addWidget(self.calib_status)
        self.mesh_status = _note("")
        box.addWidget(self.mesh_status)
        self.mesh_window = None
        box.addStretch(1)
        self._update_eye_status()
        return w

    def _build_mouth_tab(self):
        m = self.rt.calib.mouth
        w = QWidget()
        box = QVBoxLayout(w)
        box.addWidget(_section("Mouth settings"))
        self.mouth_sliders = {}
        for attr, title, lo, hi, step, dec, desc, suffix in (
                ("filter", "Smoothing", 0.0, 0.5, 0.01, 2, "Higher values make mouth tracking smoother.", ""),
                ("rotation", "Camera rotation", -180.0, 180.0, 5.0, 0,
                 "Turns the camera image to match how the camera is mounted.", "°")):
            s = SliderRow(title, lo, hi, step, dec, desc, suffix)
            s.set_value(getattr(m, attr))
            s.valueChanged.connect(lambda v, a=attr: self._changed(self.rt.set_mouth_setting, a, v))
            self.mouth_sliders[attr] = s
            box.addWidget(s)
        box.addWidget(_section("Mouth area"))
        self.auto_roi = QCheckBox("Find the mouth automatically")
        self.auto_roi.setChecked(m.auto_roi)
        self.auto_roi.toggled.connect(self._auto_roi_toggled)
        box.addWidget(self.auto_roi)
        box.addWidget(_note("Cymple's mouth ROI model looks for the mouth once a second. Turn it off to set the "
                            "area by hand: drag a rectangle on the mouth camera image, or use the sliders."))
        self.area_sliders = {}
        for attr, title, lo, hi in (("upper", "Top", 0.0, 98.0), ("bottom", "Bottom", 1.0, 99.0),
                                    ("width", "Width", 1.0, 100.0), ("x_offset", "Horizontal centre", 0.0, 100.0)):
            s = SliderRow(title, lo, hi, 0.5, 1, suffix=" %", compact=True)
            s.valueChanged.connect(lambda v, a=attr: self._area_changed(a, v))
            self.area_sliders[attr] = s
            box.addWidget(s)
        self._sync_area_sliders()
        box.addStretch(1)
        return w

    def _build_amp_tab(self):
        w = QWidget()
        box = QVBoxLayout(w)
        box.addWidget(_note("Higher values make the expression stronger (0 to 5)."))
        row = QHBoxLayout()
        self.amp_scope = QLabel()
        self.amp_scope.setWordWrap(True)
        row.addWidget(self.amp_scope, 1)
        self.amp_button = QPushButton()
        self.amp_button.clicked.connect(self._amp_button)
        row.addWidget(self.amp_button)
        box.addLayout(row)
        grid = QGridLayout()
        self.amp_sliders = {}
        for k, slot in enumerate(AMP_ORDER):
            s = SliderRow(AMP_LABELS[slot], 0.0, 5.0, 0.05, 2, compact=True)
            s.valueChanged.connect(lambda v, i=slot: self._changed(self.rt.set_amp, i, v))
            self.amp_sliders[slot] = s
            grid.addWidget(s, k // 2, k % 2)
        box.addLayout(grid)
        box.addStretch(1)
        self._update_amp_view(force=True)
        return w

    def _build_output_tab(self):
        o = self.opts["osc"]
        w = QWidget()
        box = QVBoxLayout(w)
        g = QGroupBox("VRChat (OSC)")
        form = QFormLayout(g)
        self.osc_enabled = QCheckBox("Send avatar parameters to VRChat")
        self.osc_enabled.setChecked(bool(o["enabled"]) and not self.args.no_osc)
        self.osc_enabled.toggled.connect(self._osc_option_changed)
        form.addRow(self.osc_enabled)
        self.osc_target = QLineEdit(o["target"])
        self.osc_target.editingFinished.connect(self._osc_option_changed)
        form.addRow("Target", self.osc_target)
        self.osc_avatar = QComboBox()
        self.osc_avatar.addItem("Ask VRChat (OSCQuery, else its avatar config files)", "auto")
        self.osc_avatar.addItem("None: default addresses and native eye tracking", "none")
        self.osc_avatar.setCurrentIndex(max(0, self.osc_avatar.findData(o["avatar"])))
        self.osc_avatar.currentIndexChanged.connect(self._osc_option_changed)
        form.addRow("Avatar parameters", self.osc_avatar)
        self.osc_status = QLabel("-")
        self.osc_status.setWordWrap(True)
        form.addRow("Status", self.osc_status)
        box.addWidget(g)

        g = QGroupBox("Cameras")
        v = QVBoxLayout(g)
        self.source_label = QLabel()
        self.source_label.setWordWrap(True)
        self.source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.source_label)
        box.addWidget(g)

        g = QGroupBox("Settings")
        v = QVBoxLayout(g)
        readonly = self.rt.writer.readonly
        v.addWidget(_note("Changes are saved to %s%s" % (
            self.rt.ini, " (no: this is Cymple's own INI, which only Export changes)" if readonly else
            ". Windows Cymple keeps its own settings until you export.")))
        b = QPushButton("Export to Windows Cymple…")
        b.setEnabled(not readonly)
        b.clicked.connect(self._export)
        row = QHBoxLayout()
        row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        box.addWidget(g)
        box.addStretch(1)
        return w

    # ---- changes -------------------------------------------------------------------------
    def _changed(self, setter, key, value):
        setter(key, value)
        self.save_timer.start()

    def _save(self):
        self.rt.flush()

    def _calibrate_roi(self, eyes):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = self.rt.calibrate_eye_roi(eyes)
        except Exception as e:                   # model load / inference
            log.exception("ROI calibration failed")
            QMessageBox.warning(self, "Eye crop", "ROI calibration failed: %s" % e)
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.save_timer.start()
        names = ("left", "right")
        ok = [names[e] for e, q in result.items() if q is not None]
        bad = [names[e] for e, q in result.items() if q is None]
        msg = []
        if ok:
            msg.append("updated: %s" % ", ".join(ok))
        if bad:
            msg.append("no valid crop for %s (no frames, or the model found no eye)" % ", ".join(bad))
        self._update_eye_status("; ".join(msg))

    def _calibration_button(self):
        if self.rt.calibrator.coarse is not None:
            self.rt.stop_coarse()
        elif self.rt.calib.precise_gaze:
            if self.mesh_window is None:
                from .calibration import MeshTargetWindow
                self.mesh_window = MeshTargetWindow(self.rt)
                self.mesh_window.finished.connect(self._mesh_window_closed)
                self.mesh_window.show()
                self.mesh_window.raise_()
                self.mesh_window.activateWindow()
        else:
            self.rt.start_coarse()

    def _headset_button(self):
        if self.headset.running:
            self.headset.stop()
            return
        why = self.headset.start()
        if why:
            self.rt.calib_status = why

    def _mesh_window_closed(self):
        self.mesh_window = None
        self._update_eye_status()

    def _update_eye_status(self, last=None):
        if last is not None:
            self.roi_last = last
        last = getattr(self, "roi_last", None)
        cal = self.rt.calib
        roi = ", ".join("%s %s" % (n, "stored" if e.roi_quad is not None else "none (whole frame)")
                        for n, e in zip(("left", "right"), cal.eyes))
        self.roi_status.setText("ROI: %s%s" % (roi, "\nLast calibration: %s" % last if last else ""))
        mesh = ", ".join("%s %s" % (n, "valid" if e.mesh_valid else "missing") for n, e in zip(("left", "right"), cal.eyes))
        self.mesh_status.setText("13-point gaze mesh: %s. Coarse ranges (yaw): %s." % (mesh, ", ".join(
            "%s %+.3f / %+.3f / %+.3f" % ((n,) + tuple(e.range_yaw)) for n, e in zip(("left", "right"), cal.eyes))))

    def _auto_roi_toggled(self, on):
        self._changed(self.rt.set_mouth_setting, "auto_roi", on)
        for s in self.area_sliders.values():
            s.setEnabled(not on)

    def _sync_area_sliders(self):
        m = self.rt.calib.mouth
        for attr, s in self.area_sliders.items():
            s.set_value(getattr(m, attr) * 100.0)
            s.setEnabled(not m.auto_roi)

    def _set_area(self, upper, bottom, width, x_offset):
        for attr, v in (("upper", upper), ("bottom", bottom), ("width", width), ("x_offset", x_offset)):
            self.rt.set_mouth_setting(attr, v / 100.0)
        self._sync_area_sliders()
        self.save_timer.start()

    def _area_changed(self, attr, v):
        """The rules of Cymple's mouth page: bottom stays below top, the area inside the image."""
        m = self.rt.calib.mouth
        a = {k: getattr(m, k) * 100.0 for k in ("upper", "bottom", "width", "x_offset")}
        a[attr] = v
        if attr == "upper" and a["bottom"] <= a["upper"]:
            a["bottom"] = min(99.0, a["upper"] + 1.0)
        elif attr == "bottom" and a["bottom"] <= a["upper"]:
            a["upper"] = max(0.0, a["bottom"] - 1.0)
        if a["x_offset"] - 0.5 * a["width"] < 0:
            a["width"] = 2.0 * a["x_offset"]
        elif a["x_offset"] + 0.5 * a["width"] > 100:
            a["width"] = 2.0 * (100.0 - a["x_offset"])
        self._set_area(a["upper"], a["bottom"], a["width"], a["x_offset"])

    def _mouth_rect_drawn(self, x0, y0, x1, y1):
        if self.auto_roi.isChecked():
            self.auto_roi.setChecked(False)
        self._set_area(y0 * 100.0, min(99.0, y1 * 100.0), (x1 - x0) * 100.0, (x0 + x1) * 50.0)

    def _update_amp_view(self, force=False):
        view = (self.rt.avatar_id, self.rt.amps_own())
        if view == self.amp_view and not force:
            return
        self.amp_view = view
        for slot, v in enumerate(self.rt.amp_values()):
            self.amp_sliders[slot].set_value(v)
        avatar, own = view
        name = self._avatar_name(avatar)
        readonly = self.rt.writer.readonly
        if own:
            self.amp_scope.setText("Editing the own set of %s. Deleting it goes back to the default set." % name)
            self.amp_button.setText("Delete this avatar's set")
        else:
            self.amp_scope.setText("Editing the default set, used by every avatar without its own set%s."
                                   % (" (current avatar: %s)" % name if avatar else ""))
            self.amp_button.setText("Give %s its own set" % ("this avatar" if avatar else "an avatar"))
        self.amp_button.setEnabled(bool(avatar) and not readonly)

    def _amp_button(self):
        if self.rt.amps_own():
            if QMessageBox.question(self, "Amplitudes", "Delete the own amplitude set of %s?"
                                    % self._avatar_name(self.rt.avatar_id)) != QMessageBox.StandardButton.Yes:
                return
            self.rt.delete_own_amps()
        else:
            self.rt.create_own_amps()
        self._update_amp_view(force=True)

    def _avatar_name(self, avatar_id):
        w = self.rt.watcher
        if w is not None and w.current is not None and w.current.id == avatar_id and w.current.name:
            return w.current.name
        return avatar_id or "-"

    def _osc_option_changed(self, *_):
        target = self.osc_target.text().strip()
        if not TARGET_RE.match(target):
            self.osc_target.setStyleSheet("color: #e04040")
            return
        self.osc_target.setStyleSheet("")
        o = self.opts["osc"]
        new = {"enabled": self.osc_enabled.isChecked(), "target": target, "avatar": self.osc_avatar.currentData()}
        if new == o and (self.rt.vrchat is not None) == new["enabled"]:
            return
        o.update(new)
        S.save_options(self.opts)
        self._apply_osc()

    def _apply_osc(self):
        o = self.opts["osc"]
        if self.osc_enabled.isChecked():
            try:
                self.rt.start_osc(o["target"], o["avatar"], self.args.osc_listen)
            except (OSError, ValueError) as e:
                QMessageBox.warning(self, "VRChat output", "Cannot start the OSC output: %s" % e)
        else:
            self.rt.stop_osc()

    def _export(self):
        from ..oracle import cymple_pids
        target = C.find_ini()
        if not target:
            QMessageBox.information(self, "Export", "No Windows Cymple installation found.")
            return
        if cymple_pids():
            QMessageBox.warning(self, "Export", "Windows Cymple is running. Close it first: it writes its INI "
                                                "again when it exits.")
            return
        self.rt.flush()
        keys, files = S.export_plan(self.rt.ini, target)
        if not keys and not files:
            QMessageBox.information(self, "Export", "Windows Cymple already has these settings.")
            return
        lines = ["[%s] %s = %s" % (sec, k, v) for sec, vals in keys.items() for k, v in vals.items()]
        lines += ["%s/%s" % (S.AVATAR_DIR, n) for n in files]
        shown = "\n".join(lines[:25]) + ("\n… and %d more" % (len(lines) - 25) if len(lines) > 25 else "")
        if QMessageBox.question(self, "Export", "Write these settings into %s?\nA backup is kept.\n\n%s"
                                % (target, shown)) != QMessageBox.StandardButton.Yes:
            return
        _, _, backup = S.export_to_cymple(self.rt.ini, target)
        QMessageBox.information(self, "Export", "Done.%s" % (" Backup: %s" % backup if backup else ""))

    # ---- refresh -------------------------------------------------------------------------
    def _refresh(self):
        if self.isMinimized():
            return
        self.ticks += 1
        snap = self.rt.pipeline.snapshot()
        for stream, panel in self.eye_panels.items():
            self._update_eye_panel(panel, self.rt.pipeline.workers[stream], snap["eyes"][P.EYE_STREAMS[stream]])
        if self.mouth_panel is not None:
            self._update_mouth_panel(self.mouth_panel, self.rt.pipeline.workers[P.MOUTH_V1_STREAM], snap["mouth"])
        self.gaze.set_state(snap["gaze"])
        if self.tabs.currentIndex() == 0:
            fields = dict(snap["gaze"] or {})
            fields.update(snap["mouth_fields"] or {})
            self.bars.set_values(fields)
        if self.ticks % SLOW_EVERY == 0:
            self._refresh_slow(snap)

    def _update_eye_panel(self, panel, worker, obs):
        frame = worker.last_frame
        if frame is not panel.last_raw:
            panel.last_raw = frame
            panel.raw.set_image(gray_qimage(frame) if frame is not None else None)
        if panel.raw.quad is not worker.tracker.roi_quad:
            panel.raw.quad = worker.tracker.roi_quad
            panel.raw.update()
        if obs is not None and obs is not panel.last_obs:
            panel.last_obs = obs
            panel.crop.points = [(float(x), float(y), c) for (x, y), c in zip(obs.landmarks, LANDMARK_COLORS)]
            panel.crop.set_image(gray_qimage(obs.crop))

    def _update_mouth_panel(self, panel, worker, obs):
        m = self.rt.calib.mouth
        frame = worker.last_frame
        key = (id(frame), m.rotation)
        if key != panel.last_raw:
            panel.last_raw = key
            panel.raw.set_image(None if frame is None else gray_qimage(imgproc.rotate_frame(frame, m.rotation)))
        if m.auto_roi and obs is not None and obs.quad is not None:
            panel.raw.quad, panel.raw.rect = obs.quad, None
        else:
            panel.raw.quad = None
            panel.raw.rect = (m.x_offset - m.width / 2, m.upper, m.x_offset + m.width / 2, m.bottom)
        panel.raw.update()
        if obs is not None and obs is not panel.last_obs:
            panel.last_obs = obs
            panel.crop.set_image(gray_qimage(obs.crop))

    def _refresh_slow(self, snap):
        rx = self.rt.ingest.snapshot()
        panels = dict(self.eye_panels)
        if self.mouth_panel is not None:
            panels[P.MOUTH_V1_STREAM] = self.mouth_panel
        for stream, panel in panels.items():
            st = rx["streams"].get(stream)
            ws = self.rt.pipeline.workers[stream].stats()
            if not st:
                panel.stats.setText("waiting for frames")
                continue
            ip = st["source"] or "?"
            kind = "MOUTH" if stream == P.MOUTH_V1_STREAM else "EYE"     # the relay sends all from one IP
            devices = ", ".join("%s fw %s" % (k.split("/", 1)[1], fw) for k, fw in rx["devices"].items()
                                if k.split("/", 1)[0] == ip and k.split("/", 1)[1].startswith(kind))
            fmt = lambda v, f: f % v if v is not None else "-"
            panel.stats.setText("%s %s · received %.1f fps · tracked %.1f fps · inference %s ms · latency %s ms · "
                                "dropped %d%s" % (ip, devices, st["fps"], ws["fps"], fmt(ws["infer_ms"], "%.2f"),
                                                  fmt(ws["latency_ms"], "%.1f"), ws["dropped"],
                                                  " · errors %d" % ws["errors"] if ws["errors"] else ""))
        lines = [self.rt.source_desc or "-"]
        for ip, hs in rx["headsets"].items():
            lines.append("headset %s: %s" % (ip, hs))
        for k, fw in rx["devices"].items():
            lines.append("device %s, firmware %s" % (k.replace("/", ": "), fw))
        for s, st in rx["streams"].items():
            lines.append("stream %d (%s): %d frames, %d missing, %d incomplete" % (
                s, P.STREAM_KIND.get(s, "?"), st["frames"], st["missing"], st["incomplete"]))
        if rx["errors"]:
            lines.append("errors: %s" % rx["errors"])
        self.source_label.setText("\n".join(lines))

        v = self.rt.vrchat
        if v is None:
            self.osc_status.setText("off")
            osc = "VRChat output off"
        else:
            st = v.status()
            now = time.monotonic()
            t0, n0 = self.sent_prev
            rate = (st["sent"] - n0) / (now - t0) if now > t0 else 0.0
            self.sent_prev = (now, st["sent"])
            self.osc_status.setText("avatar %s (%s), %d parameters, %.0f messages/s" % (
                st["avatar"] or "unknown: default addresses", st["source"], st["parameters"], rate))
            osc = "VRChat: %s, %d parameters" % (st["avatar"] or "default addresses", st["parameters"])
        self._update_amp_view()
        headset_on = self.headset.running
        status = self.rt.calib_status
        hs = self.headset.status()
        if hs:
            status = "%s%s" % (hs, ". " + status if status and headset_on else "")
        self.calib_status.setText(status)
        if self.precise_radio.isChecked() != self.rt.calib.precise_gaze:     # changed from the tray / CLI
            for b in (self.precise_radio, self.coarse_radio):
                b.blockSignals(True)
            (self.precise_radio if self.rt.calib.precise_gaze else self.coarse_radio).setChecked(True)
            for b in (self.precise_radio, self.coarse_radio):
                b.blockSignals(False)
        left = self.rt.coarse_remaining()
        if left is not None:
            self.calib_button.setText("Stop coarse calibration (%.0f s)" % left)
        else:
            self.calib_button.setText("Calibration window open" if self.mesh_window is not None else
                                      "Calibrate on desktop" if self.rt.calib.precise_gaze else
                                      "Start coarse calibration (5 s)")
        self.calib_button.setEnabled(self.mesh_window is None and not headset_on)
        self.headset_button.setText("Stop headset calibration" if headset_on else "Calibrate in headset")
        self.headset_button.setEnabled(headset_on or (self.mesh_window is None and left is None))
        self._update_eye_status()
        saved = "not saved (Cymple's INI)" if self.rt.writer.readonly else (
            "saving…" if self.save_timer.isActive() else "saved")
        self.status.setText("%s · %s · settings %s" % (self.rt.source_desc or "-", osc, saved))

    # ---- shutdown ------------------------------------------------------------------------
    def closeEvent(self, event):
        self.timer.stop()
        self.save_timer.stop()
        self.opts.setdefault("window", {})["geometry"] = bytes(self.saveGeometry().toBase64()).decode()
        S.save_options(self.opts)
        if self.shell is not None:               # OpenCymple keeps running in the tray
            self.rt.flush()
            self.shell.window_closed(self)
            super().closeEvent(event)
            return
        self.headset.stop(wait=5.0)
        self.rt.stop()
        super().closeEvent(event)


def _apply_theme(app):
    """Fusion; dark like Cymple unless the desktop asks for a light scheme."""
    app.setStyle("Fusion")
    if QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Light:
        return
    p = QPalette()
    roles = {QPalette.ColorRole.Window: "#202126", QPalette.ColorRole.WindowText: "#e6e6e6",
             QPalette.ColorRole.Base: "#2e3037", QPalette.ColorRole.AlternateBase: "#353740",
             QPalette.ColorRole.Text: "#e6e6e6", QPalette.ColorRole.Button: "#363840",
             QPalette.ColorRole.Light: "#55585f", QPalette.ColorRole.Midlight: "#44474f",
             QPalette.ColorRole.Mid: "#2a2c32", QPalette.ColorRole.Dark: "#141518",
             QPalette.ColorRole.Shadow: "#0a0a0c",
             QPalette.ColorRole.ButtonText: "#e6e6e6", QPalette.ColorRole.Highlight: "#3491fa",
             QPalette.ColorRole.HighlightedText: "#ffffff", QPalette.ColorRole.ToolTipBase: "#2b2c33",
             QPalette.ColorRole.ToolTipText: "#e6e6e6", QPalette.ColorRole.PlaceholderText: "#9aa0aa",
             QPalette.ColorRole.Link: "#4aa3ff"}
    for role, color in roles.items():
        p.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor("#70747c"))
    app.setPalette(p)


def _screenshots(win, path):
    """--screenshot: one PNG per tab (<stem>-<tab>.png), for checks without a display."""
    stem = path[:-4] if path.lower().endswith(".png") else path
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
        win._refresh()
        win._refresh_slow(win.rt.pipeline.snapshot())
        QApplication.processEvents()
        win.grab().save("%s-%s.png" % (stem, win.tabs.tabText(i).lower()))


def run_gui(args, ini, model_dir, source_kwargs):
    # Qt's Wayland text-input logs a warning on every focus change; keep the terminal readable
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.wayland.textinput=false")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("cymple-linux")
    from PySide6.QtGui import QIcon
    from ..desktop import icon_path
    app.setWindowIcon(QIcon(icon_path()))
    _apply_theme(app)
    rt = Runtime(ini, model_dir, args.provider, args.threads, mouth=not args.no_mouth)
    rt.start()
    rt.start_source(**source_kwargs)
    win = MainWindow(rt, args, S.load_options())
    win.show()
    signal.signal(signal.SIGINT, lambda *_: win.close())
    signal.signal(signal.SIGTERM, lambda *_: win.close())
    wake = QTimer(interval=200)                  # lets Python run its signal handlers
    wake.timeout.connect(lambda: None)
    wake.start()
    if getattr(args, "calibrate", False):
        QTimer.singleShot(1500, win._headset_button)     # first frames and landmarks before the overlay
    if args.duration:
        def finish():
            if args.screenshot:
                _screenshots(win, args.screenshot)
            win.close()
        QTimer.singleShot(int(args.duration * 1000), finish)
    try:
        return app.exec()
    finally:
        rt.stop()
