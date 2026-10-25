"""`opencymple-linux gui`: the pipeline of `run` plus a window in the same process.

Sidebar, upper deck: the 70 values, Cymple's eye and mouth settings, the 28 amplitudes and the VRChat
output. Lower deck: the log window and the settings copy's import and export. Beside the pages:
previews (the eyes' model inputs with landmarks, the mouth's camera frame, rates) and the gaze plot.
Every change applies at once and is saved to our settings copy (settings.py) after a short pause;
The settings copy is imported from and exported to a file of the user's choice.
"""
import logging
import os
import re
import signal
import sys
import time

from PySide6.QtCore import QByteArray, QRect, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFontDatabase, QIcon, QPainter, QPalette, QPen
from PySide6.QtWidgets import (QAbstractButton, QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPushButton, QScrollArea, QSizeGrip, QSizePolicy, QSplitter, QTabWidget,
                               QToolButton, QVBoxLayout, QWidget)

from .. import __version__
from .. import calibration as C
from .. import imgproc
from .. import logs
from .. import protocol as P
from .. import settings as S
from ..params import FIELDS
from ..outputs import avatar as avatar_out
from ..runtime import Runtime
from .. import desktop
from ..desktop import icon_path
from .icons import icon
from .widgets import LANDMARK_COLORS, GazeView, ImageView, LogView, SliderRow, ValueBars, dim, gray_qimage
from .xr_overlay import HeadsetLauncher

log = logging.getLogger(__name__)
RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources")

# Cymple's eye page (Cympleeye QML): attribute, title, from, to, step, decimals, description
EYE_SLIDERS = (
    ("eye_expression_filter", "Eyelid smoothing", 0.0, 0.5, 0.01, 2, None),
    ("eye_gaze_filter", "Gaze smoothing", 0.2, 1.0, 0.01, 2, None),
    ("eye_blink_close_amplitude", "Eye closing", 0.1, 0.3, 0.01, 2,
     "Higher values close the eyes more easily; raise it when the eyes do not close fully."),
    ("eye_gaze_amplitude", "Gaze range", 0.5, 1.0, 0.01, 2, "Lower values make eye movement smaller."),
    ("eye_squint_gaze_compensation", "Squint compensation", 0.0, 1.0, 0.01, 2,
     "Raise it when squinting makes the eyes look up."),
    ("eye_gaze_sync_rate", "Eye sync", 0.0, 100.0, 1.0, 0, None),
)
# EyeOsc QML: switch slot, title
EYE_SWITCHES = ((7, "Flip vertical gaze"), (9, "Sync eyelids"))
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
NARROW_WIDTH = 1280                              # below this the previews move into a tab before Output
NO_MAX_WIDTH = 16777215                          # Qt's "no maximum" (QWIDGETSIZE_MAX is not exported to Python)
# Sidebar buttons by label. Upper deck: the pages in tab order (Previews when narrow); lower deck: Logs, Settings
NAV_ICONS = {"Values": "activity", "Eyes": "eye", "Mouth": "smile", "Amplitudes": "sliders-horizontal",
             "Previews": "scan-eye", "Output": "send", "Logs": "scroll-text", "Settings": "settings"}


def _card(title):
    """A titled card (group box) and its layout. The stylesheet pads the card, so the layout adds no margin of
    its own: every card's title and content share one left edge."""
    card = QGroupBox(title)
    box = QVBoxLayout(card)
    box.setContentsMargins(0, 0, 0, 0)
    box.setSpacing(8)
    return card, box


def _repolish(widget):
    """Apply a changed dynamic property (such as invalid) to the stylesheet."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class ToggleRow(QCheckBox):
    """A switch: a pill track with a knob, then the label. Painted here, as the stylesheet cannot draw a knob.
    It has no surface and no padding, so it lines up with the other content of its card. Only a Tab (not a
    click) focuses it, so the focus ring is a keyboard cue."""
    ROW_H, TRACK_W, TRACK_H, KNOB, GAP = 32, 40, 22, 16, 10
    TRACK_OFF, ACCENT, KNOB_OFF, KNOB_ON = (QColor(c) for c in ("#4b647e", "#4cc2ff", "#f2f6fb", "#04121f"))

    def __init__(self, text):
        super().__init__(text)
        self.setObjectName("toggle")             # the stylesheet gives it the label's font
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def sizeHint(self):
        return QSize(self.TRACK_W + self.GAP + self.fontMetrics().horizontalAdvance(self.text()), self.ROW_H)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        on, enabled = self.isChecked(), self.isEnabled()
        track = QRectF(0, (self.height() - self.TRACK_H) / 2, self.TRACK_W, self.TRACK_H)
        p.setOpacity(1.0 if enabled else 0.4)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.ACCENT if on else self.TRACK_OFF)
        p.drawRoundedRect(track, self.TRACK_H / 2, self.TRACK_H / 2)
        inset = (self.TRACK_H - self.KNOB) / 2
        knob_x = track.right() - inset - self.KNOB if on else track.left() + inset
        p.setBrush(self.KNOB_ON if on else self.KNOB_OFF)
        p.drawEllipse(QRectF(knob_x, track.top() + inset, self.KNOB, self.KNOB))
        p.setOpacity(1.0)
        if self.hasFocus():
            p.setPen(QPen(self.ACCENT, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            ring = track.adjusted(-2, -2, 2, 2)
            p.drawRoundedRect(ring, ring.height() / 2, ring.height() / 2)
        p.setPen(self.palette().color(QPalette.ColorGroup.Normal if enabled else QPalette.ColorGroup.Disabled,
                                      QPalette.ColorRole.WindowText))
        p.setFont(self.font())
        p.drawText(QRectF(track.right() + self.GAP, 0, self.width() - track.right() - self.GAP, self.height()),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())
        p.end()


class NavButton(QToolButton):
    """A sidebar button: the icon over the label, the two centred as one group (Qt's own tool button puts the
    label at the bottom edge and the icon in what is left, so the content sits off-centre)."""
    HEIGHT, ICON, GAP = 82, 24, 6
    IDLE, HOVER, CHECKED_BG = QColor(76, 194, 255, 10), QColor(76, 194, 255, 18), QColor(76, 194, 255, 41)   # tints of the accent
    MUTED, ACCENT = QColor("#8ba3bd"), QColor("#4cc2ff")

    def __init__(self, text):
        super().__init__()
        self.setObjectName("nav")                # the stylesheet gives it the label's weight
        self.setText(text)
        self.setCheckable(True)
        self.setFixedHeight(self.HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        on = self.isChecked()
        row = QRectF(self.rect())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.CHECKED_BG if on else self.HOVER if self.underMouse() else self.IDLE)
        p.drawRoundedRect(row, 8, 8)
        if self.hasFocus():
            p.setPen(QPen(self.ACCENT, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(row.adjusted(1, 1, -1, -1), 7, 7)
        text_h = self.fontMetrics().height()
        top = (self.height() - self.ICON - self.GAP - text_h) / 2
        self.icon().paint(p, QRect(int((self.width() - self.ICON) / 2), int(top), self.ICON, self.ICON),
                          Qt.AlignmentFlag.AlignCenter, QIcon.Mode.Normal,
                          QIcon.State.On if on else QIcon.State.Off)
        p.setPen(self.ACCENT if on else self.MUTED)
        p.setFont(self.font())
        p.drawText(QRectF(0, top + self.ICON + self.GAP, self.width(), text_h), Qt.AlignmentFlag.AlignCenter, self.text())
        p.end()


def _badge(text, kind):
    """A pill label: kind "success" (version) or "accent" (status)."""
    label = QLabel(text)
    label.setObjectName("badge")
    label.setProperty("kind", kind)
    return label


class TitleBar(QWidget):
    """The title bar of the frameless window: dragging it moves the window, a double click maximises."""

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setObjectName("titlebar")
        self.setFixedHeight(40)

    def mousePressEvent(self, event):
        handle = self.win.windowHandle()
        if event.button() == Qt.MouseButton.LeftButton and handle is not None:
            handle.startSystemMove()

    def mouseDoubleClickEvent(self, event):
        self.win.toggle_maximized()


def _note(text):
    label = QLabel(text)
    label.setWordWrap(True)
    return dim(label)


def _scroll(widget):
    area = QScrollArea()
    area.setWidgetResizable(True)
    widget.layout().setContentsMargins(16, 16, 16, 16)
    widget.layout().setSpacing(12)               # the gap between a page's cards
    area.setWidget(widget)
    return area


class StreamPanel(QGroupBox):
    """One stream: the camera frame (`raw`) and the model's input crop (`crop`); `camera` / `model` hide one
    of them. A hidden view still receives its images."""

    def __init__(self, title, editable=False, camera=True, model=True):
        super().__init__(title)
        self.raw, self.crop = ImageView("camera"), ImageView("model input")
        self.raw.editable = editable
        self.raw.setVisible(camera)
        self.crop.setVisible(model)
        self.stats = dim(QLabel("waiting for frames"))
        self.stats.setWordWrap(True)
        self.last_raw = self.last_obs = None
        row = QHBoxLayout()
        row.addWidget(self.raw, 3)
        row.addWidget(self.crop, 2)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
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
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.ticks = 0
        self.sent_prev = (time.monotonic(), 0)
        self.amp_view = None                     # (avatar id, own set) the amplitude page shows
        self.save_timer = QTimer(self, singleShot=True, interval=SAVE_DELAY_MS)
        self.save_timer.timeout.connect(self._save)

        self.tabs = QTabWidget()
        self.tabs.tabBar().hide()                # the sidebar switches the pages
        self.tabs.setFocusPolicy(Qt.FocusPolicy.NoFocus)   # a Tab stop on the hidden bar would bounce focus back
        self.nav_buttons = {}                    # page widget -> its sidebar button
        self.previews_narrow = None
        self.previews_area = QScrollArea()
        self.previews_area.setWidgetResizable(True)
        self.previews_area.setWidget(self._build_previews())
        self.tabs.addTab(self._build_values_tab(), "Values")
        self.tabs.addTab(_scroll(self._build_eye_tab()), "Eyes")
        self.tabs.addTab(_scroll(self._build_mouth_tab()), "Mouth")
        self.tabs.addTab(_scroll(self._build_amp_tab()), "Amplitudes")
        self.output_page = _scroll(self._build_output_tab())
        self.tabs.addTab(self.output_page, "Output")
        self.logs_page = self._build_logs_tab()
        self.tabs.addTab(self.logs_page, "Logs")
        self.settings_page = _scroll(self._build_settings_tab())
        self.tabs.addTab(self.settings_page, "Settings")
        self.tabs.currentChanged.connect(self._sync_nav)
        page = QWidget()
        page.setObjectName("panel")
        page_box = QVBoxLayout(page)
        page_box.setContentsMargins(0, 0, 0, 0)
        page_box.addWidget(self.tabs)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.split = split
        split.setHandleWidth(8)
        split.addWidget(page)
        split.addWidget(self.previews_area)
        split.setStretchFactor(0, 1)             # the pages take what the window adds; the previews keep their width
        split.setStretchFactor(1, 0)
        body = QHBoxLayout()
        body.setSpacing(8)
        body.addWidget(self._build_sidebar())
        body.addWidget(split, 1)
        shell = QWidget()
        shell.setObjectName("shell")
        column = QVBoxLayout(shell)
        column.setContentsMargins(8, 8, 8, 8)
        column.setSpacing(8)
        column.addWidget(self._build_titlebar())
        column.addLayout(body, 1)
        self.setCentralWidget(shell)
        self.grip = QSizeGrip(shell)             # the frameless window is resized from its corner

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
        self._sync_nav()
        self._place_previews(self.width() < NARROW_WIDTH)
        for button in self.findChildren(QAbstractButton):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        for b in self.winctl_buttons:
            b.setVisible(self.window_buttons.isChecked())

    # ---- layout --------------------------------------------------------------------------
    def _build_sidebar(self):
        """The nav buttons keep their size: when the window is too low for all of them the sidebar scrolls."""
        inner = QWidget()
        box = QVBoxLayout(inner)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(8)
        for i in range(4):                                         # upper deck: Values .. Amplitudes
            box.addWidget(self._nav_button(self.tabs.widget(i), self.tabs.tabText(i)))
        self.previews_nav = self._nav_button(self.previews_area, "Previews")   # shown when narrow
        self.previews_nav.hide()
        box.addWidget(self.previews_nav)
        box.addWidget(self._nav_button(self.output_page, "Output"))
        box.addStretch(1)
        box.addWidget(self._nav_button(self.logs_page, "Logs"))      # lower deck: Logs, Settings
        box.addWidget(self._nav_button(self.settings_page, "Settings"))
        w = QScrollArea()
        w.setObjectName("sidebar")
        w.setWidgetResizable(True)
        w.setWidget(inner)
        w.setFixedWidth(108)
        w.setFrameShape(QFrame.Shape.NoFrame)
        w.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        w.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)     # the wheel scrolls it
        return w

    def _nav_button(self, page, text):
        muted = self.palette().color(QPalette.ColorRole.PlaceholderText)
        b = NavButton(text)
        b.setIcon(icon(NAV_ICONS[text], muted, checked=self.palette().color(QPalette.ColorRole.Highlight)))
        b.clicked.connect(lambda _=False, w=page: self._show_page(w))
        self.nav_buttons[page] = b
        return b

    def _show_page(self, page):
        self.tabs.setCurrentWidget(page)
        self._sync_nav()

    def _sync_nav(self, _index=None):
        """Only the current page's button is checked. The buttons are not an exclusive group: Qt then lets Tab
        stop at the checked button, so the rest of the sidebar could not be reached from the keyboard."""
        current = self.nav_buttons.get(self.tabs.currentWidget())
        for button in self.nav_buttons.values():
            button.setChecked(button is current)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.grip.move(self.centralWidget().width() - self.grip.width(), self.centralWidget().height() - self.grip.height())
        self.grip.raise_()
        self._place_previews(self.width() < NARROW_WIDTH)

    def _place_previews(self, narrow):
        """Narrow window: the previews become a tab before Output, as wide as the window and never narrower than
        their content. Wide: beside the pages at the fixed width their camera cells want, so a wider window gives
        the extra width to the pages. Neither scrolls sideways."""
        if narrow == self.previews_narrow:
            return
        self.previews_narrow = narrow
        content = self.previews_area.widget()
        scrollbar = self.previews_area.verticalScrollBar().sizeHint().width()
        if narrow:
            self.previews_area.setMaximumWidth(NO_MAX_WIDTH)
            self.previews_area.setMinimumWidth(content.minimumSizeHint().width() + scrollbar)
            self.tabs.insertTab(self.tabs.indexOf(self.output_page), self.previews_area, "Previews")
        else:
            if self.tabs.indexOf(self.previews_area) >= 0:
                self.tabs.removeTab(self.tabs.indexOf(self.previews_area))
                self.split.addWidget(self.previews_area)
                self.previews_area.show()
            self.previews_area.setFixedWidth(content.sizeHint().width() + scrollbar)
        self.previews_nav.setVisible(narrow)

    def _build_titlebar(self):
        bar = TitleBar(self)
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 0, 8, 0)
        row.setSpacing(8)
        logo = QLabel()
        logo.setPixmap(QIcon(icon_path()).pixmap(22, 22))
        row.addWidget(logo)
        name = QLabel("OpenCymple")
        name.setObjectName("appName")
        row.addWidget(name)
        row.addWidget(_badge("v%s" % __version__, "success"))
        self.source_badge = _badge("", "accent")
        row.addWidget(self.source_badge)
        row.addStretch(1)
        muted = self.palette().color(QPalette.ColorRole.PlaceholderText)
        self.winctl_buttons = []
        for name, label, slot in (("minus", "Minimize", self.showMinimized), ("maximize", "Maximize", self.toggle_maximized),
                                  ("x", "Close", self.close)):
            b = QToolButton()
            b.setObjectName("winctl")
            b.setIcon(icon(name, muted))
            b.setIconSize(QSize(16, 16))
            b.setToolTip(label)                  # icon only: the tooltip is also the accessible name
            b.setAccessibleName(label)
            b.clicked.connect(lambda _=False, f=slot: f())
            row.addWidget(b)
            self.winctl_buttons.append(b)
        return bar

    def toggle_maximized(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _build_previews(self):
        w = QWidget()
        w.setObjectName("panel")
        outer = QVBoxLayout(w)
        outer.setContentsMargins(16, 16, 16, 16)
        title = QLabel("Previews")
        title.setObjectName("panelTitle")
        outer.addWidget(title)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        outer.addLayout(grid, 1)
        self.eye_panels = {}
        for stream, eye in P.EYE_STREAMS.items():                      # eyes: the model input only
            if stream in self.rt.pipeline.workers:
                self.eye_panels[stream] = StreamPanel(("Left eye", "Right eye")[eye], camera=False)
                grid.addWidget(self.eye_panels[stream], 0, eye)
        self.mouth_panel = None
        if P.MOUTH_V1_STREAM in self.rt.pipeline.workers:             # mouth: the camera feed only
            self.mouth_panel = StreamPanel("Mouth", editable=True, model=False)
            self.mouth_panel.raw.rectDrawn.connect(self._mouth_rect_drawn)
            grid.addWidget(self.mouth_panel, 1, 0)
        box = QGroupBox("Gaze and eyelids")
        self.gaze = GazeView()
        inner = QVBoxLayout(box)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.addWidget(self.gaze)
        grid.addWidget(box, 1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        return w

    def _build_values_tab(self):
        self.bars = ValueBars(FIELDS, signed=FIELDS[:4], inactive=INACTIVE_FIELDS)
        w = QWidget()
        box = QVBoxLayout(w)
        card, inner = _card("Values")
        self.afk_toggle = ToggleRow("AFK Detection")
        self.afk_toggle.setChecked(self.opts["afk"]["enabled"])
        self.rt.pipeline.afk.enabled = self.afk_toggle.isChecked()
        self.afk_toggle.toggled.connect(self._afk_toggled)
        inner.addWidget(self.afk_toggle)
        inner.addWidget(_note("While the headset is off, send every value at its default instead of what the "
                              "models make of the empty cameras."))
        self.afk_status = QLabel("Headset off: sending default values")
        self.afk_status.hide()                   # shown while the headset is off
        inner.addWidget(self.afk_status)
        inner.addWidget(self.bars)
        box.addWidget(card)
        box.addStretch(1)
        return _scroll(w)

    def _build_eye_tab(self):
        cal = self.rt.calib
        w = QWidget()
        box = QVBoxLayout(w)
        card, inner = _card("Eye settings")
        self.eye_sliders = {}
        for attr, title, lo, hi, step, dec, desc in EYE_SLIDERS:
            s = SliderRow(title, lo, hi, step, dec, desc, default=getattr(C.Calibration(), attr))
            s.set_value(getattr(cal, attr))
            s.valueChanged.connect(lambda v, a=attr: self._changed(self.rt.set_eye_setting, a, v))
            self.eye_sliders[attr] = s
            inner.addWidget(s)
        box.addWidget(card)
        card, inner = _card("Eye switches")
        self.eye_switches = {}
        for slot, title in EYE_SWITCHES:
            cb = ToggleRow(title)
            self.eye_switches[slot] = cb
            cb.setChecked(bool(cal.switches.get(slot)))
            cb.toggled.connect(lambda on, s=slot: self._changed(self.rt.set_switch, s, on))
            inner.addWidget(cb)
        box.addWidget(card)
        card, inner = _card("Eye crop (ROI)")
        row = QHBoxLayout()
        for label, eyes in (("Calibrate both", (0, 1)), ("Left only", (0,)), ("Right only", (1,))):
            b = QPushButton(label)
            b.setProperty("primary", eyes == (0, 1))
            b.clicked.connect(lambda _=False, e=eyes: self._calibrate_roi(e))
            row.addWidget(b)
        row.addStretch(1)
        inner.addLayout(row)
        self.roi_status = QLabel()
        self.roi_status.hide()                   # shown once a calibration has run
        inner.addWidget(self.roi_status)
        box.addWidget(card)
        card, inner = _card("Gaze calibration")
        inner.addWidget(_note("Recenter: look straight ahead with relaxed eyes. It updates the eye centre, the "
                              "eyelid, squint and widen references and the expression neutral; in precise mode it "
                              "also moves the 13-point calibration to the new centre."))
        inner.addWidget(_note("Precise shows 13 dots to look at, one capture each (Cymple's full-view calibration). "
                              "In the headset: left stick sets the size, right stick the height, right trigger "
                              "starts; then look at each yellow dot and pull the trigger."))
        row = QHBoxLayout()
        recenter = QPushButton("Recenter both eyes")
        recenter.clicked.connect(lambda: self.rt.recenter(wait=False))
        self.headset_button = QPushButton("Calibrate in headset")
        self.headset_button.setProperty("primary", True)
        self.headset_button.clicked.connect(self._headset_button)
        row.addWidget(recenter)
        row.addWidget(self.headset_button)
        row.addStretch(1)
        inner.addLayout(row)
        self.calib_status = QLabel()
        self.calib_status.setWordWrap(True)
        inner.addWidget(self.calib_status)
        box.addWidget(card)
        box.addStretch(1)
        return w

    def _build_mouth_tab(self):
        m = self.rt.calib.mouth
        w = QWidget()
        box = QVBoxLayout(w)
        card, inner = _card("Mouth settings")
        self.mouth_sliders = {}
        for attr, title, lo, hi, step, dec, desc, suffix in (
                ("filter", "Smoothing", 0.0, 0.5, 0.01, 2, None, ""),
                ("rotation", "Camera rotation", -180.0, 180.0, 5.0, 0, None, "°")):
            s = SliderRow(title, lo, hi, step, dec, desc, suffix, default=getattr(C.MouthCalibration(), attr))
            s.set_value(getattr(m, attr))
            s.valueChanged.connect(lambda v, a=attr: self._changed(self.rt.set_mouth_setting, a, v))
            self.mouth_sliders[attr] = s
            inner.addWidget(s)
        box.addWidget(card)
        card, inner = _card("Mouth area")
        self.auto_roi = ToggleRow("Find the mouth automatically")
        self.auto_roi.setChecked(m.auto_roi)
        self.auto_roi.toggled.connect(self._auto_roi_toggled)
        inner.addWidget(self.auto_roi)
        self.area_sliders = {}
        self.area_box = QWidget()
        area = QVBoxLayout(self.area_box)
        area.setContentsMargins(0, 0, 0, 0)
        area.setSpacing(8)
        for attr, title, lo, hi in (("upper", "Top", 0.0, 98.0), ("bottom", "Bottom", 1.0, 99.0),
                                    ("width", "Width", 1.0, 100.0), ("x_offset", "Horizontal centre", 0.0, 100.0)):
            s = SliderRow(title, lo, hi, 0.5, 1, suffix=" %", default=getattr(C.MouthCalibration(), attr) * 100.0)
            s.valueChanged.connect(lambda v, a=attr: self._area_changed(a, v))
            self.area_sliders[attr] = s
            area.addWidget(s)
        inner.addWidget(self.area_box)
        self._sync_area_sliders()
        box.addWidget(card)
        box.addStretch(1)
        return w

    def _build_amp_tab(self):
        w = QWidget()
        box = QVBoxLayout(w)
        card, inner = _card("Amplitudes")
        inner.addWidget(_note("Higher values make the expression stronger (0 to 5)."))
        row = QHBoxLayout()
        self.amp_scope = QLabel()
        self.amp_scope.setWordWrap(True)
        row.addWidget(self.amp_scope, 1)
        self.amp_button = QPushButton()
        self.amp_button.clicked.connect(self._amp_button)
        row.addWidget(self.amp_button)
        inner.addLayout(row)
        grid = QGridLayout()
        self.amp_sliders = {}
        for k, slot in enumerate(AMP_ORDER):
            s = SliderRow(AMP_LABELS[slot], 0.0, 5.0, 0.05, 2, default=1.0)
            s.valueChanged.connect(lambda v, i=slot: self._changed(self.rt.set_amp, i, v))
            self.amp_sliders[slot] = s
            grid.addWidget(s, k // 2, k % 2)
        inner.addLayout(grid)
        box.addWidget(card)
        box.addStretch(1)
        self._update_amp_view(force=True)
        return w

    def _build_output_tab(self):
        o = self.opts["osc"]
        w = QWidget()
        box = QVBoxLayout(w)
        g = QGroupBox("VRChat (OSC)")
        form = QFormLayout(g)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.osc_enabled = ToggleRow("Send avatar parameters to VRChat")
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
        self.osc_config_dir = QLineEdit(o["config_dir"])
        detected = avatar_out.osc_config_dirs()
        self.osc_config_dir.setPlaceholderText(detected[0] if detected else "Autodetect (nothing found)")
        self.osc_config_dir.setToolTip("VRChat's OSC folder, the one containing usr_*/Avatars. "
                                       "Leave empty to autodetect in the Steam Proton prefix.")
        self.osc_config_dir.editingFinished.connect(self._osc_option_changed)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_osc_config_dir)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.osc_config_dir, 1)
        row.addWidget(browse)
        form.addRow("Avatar config folder", row)
        self.osc_status = QLabel("-")
        self.osc_status.setWordWrap(True)
        form.addRow("Status", self.osc_status)
        box.addWidget(g)

        g, v = _card("Cameras")
        self.source_label = QLabel()
        self.source_label.setWordWrap(True)
        self.source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.source_label)
        box.addWidget(g)
        box.addStretch(1)
        return w

    def _build_logs_tab(self):
        w = QWidget()
        box = QVBoxLayout(w)
        box.setContentsMargins(16, 16, 16, 16)
        row = QHBoxLayout()
        row.addWidget(_note("Log lines since OpenCymple started, newest last."), 1)
        b = QPushButton("Export Logs")
        b.setIcon(icon("file-up", self.palette().color(QPalette.ColorRole.ButtonText)))
        b.clicked.connect(self._export_logs)
        row.addWidget(b)
        box.addLayout(row)
        self.log_view = LogView(logs.BUFFER)
        box.addWidget(self.log_view, 1)
        return w

    def _build_settings_tab(self):
        w = QWidget()
        box = QVBoxLayout(w)
        g, v = _card("Window")
        self.window_buttons = ToggleRow("Show minimize, maximize and close buttons")
        self.window_buttons.setChecked(self.opts.get("window", {}).get("buttons", True))
        self.window_buttons.toggled.connect(self._window_buttons_toggled)
        v.addWidget(self.window_buttons)
        self.close_to_tray = ToggleRow("Closing the window keeps OpenCymple running in the tray")
        self.close_to_tray.setChecked(self.opts.get("window", {}).get("close_to_tray", True))
        self.close_to_tray.toggled.connect(self._close_to_tray_toggled)
        v.addWidget(self.close_to_tray)
        box.addWidget(g)
        g, v = _card("OpenCymple config")
        text = self.palette().color(QPalette.ColorRole.ButtonText)
        row = QHBoxLayout()
        b = QPushButton("Import")
        b.setIcon(icon("file-down", text))
        b.clicked.connect(self._import)
        row.addWidget(b)
        b = QPushButton("Export")
        b.setIcon(icon("file-up", text))
        b.clicked.connect(self._export)
        row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        box.addWidget(g)
        g, v = _card("Integration")
        self.entry_button = QPushButton()
        self.entry_button.clicked.connect(self._toggle_entry)
        v.addWidget(self.entry_button)
        self.link_button = QPushButton()
        self.link_button.clicked.connect(self._toggle_link)
        v.addWidget(self.link_button)
        self.integration_note = _note("")
        v.addWidget(self.integration_note)
        self._sync_integration()
        box.addWidget(g)
        box.addStretch(1)
        return w

    # ---- changes -------------------------------------------------------------------------
    def _sync_integration(self, note=""):
        self.entry_button.setText("Remove desktop entry" if desktop.installed() else "Install desktop entry")
        self.link_button.setText("Remove `opencymple` from terminal" if desktop.linked()
                                 else "Add `opencymple` to terminal")
        if not note and desktop.linked() and not desktop.on_path():
            note = "%s is not on your PATH" % os.path.dirname(desktop.link_path())
        self.integration_note.setText(note)

    def _toggle_entry(self):
        try:
            if desktop.installed():
                desktop.uninstall_entry()
                note = "Desktop entry removed."
            else:
                desktop.install_entry()
                note = "OpenCymple is in the application menu and in WiVRn's list of applications."
        except OSError as e:
            log.error("desktop entry failed: %s", e)
            note = "Failed: %s" % e
        self._sync_integration(note)

    def _toggle_link(self):
        try:
            if desktop.linked():
                desktop.unlink_command()
                note = "Removed %s." % desktop.link_path()
            elif desktop.link_command():
                note = "Linked %s." % desktop.link_path()
            else:
                note = "Nothing linked: %s exists or OpenCymple has no console script." % desktop.link_path()
        except OSError as e:
            log.error("terminal command failed: %s", e)
            note = "Failed: %s" % e
        self._sync_integration(note)

    def _afk_toggled(self, on):
        self.rt.pipeline.afk.enabled = on
        self.opts["afk"]["enabled"] = on
        S.save_options(self.opts)

    def _window_buttons_toggled(self, on):
        """Tiling window managers have no use for the title bar's buttons."""
        self.opts.setdefault("window", {})["buttons"] = on
        S.save_options(self.opts)
        for b in self.winctl_buttons:
            b.setVisible(on)

    def _close_to_tray_toggled(self, on):
        self.opts.setdefault("window", {})["close_to_tray"] = on
        S.save_options(self.opts)

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

    def _headset_button(self):
        if self.headset.running:
            self.headset.stop()
            return
        why = self.headset.start()
        if why:
            self.rt.calib_status = why

    def _update_eye_status(self, last):
        self.roi_status.setText("Last calibration: %s" % last)
        self.roi_status.show()

    def _auto_roi_toggled(self, on):
        self._changed(self.rt.set_mouth_setting, "auto_roi", on)
        self.area_box.setVisible(not on)

    def _sync_area_sliders(self):
        m = self.rt.calib.mouth
        for attr, s in self.area_sliders.items():
            s.set_value(getattr(m, attr) * 100.0)
        self.area_box.setVisible(not m.auto_roi)

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
        if own:
            self.amp_scope.setText("Editing the own set of %s. Deleting it goes back to the default set." % name)
            self.amp_button.setText("Delete this avatar's set")
        else:
            self.amp_scope.setText("Editing the default set, used by every avatar without its own set%s."
                                   % (" (current avatar: %s)" % name if avatar else ""))
            self.amp_button.setText("Give %s its own set" % ("this avatar" if avatar else "an avatar"))
        self.amp_button.setEnabled(bool(avatar))

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

    def _browse_osc_config_dir(self):
        start = self.osc_config_dir.text().strip() or self.osc_config_dir.placeholderText()
        d = QFileDialog.getExistingDirectory(self, "VRChat OSC config folder",
                                             start if os.path.isdir(start) else os.path.expanduser("~"))
        if d:
            self.osc_config_dir.setText(d)
            self._osc_option_changed()

    def _osc_option_changed(self, *_):
        target = self.osc_target.text().strip()
        invalid = not TARGET_RE.match(target)
        self.osc_target.setProperty("invalid", invalid)
        _repolish(self.osc_target)
        if invalid:
            return
        o = self.opts["osc"]
        new = {"enabled": self.osc_enabled.isChecked(), "target": target, "avatar": self.osc_avatar.currentData(),
               "config_dir": self.osc_config_dir.text().strip()}
        if new == o and (self.rt.vrchat is not None) == new["enabled"]:
            return
        o.update(new)
        S.save_options(self.opts)
        self._apply_osc()

    def _apply_osc(self):
        o = self.opts["osc"]
        if self.osc_enabled.isChecked():
            try:
                self.rt.start_osc(o["target"], o["avatar"], self.args.osc_listen, o["config_dir"])
            except (OSError, ValueError) as e:
                QMessageBox.warning(self, "VRChat output", "Cannot start the OSC output: %s" % e)
        else:
            self.rt.stop_osc()

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import", "", "OpenCymple config (*.json);;All files (*)")
        if not path:
            return
        self.rt.flush()                          # pending changes go into our copy first, so the backup has them
        try:
            S.import_bundle(path)
            self.rt.reload()
        except (OSError, ValueError) as e:
            log.error("import from %s failed: %s", path, e)
            return
        self._sync_from_runtime()
        log.info("imported %s", path)

    def _sync_from_runtime(self):
        """Show the running calibration in the controls (after an import)."""
        cal = self.rt.calib
        for attr, s in self.eye_sliders.items():
            s.set_value(getattr(cal, attr))
        for slot, cb in self.eye_switches.items():
            cb.blockSignals(True)
            cb.setChecked(bool(cal.switches.get(slot)))
            cb.blockSignals(False)
        for attr, s in self.mouth_sliders.items():
            s.set_value(getattr(cal.mouth, attr))
        self.auto_roi.blockSignals(True)
        self.auto_roi.setChecked(cal.mouth.auto_roi)
        self.auto_roi.blockSignals(False)
        self._sync_area_sliders()
        self._update_amp_view(force=True)

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export", "opencymple-config.json",
                                              "OpenCymple config (*.json);;All files (*)")
        if not path:
            return
        self.rt.flush()
        try:
            S.export_bundle(path, self.rt.ini)
        except OSError as e:
            log.error("export to %s failed: %s", path, e)

    def _export_logs(self):
        name = "opencymple-%s.log" % time.strftime("%Y%m%d-%H%M%S")
        path, _ = QFileDialog.getSaveFileName(self, "Export Logs", name, "Log files (*.log *.txt);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(logs.BUFFER.text())
        except OSError as e:
            QMessageBox.warning(self, "Export Logs", "Cannot write %s: %s" % (path, e))

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
        self.log_view.refresh()
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
        else:
            st = v.status()
            now = time.monotonic()
            t0, n0 = self.sent_prev
            rate = (st["sent"] - n0) / (now - t0) if now > t0 else 0.0
            self.sent_prev = (now, st["sent"])
            self.osc_status.setText("avatar %s (%s), %d parameters, %.0f messages/s" % (
                st["avatar"] or "unknown: default addresses", st["source"], st["parameters"], rate))
        self._update_amp_view()
        if self.rt.pipeline.afk.away == self.afk_status.isHidden():
            self.afk_status.setVisible(self.rt.pipeline.afk.away)
        headset_on = self.headset.running
        status = self.rt.calib_status
        hs = self.headset.status()
        if hs:
            status = "%s%s" % (hs, ". " + status if status and headset_on else "")
        self.calib_status.setText(status)
        left = self.rt.coarse_remaining()        # a coarse run started from the tray or the CLI
        self.headset_button.setText("Stop headset calibration" if headset_on else "Calibrate in headset")
        self.headset_button.setEnabled(headset_on or left is None)
        self.source_badge.setText(self.rt.source_desc or "-")

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
            if not self.opts.get("window", {}).get("close_to_tray", True) and not self.shell.quitting:
                QTimer.singleShot(0, self.shell.quit)
            return
        self.headset.stop(wait=5.0)
        self.rt.stop()
        super().closeEvent(event)


def _load_fonts():
    """Poppins from resources/fonts (OFL-Poppins.txt, the licence, sits beside it and is not a font); without
    the files Qt uses the system sans."""
    folder = os.path.join(RESOURCES, "fonts")
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".ttf", ".otf")):
                QFontDatabase.addApplicationFont(os.path.join(folder, name))


def _apply_theme(app):
    """Fusion with the dark SlimeVR-style theme (resources/style.qss), whatever the desktop's scheme."""
    app.setStyle("Fusion")
    _load_fonts()
    p = QPalette()
    roles = {QPalette.ColorRole.Window: "#060f1c", QPalette.ColorRole.WindowText: "#f2f6fb",
             QPalette.ColorRole.Base: "#153453", QPalette.ColorRole.AlternateBase: "#0f2742",
             QPalette.ColorRole.Text: "#f2f6fb", QPalette.ColorRole.Button: "#0f2742",
             QPalette.ColorRole.Light: "#1b3a5c", QPalette.ColorRole.Midlight: "#153453",
             QPalette.ColorRole.Mid: "#0a1d33", QPalette.ColorRole.Dark: "#04121f",
             QPalette.ColorRole.Shadow: "#000a14",
             QPalette.ColorRole.ButtonText: "#f2f6fb", QPalette.ColorRole.Highlight: "#4cc2ff",
             QPalette.ColorRole.HighlightedText: "#04121f", QPalette.ColorRole.ToolTipBase: "#153453",
             QPalette.ColorRole.ToolTipText: "#f2f6fb", QPalette.ColorRole.PlaceholderText: "#8ba3bd",
             QPalette.ColorRole.Link: "#4cc2ff"}
    for role, color in roles.items():
        p.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor("#5a7691"))
    app.setPalette(p)
    with open(os.path.join(RESOURCES, "style.qss"), encoding="utf-8") as f:
        app.setStyleSheet(f.read())


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
    # file dialogs go through xdg-desktop-portal: the desktop's own file manager, not Qt's built-in one
    os.environ["QT_QPA_PLATFORMTHEME"] = "xdgdesktopportal"
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("opencymple-linux")
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
