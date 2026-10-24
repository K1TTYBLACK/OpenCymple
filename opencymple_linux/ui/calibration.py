"""Target display for the 13-point gaze calibration on the desktop (or through WayVR's desktop view
in the headset): the dots of Cymple's full-view layout, the current one highlighted. Space, Enter or
a click captures the point the user is looking at; Esc stops; + and - change the ring size on screen
(display only, the saved overlay geometry is kept)."""
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..calib import MESH_TARGETS

BACKGROUND, DOT, CURRENT, DONE = QColor("#0b0c10"), QColor("#5b6070"), QColor("#ffe000"), QColor("#3ccf8e")


class MeshTargetWindow(QWidget):
    finished = Signal()

    def __init__(self, runtime, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.rt = runtime
        self.setWindowTitle("OpenCymple Linux: 13-point gaze calibration")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.scale = 0.40                        # outer ring radius as a fraction of the shorter side
        self.message = "Look at the yellow dot, then press Space."
        self.done = False
        self.pulse = 0
        self.timer = QTimer(self, interval=50)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.rt.start_mesh()
        self.resize(1200, 900)

    def _tick(self):
        self.pulse = (self.pulse + 1) % 20
        self.update()

    def _capture(self):
        if self.done:
            return
        rep = self.rt.capture_mesh_point()
        if rep is None:
            return
        if rep.get("done"):
            self.done = True
            saved = rep["saved"]
            names = ("left", "right")
            self.message = "Saved: %s. Closing..." % ", ".join(
                "%s %s" % (names[e], "yes" if ok else "no (points missing)") for e, ok in saved.items())
            QTimer.singleShot(2000, self.close)
        elif "skipped" in rep:
            self.message = "Point %d not captured: %s. Keep the eyes open and try again." % (rep["point"], rep["skipped"])
        else:
            self.message = "Point %d captured. Look at the next dot, then press Space." % rep["point"]
        self.update()

    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._capture()
        elif k == Qt.Key.Key_Escape:
            self.close()
        elif k in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.scale = min(0.49, self.scale + 0.02)
        elif k == Qt.Key.Key_Minus:
            self.scale = max(0.15, self.scale - 0.02)
        elif k == Qt.Key.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        else:
            super().keyPressEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._capture()

    def closeEvent(self, e):
        self.timer.stop()
        if not self.done:
            self.rt.cancel_mesh()
        self.finished.emit()
        super().closeEvent(e)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BACKGROUND)
        cx, cy = self.width() / 2, self.height() / 2
        r = self.scale * min(self.width(), self.height())
        m = self.rt.calibrator.mesh
        step = m.step if m is not None else len(MESH_TARGETS)
        captured = (set(i for i, v in enumerate(m.raw[0]) if v is not None) |
                    set(i for i, v in enumerate(m.raw[1]) if v is not None)) if m is not None else set()
        p.setPen(QPen(QColor(255, 255, 255, 25), 1))
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.drawEllipse(QPointF(cx, cy), 0.45 * r, 0.45 * r)
        for i, (ty, tp) in enumerate(MESH_TARGETS):
            pt = QPointF(cx + ty * r, cy - tp * r)          # yaw > 0 right, pitch > 0 up
            if i == step and not self.done:
                rad = 14 + (self.pulse if self.pulse < 10 else 20 - self.pulse) * 0.8
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 224, 0, 60))
                p.drawEllipse(pt, rad * 2, rad * 2)
                p.setBrush(CURRENT)
                p.drawEllipse(pt, 9, 9)
                p.setBrush(BACKGROUND)
                p.drawEllipse(pt, 2.5, 2.5)
            else:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(DONE if i in captured else DOT)
                p.drawEllipse(pt, 6, 6)
            p.setPen(QColor("#9aa0aa"))
            p.setFont(QFont(p.font().family(), 9))
            p.drawText(QRectF(pt.x() + 10, pt.y() - 22, 30, 16), Qt.AlignmentFlag.AlignLeft, str(i))
        p.setPen(QColor("#e6e6e6"))
        f = p.font()
        f.setPointSize(13)
        p.setFont(f)
        status = "Point %d of 12" % min(step, 12) if not self.done else "Done"
        p.drawText(QRectF(20, 16, self.width() - 40, 24), Qt.AlignmentFlag.AlignLeft, status)
        p.drawText(QRectF(20, self.height() - 70, self.width() - 40, 24), Qt.AlignmentFlag.AlignLeft, self.message)
        f.setPointSize(10)
        p.setFont(f)
        p.setPen(QColor("#9aa0aa"))
        p.drawText(QRectF(20, self.height() - 40, self.width() - 40, 20), Qt.AlignmentFlag.AlignLeft,
                   "Space / Enter / click: capture   Esc: stop   + / -: ring size   F11: full screen")
        p.end()
