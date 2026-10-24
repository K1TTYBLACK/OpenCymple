"""Painted widgets for the preview window: camera image with overlays, slider rows, the
70-value bars and the gaze plot. Nothing here knows about the pipeline."""
import math
from collections import deque

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import (QDoubleSpinBox, QHBoxLayout, QLabel, QSizePolicy, QSlider, QVBoxLayout,
                               QWidget)

LANDMARK_COLORS = ["#ffc800"] * 2 + ["#00ff00"] * 2 + ["#00a0ff"] * 2 + ["#ff3030", "#ff00ff"]
OVERLAY = QColor("#ffe000")
LEFT_COLOR, RIGHT_COLOR = QColor("#4aa3ff"), QColor("#ff9d3a")


def dim(label):
    """Secondary text: the palette's placeholder colour."""
    pal = label.palette()
    pal.setColor(QPalette.ColorRole.WindowText, pal.color(QPalette.ColorRole.PlaceholderText))
    label.setPalette(pal)
    return label


def gray_qimage(arr):
    arr = np.ascontiguousarray(arr, np.uint8)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, arr.strides[0], QImage.Format.Format_Grayscale8).copy()


class ImageView(QWidget):
    """A camera image scaled to fit, plus overlays in normalized image coordinates: `quad`
    (Cymple's P0..P3 = LT, LB, RT, RB), `rect` (x0, y0, x1, y1), `points` [(x, y, color)].
    With `editable` set, dragging draws a new rectangle (rectDrawn)."""
    rectDrawn = Signal(float, float, float, float)

    def __init__(self, caption="", parent=None):
        super().__init__(parent)
        self.caption, self.placeholder = caption, "no image"
        self.image = None
        self.quad = self.rect = None
        self.points = []
        self.editable = False
        self._drag = None
        self.setMinimumSize(120, 90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, image):
        self.image = image
        self.update()

    def _target(self):
        if self.image is None or self.image.isNull():
            return QRectF(self.rect_widget())
        iw, ih = self.image.width(), self.image.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def rect_widget(self):
        return QRectF(0, 0, self.width(), self.height())

    def _map(self, t, x, y):
        return QPointF(t.x() + x * t.width(), t.y() + y * t.height())

    def _unmap(self, pos):
        t = self._target()
        return (min(max((pos.x() - t.x()) / t.width(), 0.0), 1.0),
                min(max((pos.y() - t.y()) / t.height(), 0.0), 1.0))

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect_widget(), QColor("#101014"))
        t = self._target()
        if self.image is None:
            p.setPen(QColor("#808080"))
            p.drawText(self.rect_widget(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
        else:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.drawImage(t, self.image)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(OVERLAY, 1.5)
            p.setPen(pen)
            if self.quad is not None:
                q = self.quad
                p.drawPolygon(QPolygonF([self._map(t, *q[i]) for i in (0, 2, 3, 1)]))
            r = self._drag or self.rect
            if r is not None:
                if self._drag:
                    p.setPen(QPen(OVERLAY, 1.5, Qt.PenStyle.DashLine))
                p.drawRect(QRectF(self._map(t, r[0], r[1]), self._map(t, r[2], r[3])))
            rad = max(2.0, t.width() / 60)
            for x, y, c in self.points:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(c))
                p.drawEllipse(self._map(t, x, y), rad, rad)
        if self.caption:
            p.setPen(QColor("#e0e0e0"))
            p.drawText(QRectF(6, 4, self.width() - 12, 18), Qt.AlignmentFlag.AlignLeft, self.caption)
        p.end()

    def mousePressEvent(self, e):
        if self.editable and self.image is not None and e.button() == Qt.MouseButton.LeftButton:
            x, y = self._unmap(e.position())
            self._origin = (x, y)
            self._drag = (x, y, x, y)
            self.update()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            x, y = self._unmap(e.position())
            ox, oy = self._origin
            self._drag = (min(ox, x), min(oy, y), max(ox, x), max(oy, y))
            self.update()

    def mouseReleaseEvent(self, e):
        if self._drag is not None:
            r, self._drag = self._drag, None
            if r[2] - r[0] > 0.02 and r[3] - r[1] > 0.02:
                self.rectDrawn.emit(*r)
            self.update()


class SliderRow(QWidget):
    """Title, optional description, slider and spin box; valueChanged only on user input."""
    valueChanged = Signal(float)

    def __init__(self, title, lo, hi, step, decimals=2, description="", suffix="", compact=False, parent=None):
        super().__init__(parent)
        self.lo, self.step = lo, step
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, round((hi - lo) / step))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        self.spin.setSuffix(suffix)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(76)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title)
        if compact:
            title_label.setMinimumWidth(115)
            row.addWidget(title_label)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)
        if compact:
            self.setLayout(row)
            return
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 4, 0, 4)
        box.setSpacing(2)
        title_label.setStyleSheet("font-weight: bold")
        box.addWidget(title_label)
        if description:
            d = QLabel(description)
            d.setWordWrap(True)
            box.addWidget(dim(d))
        box.addLayout(row)

    def value(self):
        return self.spin.value()

    def set_value(self, v):
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        self.spin.setValue(v)
        self.slider.setValue(round((v - self.lo) / self.step))
        for w in (self.slider, self.spin):
            w.blockSignals(False)

    def _from_slider(self, i):
        v = self.lo + i * self.step
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.valueChanged.emit(self.spin.value())

    def _from_spin(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(round((v - self.lo) / self.step))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)


class ValueBars(QWidget):
    """One bar per field: signed fields from the centre, the rest from the left; fields the
    original never fills are greyed out."""

    COLUMN_MIN_W = 280

    def __init__(self, fields, signed=(), inactive=(), parent=None):
        super().__init__(parent)
        self.fields, self.signed, self.inactive = list(fields), set(signed), set(inactive)
        self.values = {}
        self.row_h = 17
        self.columns = 0
        self.setMinimumWidth(self.COLUMN_MIN_W)
        self._layout_columns(2)

    def _layout_columns(self, n):
        if n != self.columns:
            self.columns = n
            self.setMinimumHeight(math.ceil(len(self.fields) / n) * self.row_h + 4)

    def resizeEvent(self, event):
        self._layout_columns(2 if self.width() >= 2 * self.COLUMN_MIN_W else 1)
        super().resizeEvent(event)

    def set_values(self, values):
        self.values = values or {}
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        rows = math.ceil(len(self.fields) / self.columns)
        col_w = self.width() / self.columns
        name_w = min(150.0, col_w * 0.48)
        fm = p.fontMetrics()
        pal = self.palette()
        text, dim = pal.windowText().color(), pal.placeholderText().color()
        for i, name in enumerate(self.fields):
            c, r = divmod(i, rows)
            x, y = c * col_w + 2, 2 + r * self.row_h
            v = self.values.get(name)
            off = name in self.inactive
            p.setPen(dim if off else text)
            p.drawText(QRectF(x, y, name_w - 4, self.row_h), Qt.AlignmentFlag.AlignVCenter,
                       fm.elidedText(name, Qt.TextElideMode.ElideRight, int(name_w - 4)))
            bx, bw = x + name_w, col_w - name_w - 70
            bar = QRectF(bx, y + 3, bw, self.row_h - 6)
            p.fillRect(bar, QColor(128, 128, 128, 50))
            if v is not None and math.isfinite(v) and not off:
                color = QColor("#4aa3ff") if i < 20 else QColor("#3ccf8e")
                if name in self.signed:
                    mid = bx + bw / 2
                    w = max(-1.0, min(1.0, v)) * bw / 2
                    p.fillRect(QRectF(min(mid, mid + w), bar.y(), abs(w), bar.height()), color)
                    p.fillRect(QRectF(mid - 0.5, bar.y(), 1, bar.height()), text)
                else:
                    p.fillRect(QRectF(bx, bar.y(), max(0.0, min(1.0, v)) * bw, bar.height()), color)
            p.setPen(dim if off else text)
            p.drawText(QRectF(bx + bw + 4, y, 44, self.row_h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                       "-" if v is None or off else "%+.2f" % v if name in self.signed else "%.2f" % v)
        p.end()


class GazeView(QWidget):
    """Yaw/pitch of both eyes in [-1, 1] (up = positive pitch), short trails, and the lid-close
    value of each eye as a bar on its side."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.trail = [deque(maxlen=25), deque(maxlen=25)]
        self.lids = (None, None)
        self.setMinimumSize(200, 200)

    def set_state(self, fields):
        if not fields:
            self.lids = (None, None)
            for t in self.trail:
                t.clear()
        else:
            self.trail[0].append((fields["EyeYaw_L"], fields["EyePitch_L"]))
            self.trail[1].append((fields["EyeYaw_R"], fields["EyePitch_R"]))
            self.lids = (fields["EyeLidCloseLeft"], fields["EyeLidCloseRight"])
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bar_w, gap = 14, 10
        side = min(self.width() - 2 * (bar_w + gap), self.height()) - 8
        cx, cy = self.width() / 2, self.height() / 2
        sq = QRectF(cx - side / 2, cy - side / 2, side, side)
        text = self.palette().windowText().color()
        grid = QColor(text)
        grid.setAlpha(60)
        p.setPen(QPen(grid, 1))
        p.drawRect(sq)
        p.drawEllipse(sq.center(), side / 4, side / 4)
        p.drawLine(QPointF(sq.left(), cy), QPointF(sq.right(), cy))
        p.drawLine(QPointF(cx, sq.top()), QPointF(cx, sq.bottom()))
        for eye, color in ((0, LEFT_COLOR), (1, RIGHT_COLOR)):
            pts = list(self.trail[eye])
            for k, (yaw, pitch) in enumerate(pts):
                c = QColor(color)
                last = k == len(pts) - 1
                c.setAlpha(255 if last else 40 + 120 * k // max(1, len(pts)))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(c)
                pt = QPointF(cx + max(-1.0, min(1.0, yaw)) * side / 2, cy - max(-1.0, min(1.0, pitch)) * side / 2)
                p.drawEllipse(pt, 7 if last else 3, 7 if last else 3)
        for eye, (x, color) in enumerate(((sq.left() - gap - bar_w, LEFT_COLOR), (sq.right() + gap, RIGHT_COLOR))):
            r = QRectF(x, sq.top(), bar_w, side)
            p.fillRect(r, QColor(128, 128, 128, 50))
            v = self.lids[eye]
            if v is not None:
                h = max(0.0, min(1.0, v)) * side
                p.fillRect(QRectF(x, sq.top(), bar_w, h), color)
        p.setPen(text)
        p.drawText(QRectF(sq.left() + 4, sq.top() + 2, 200, 16), Qt.AlignmentFlag.AlignLeft, "gaze (L blue, R orange)")
        p.drawText(QRectF(sq.left() - gap - bar_w - 4, sq.bottom() + 2, bar_w + 8, 14), Qt.AlignmentFlag.AlignCenter, "L")
        p.drawText(QRectF(sq.right() + gap - 4, sq.bottom() + 2, bar_w + 8, 14), Qt.AlignmentFlag.AlignCenter, "R")
        p.end()
