"""Icons for the window: Lucide (ISC, resources/icons/LICENSE-lucide.txt). The SVGs stroke with
currentColor, so one file is drawn in whatever colour a widget state needs."""
import os

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

ICON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "icons")
RENDER_PX = 64                                   # drawn large, scaled down by Qt: crisp at button sizes


def _pixmap(name, color):
    with open(os.path.join(ICON_DIR, name + ".svg"), encoding="utf-8") as f:
        svg = f.read().replace("currentColor", color.name())
    pm = QPixmap(RENDER_PX, RENDER_PX)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    QSvgRenderer(QByteArray(svg.encode("utf-8"))).render(p, QRectF(0, 0, RENDER_PX, RENDER_PX))
    p.end()
    return pm


def icon(name, color, checked=None):
    """The icon `name` (a file in resources/icons, without .svg) in `color`; `checked`, when given, is
    the colour a checked button shows (its On state)."""
    qicon = QIcon(_pixmap(name, color))
    if checked is not None:
        qicon.addPixmap(_pixmap(name, checked), QIcon.Mode.Normal, QIcon.State.On)
    return qicon
