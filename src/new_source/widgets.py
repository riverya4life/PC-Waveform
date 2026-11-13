"""Виджеты графика: WavePlotWidget, оси, FlowLayout, свечение, тень, градиент."""

from typing import Any

import backend  # noqa: F401
import math
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from theme import DEFAULT_THEME, THEME_HEX


class GlowItem(pg.GraphicsObject):
    def __init__(self, specs) -> None:
        super().__init__()
        self._specs: Any = specs
        self._pens = []
        self._path = None
        self._rect = QtCore.QRectF()
        self._dy = 0.0
        self.setZValue(-1)

    def set_color(self, color) -> None:
        self._pens = []
        for _dy, alpha in self._specs:
            c = QtGui.QColor(color)
            c.setAlpha(alpha)
            self._pens.append(pg.mkPen(c, width=1))
        self.update()

    def set_data(self, x, y, dy_data, lim) -> None:
        if len(x) < 2:
            self.clear()
            return
        self._path: QtGui.QPainterPath = pg.functions.arrayToQPath(
            x, y, connect="all", finiteCheck=False
        )
        self._dy = dy_data
        yb = lim + 4.0 * dy_data
        x0: int = math.floor(float(x[0])) - 1
        x1: int = math.ceil(float(x[-1])) + 1
        r = QtCore.QRectF(x0, -yb, x1 - x0, 2 * yb)
        if r != self._rect:
            self.prepareGeometryChange()
            self._rect: QtCore.QRectF = r
        self.update()

    def clear(self) -> None:
        if self._path is not None:
            self._path = None
            self.update()

    def boundingRect(self):
        return self._rect

    def paint(self, p, *args) -> None:
        if self._path is None:
            return
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        for (dy_px, _a), pen in zip(self._specs, self._pens):
            p.setPen(pen)
            p.save()
            p.translate(0.0, dy_px * self._dy)
            p.drawPath(self._path)
            p.restore()


class GlowRenderer:
    SPECS: list[tuple[int, int]] = [
        (-3, 18),
        (-2, 34),
        (-1, 62),
        (1, 62),
        (2, 34),
        (3, 18),
    ]
    DS_FACTOR = 1.5

    def __init__(self, plot_item) -> None:
        self._plot: Any = plot_item
        self._color = QtGui.QColor(THEME_HEX[DEFAULT_THEME])
        self._enabled = False
        self._item = None
        self._xbuf = None
        self._ybuf = None

    def set_color(self, color) -> None:
        self._color = QtGui.QColor(color)
        if self._item is not None:
            self._item.set_color(self._color)

    def set_enabled(self, on: bool) -> None:
        on = bool(on)
        self._enabled: bool = on
        if on:
            if self._item is None:
                it: GlowItem[list[tuple[int, int]]] = GlowItem(self.SPECS)
                it.set_color(self._color)
                self._plot.addItem(it)
                self._item: GlowItem[list[tuple[int, int]]] = it
        elif self._item is not None:
            self._plot.removeItem(self._item)
            self._item = None
            self._xbuf = None
            self._ybuf = None

    def is_enabled(self) -> bool:
        return self._enabled

    def update(self, x, y, viewbox, lim=1.0) -> None:
        if not self._enabled or self._item is None:
            return
        n: int = len(y)
        try:
            width_px = float(viewbox.width())
            dy_data = float(viewbox.viewPixelSize()[1])
        except Exception:
            width_px = 0.0
            dy_data = 0.0

        ds = 1
        if width_px > 1.0:
            ds: int = max(1, int(n / (width_px * self.DS_FACTOR)))
        if ds > 1:
            m: int = n // ds
            if self._xbuf is None or self._xbuf.size < m:
                self._xbuf: np.ndarray[Any, np.dtype[np.floating]] = np.empty(
                    max(m, 1024), dtype=np.float32
                )
                self._ybuf: np.ndarray[Any, np.dtype[np.floating]] = np.empty(
                    max(m, 1024), dtype=np.float32
                )
            xs: np.ndarray[Any, np.dtype[Any]] = np.add(
                x[: m * ds : ds], 0.5 * (ds - 1), out=self._xbuf[:m]
            )
            ys = np.mean(y[: m * ds].reshape(m, ds), axis=1, out=self._ybuf[:m])
        else:
            xs = x
            ys = y
        self._item.set_data(xs, ys, dy_data, lim)


# Прозрачный фон соло-режима. Ровно 0 нельзя: Windows пропускает клики сквозь
# пиксели с нулевой альфой (окно нельзя было бы двигать и закрывать по ПКМ).
# Альфа 1/255 визуально неотличима от полной прозрачности.
SOLO_HIT_ALPHA = 1


class WavePlotWidget(pg.PlotWidget):
    transparent_bg = False

    def drawBackground(self, painter, rect) -> None:
        if self.transparent_bg:
            painter.save()
            painter.setCompositionMode(
                QtGui.QPainter.CompositionMode.CompositionMode_Source
            )
            painter.fillRect(rect, QtGui.QColor(0, 0, 0, SOLO_HIT_ALPHA))
            painter.restore()
        else:
            super().drawBackground(painter, rect)


def _gv_cache_mode(none: bool):
    GV = QtWidgets.QGraphicsView
    name: str = "CacheNone" if none else "CacheBackground"
    try:
        return getattr(GV.CacheModeFlag, name)
    except Exception:
        return getattr(GV, name)


class SpacedAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing) -> list[str]:
        if self.orientation in ("bottom", "top"):
            return [f"{int(round(v)):,}".replace(",", " ") for v in values]
        return [f"{v:g}" for v in values]


class FlowLayout(QtWidgets.QLayout):
    def __init__(self, hspacing=16, vspacing=12) -> None:
        super().__init__()
        self._items = []
        self._h: int = hspacing
        self._v: int = vspacing

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        try:
            return QtCore.Qt.Orientation(0)
        except Exception:
            try:
                return QtCore.Qt.Orientations()
            except Exception:
                return 0

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width):
        return self._layout(QtCore.QRect(0, 0, width, 0), True)

    def setGeometry(self, rect) -> None:
        super().setGeometry(rect)
        self._layout(rect, False)

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:
        size = QtCore.QSize()
        for item in self._items:
            size: QtCore.QSize = size.expandedTo(item.minimumSize())
        return size

    def _layout(self, rect, test_only):
        lines, cur = [], []
        x, line_h = rect.x(), 0
        for item in self._items:
            hint = item.sizeHint()
            if cur and x + hint.width() > rect.right() + 1:
                lines.append((cur, line_h))
                cur, x, line_h = [], rect.x(), 0
            cur.append((item, hint, x))
            x += hint.width() + self._h
            line_h: int = max(line_h, hint.height())
        if cur:
            lines.append((cur, line_h))
        y = rect.y()
        for cur, lh in lines:
            if not test_only:
                for item, hint, ix in cur:
                    item.setGeometry(
                        QtCore.QRect(QtCore.QPoint(ix, y + lh - hint.height()), hint)
                    )
            y += lh + self._v
        return (y - self._v - rect.y()) if lines else 0


def dark_titlebar(widget) -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = int(widget.winId())
        val = ctypes.c_int(1)
        for attr in (20, 19):
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(val), ctypes.sizeof(val)
            )
    except Exception:
        pass


class StaticShadow(QtWidgets.QWidget):
    PAD = 40
    CORE = 120
    RADIUS = 16
    _img = None

    def __init__(self, card, parent) -> None:
        super().__init__(parent)
        self._card: Any = card
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        card.installEventFilter(self)
        self.hide()

    @classmethod
    def _image(cls) -> QtGui.QImage:
        if cls._img is None:
            p, c = cls.PAD, cls.CORE
            sw: int = 2 * p + c
            path = QtGui.QPainterPath()
            path.addRoundedRect(QtCore.QRectF(p, p, c, c), cls.RADIUS, cls.RADIUS)
            item = QtWidgets.QGraphicsPathItem(path)
            item.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
            item.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
            fx = QtWidgets.QGraphicsDropShadowEffect()
            fx.setBlurRadius(24)
            fx.setOffset(0, 6)
            fx.setColor(QtGui.QColor(0, 0, 0, 80))
            item.setGraphicsEffect(fx)
            scene = QtWidgets.QGraphicsScene()
            scene.addItem(item)
            scene.setSceneRect(0, 0, sw, sw)
            img = QtGui.QImage(sw, sw, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(0)
            ptr = QtGui.QPainter(img)
            rect = QtCore.QRectF(0, 0, sw, sw)
            scene.render(ptr, rect, rect)
            ptr.setCompositionMode(
                QtGui.QPainter.CompositionMode.CompositionMode_DestinationOut
            )
            ptr.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            ptr.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
            ptr.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
            ptr.drawPath(path)
            ptr.end()
            del scene, item
            cls._img: QtGui.QImage = img
        return cls._img

    def sync(self) -> None:
        c: Any = self._card
        par: QtWidgets.QWidget | None = self.parentWidget()
        if par is not None and c.parentWidget() is par and c.isVisible():
            self.setGeometry(
                c.geometry().adjusted(-self.PAD, -self.PAD, self.PAD, self.PAD)
            )
            if not self.isVisible():
                self.show()
            self.lower()
        elif self.isVisible():
            self.hide()

    def eventFilter(self, obj, ev) -> bool:
        if obj is self._card:
            t = ev.type()
            ET = QtCore.QEvent.Type
            if t in (ET.Move, ET.Resize, ET.Show, ET.Hide, ET.ParentChange):
                self.sync()
        return False

    def paintEvent(self, ev) -> None:
        w, h = self.width(), self.height()
        inset: int = self.PAD + self.RADIUS
        region = ev.region()
        if w > 2 * inset and h > 2 * inset:
            region = region.subtracted(
                QtGui.QRegion(QtCore.QRect(inset, inset, w - 2 * inset, h - 2 * inset))
            )
        if region.isEmpty():
            return
        img: QtGui.QImage = self._image()
        sw: int = img.width()
        a: int = sw // 2 - 1
        b: int = sw - a - 2
        if w < a + b or h < a + b:
            return
        p = QtGui.QPainter(self)
        p.setClipRegion(region)
        xs: list[tuple[int, int, int, int]] = [
            (0, a, 0, a),
            (a, 2, a, w - a - b),
            (a + 2, b, w - b, b),
        ]
        ys: list[tuple[int, int, int, int]] = [
            (0, a, 0, a),
            (a, 2, a, h - a - b),
            (a + 2, b, h - b, b),
        ]
        for sy, sh, dy, dh in ys:
            for sx, sw_, dx, dw in xs:
                p.drawImage(
                    QtCore.QRect(dx, dy, dw, dh),
                    img,
                    QtCore.QRect(sx, sy, sw_, sh),
                )
        p.end()


class GradientCurveItem(pg.GraphicsObject):
    """Пользовательский элемент для поддержки градиента в OpenGL."""

    def __init__(self) -> None:
        super().__init__()
        self._path = None
        self._rect = QtCore.QRectF()
        self._color_quiet = QtGui.QColor(0, 0, 0, 0)
        self._color_loud = QtGui.QColor(255, 255, 255, 255)
        self._width = 1.7
        self._use_gradient = False
        self._solid_color = QtGui.QColor("#ffffff")

    def setPen(self, pen) -> None:
        if isinstance(pen, QtGui.QPen):
            brush: QtGui.QBrush = pen.brush()
            if brush.style() == QtCore.Qt.BrushStyle.LinearGradientPattern:
                self._use_gradient = True
                grad: QtGui.QGradient | None = brush.gradient()
                self._color_loud: QtGui.QColor = grad.stops()[0][1]
                self._color_quiet: QtGui.QColor = grad.stops()[1][1]
                self._width: float = pen.widthF()
            else:
                self._use_gradient = False
                self._solid_color: QtGui.QColor = pen.color()
                self._width: float = pen.widthF()
            self.update()

    def setData(self, x, y, **kwargs) -> None:
        if len(x) < 2:
            self._path = None
            self.update()
            return
        self._path: QtGui.QPainterPath = pg.functions.arrayToQPath(
            x, y, connect="all", finiteCheck=False
        )
        new_rect: QtCore.QRectF = self._path.boundingRect()
        if new_rect != self._rect:
            self.prepareGeometryChange()
            self._rect: QtCore.QRectF = new_rect
        self.update()

    def clear(self) -> None:
        self._path = None
        self.update()

    def boundingRect(self):
        return self._rect

    def paint(self, p, *args) -> None:
        if self._path is None:
            return
        if self._use_gradient:
            rect = self.boundingRect()
            grad = QtGui.QLinearGradient(rect.topLeft(), rect.bottomLeft())
            grad.setColorAt(0.0, self._color_loud)
            grad.setColorAt(1.0, self._color_quiet)
            pen = QtGui.QPen(QtGui.QBrush(grad), self._width)
            pen.setCosmetic(True)
            p.setPen(pen)
        else:
            pen = QtGui.QPen(self._solid_color, self._width)
            pen.setCosmetic(True)
            p.setPen(pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawPath(self._path)


def make_gradient_pen(
    color_quiet: QtGui.QColor, color_loud: QtGui.QColor, width: float = 1.7
) -> QtGui.QPen:
    grad = QtGui.QLinearGradient(0, 0, 0, 1)
    grad.setCoordinateMode(QtGui.QGradient.CoordinateMode.ObjectBoundingMode)
    grad.setColorAt(0.0, color_loud)
    grad.setColorAt(1.0, color_quiet)
    brush = QtGui.QBrush(grad)
    pen = QtGui.QPen(brush, float(width))
    pen.setCosmetic(True)
    return pen