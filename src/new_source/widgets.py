"""Виджеты графика: WavePlotWidget, оси, FlowLayout, свечение, тень, градиент."""

import backend  # noqa: F401
import math
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from theme import DEFAULT_THEME, THEME_HEX


class GlowItem(pg.GraphicsObject):
    def __init__(self, specs):
        super().__init__()
        self._specs = specs
        self._pens = []
        self._path = None
        self._rect = QtCore.QRectF()
        self._dy = 0.0
        self.setZValue(-1)

    def set_color(self, color):
        self._pens = []
        for _dy, alpha in self._specs:
            c = QtGui.QColor(color)
            c.setAlpha(alpha)
            self._pens.append(pg.mkPen(c, width=1))
        self.update()

    def set_data(self, x, y, dy_data, lim):
        if len(x) < 2:
            self.clear()
            return
        self._path = pg.functions.arrayToQPath(x, y, connect="all", finiteCheck=False)
        self._dy = dy_data
        yb = lim + 4.0 * dy_data
        x0 = math.floor(float(x[0])) - 1
        x1 = math.ceil(float(x[-1])) + 1
        r = QtCore.QRectF(x0, -yb, x1 - x0, 2 * yb)
        if r != self._rect:
            self.prepareGeometryChange()
            self._rect = r
        self.update()

    def clear(self):
        if self._path is not None:
            self._path = None
            self.update()

    def boundingRect(self):
        return self._rect

    def paint(self, p, *args):
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
    SPECS = [(-3, 18), (-2, 34), (-1, 62), (1, 62), (2, 34), (3, 18)]
    DS_FACTOR = 1.5

    def __init__(self, plot_item):
        self._plot = plot_item
        self._color = QtGui.QColor(THEME_HEX[DEFAULT_THEME])
        self._enabled = False
        self._item = None
        self._xbuf = None
        self._ybuf = None

    def set_color(self, color):
        self._color = QtGui.QColor(color)
        if self._item is not None:
            self._item.set_color(self._color)

    def set_enabled(self, on: bool):
        on = bool(on)
        self._enabled = on
        if on:
            if self._item is None:
                it = GlowItem(self.SPECS)
                it.set_color(self._color)
                self._plot.addItem(it)
                self._item = it
        elif self._item is not None:
            self._plot.removeItem(self._item)
            self._item = None
            self._xbuf = None
            self._ybuf = None

    def is_enabled(self) -> bool:
        return self._enabled

    def update(self, x, y, viewbox, lim=1.0):
        if not self._enabled or self._item is None:
            return
        n = len(y)
        try:
            width_px = float(viewbox.width())
            dy_data = float(viewbox.viewPixelSize()[1])
        except Exception:
            width_px = 0.0
            dy_data = 0.0

        ds = 1
        if width_px > 1.0:
            ds = max(1, int(n / (width_px * self.DS_FACTOR)))
        if ds > 1:
            m = n // ds
            if self._xbuf is None or self._xbuf.size < m:
                self._xbuf = np.empty(max(m, 1024), dtype=np.float32)
                self._ybuf = np.empty(max(m, 1024), dtype=np.float32)
            xs = np.add(x[: m * ds : ds], 0.5 * (ds - 1), out=self._xbuf[:m])
            ys = np.mean(y[: m * ds].reshape(m, ds), axis=1, out=self._ybuf[:m])
        else:
            xs = x
            ys = y
        self._item.set_data(xs, ys, dy_data, lim)


class WavePlotWidget(pg.PlotWidget):
    transparent_bg = False

    def drawBackground(self, painter, rect):
        if self.transparent_bg:
            painter.save()
            painter.setCompositionMode(
                QtGui.QPainter.CompositionMode.CompositionMode_Source
            )
            painter.fillRect(rect, QtGui.QColor(0, 0, 0, 3))
            painter.restore()
        else:
            super().drawBackground(painter, rect)


def _gv_cache_mode(none: bool):
    GV = QtWidgets.QGraphicsView
    name = "CacheNone" if none else "CacheBackground"
    try:
        return getattr(GV.CacheModeFlag, name)
    except Exception:
        return getattr(GV, name)


class SpacedAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        if self.orientation in ("bottom", "top"):
            return [f"{int(round(v)):,}".replace(",", " ") for v in values]
        return [f"{v:g}" for v in values]


class FlowLayout(QtWidgets.QLayout):
    def __init__(self, hspacing=16, vspacing=12):
        super().__init__()
        self._items = []
        self._h = hspacing
        self._v = vspacing

    def addItem(self, item):
        self._items.append(item)

    def count(self):
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

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout(QtCore.QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
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
            line_h = max(line_h, hint.height())
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


def dark_titlebar(widget):
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

    def __init__(self, card, parent):
        super().__init__(parent)
        self._card = card
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        card.installEventFilter(self)
        self.hide()

    @classmethod
    def _image(cls):
        if cls._img is None:
            p, c = cls.PAD, cls.CORE
            sw = 2 * p + c
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
            cls._img = img
        return cls._img

    def sync(self):
        c = self._card
        par = self.parentWidget()
        if par is not None and c.parentWidget() is par and c.isVisible():
            self.setGeometry(
                c.geometry().adjusted(-self.PAD, -self.PAD, self.PAD, self.PAD)
            )
            if not self.isVisible():
                self.show()
            self.lower()
        elif self.isVisible():
            self.hide()

    def eventFilter(self, obj, ev):
        if obj is self._card:
            t = ev.type()
            ET = QtCore.QEvent.Type
            if t in (ET.Move, ET.Resize, ET.Show, ET.Hide, ET.ParentChange):
                self.sync()
        return False

    def paintEvent(self, ev):
        w, h = self.width(), self.height()
        inset = self.PAD + self.RADIUS
        region = ev.region()
        if w > 2 * inset and h > 2 * inset:
            region = region.subtracted(
                QtGui.QRegion(QtCore.QRect(inset, inset, w - 2 * inset, h - 2 * inset))
            )
        if region.isEmpty():
            return
        img = self._image()
        sw = img.width()
        a = sw // 2 - 1
        b = sw - a - 2
        if w < a + b or h < a + b:
            return
        p = QtGui.QPainter(self)
        p.setClipRegion(region)
        xs = [(0, a, 0, a), (a, 2, a, w - a - b), (a + 2, b, w - b, b)]
        ys = [(0, a, 0, a), (a, 2, a, h - a - b), (a + 2, b, h - b, b)]
        for sy, sh, dy, dh in ys:
            for sx, sw_, dx, dw in xs:
                p.drawImage(
                    QtCore.QRect(dx, dy, dw, dh), img, QtCore.QRect(sx, sy, sw_, sh)
                )
        p.end()


class GradientCurveItem(pg.GraphicsObject):
    """Кастомный элемент для поддержки градиента в OpenGL."""

    def __init__(self):
        super().__init__()
        self._path = None
        self._rect = QtCore.QRectF()
        self._color_quiet = QtGui.QColor(0, 0, 0, 0)
        self._color_loud = QtGui.QColor(255, 255, 255, 255)
        self._width = 1.7
        self._use_gradient = False
        self._solid_color = QtGui.QColor("#ffffff")

    def setPen(self, pen):
        if isinstance(pen, QtGui.QPen):
            brush = pen.brush()
            if brush.style() == QtCore.Qt.BrushStyle.LinearGradientPattern:
                self._use_gradient = True
                grad = brush.gradient()
                self._color_loud = grad.stops()[0][1]
                self._color_quiet = grad.stops()[1][1]
                self._width = pen.widthF()
            else:
                self._use_gradient = False
                self._solid_color = pen.color()
                self._width = pen.widthF()
            self.update()

    def setData(self, x, y, **kwargs):
        if len(x) < 2:
            self._path = None
            self.update()
            return
        self._path = pg.functions.arrayToQPath(x, y, connect="all", finiteCheck=False)
        new_rect = self._path.boundingRect()
        if new_rect != self._rect:
            self.prepareGeometryChange()
            self._rect = new_rect
        self.update()

    def clear(self):
        self._path = None
        self.update()

    def boundingRect(self):
        return self._rect

    def paint(self, p, *args):
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
