"""Темы, QSS, цветовые хелперы и кастомные виджеты интерфейса (логотип, свитчи, сегменты и т.д.)."""

import backend  # noqa: F401  (переменные окружения и GL-конфиг — до numpy/Qt)
import functools
import math
import sys
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from settings import CONFIG, DEFAULT_CONFIG

THEMES = [
    ("mint", "#00ff8c"),
    ("cyan", "#19c8ff"),
    ("azure", "#4d8dff"),
    ("indigo", "#7c6cff"),
    ("violet", "#b26bff"),
    ("pink", "#ff50a0"),
    ("raspberry", "#ff2e63"),
    ("coral", "#ff6b4a"),
    ("amber", "#ffb000"),
    ("lime", "#b6ff2e"),
    ("turquoise", "#2ee6d6"),
    ("white", "#ebebeb"),
]


THEME_HEX = dict(THEMES)


DEFAULT_THEME = "mint"


# custom/system themes are allowed; custom colour lives in CONFIG["custom_color"],
# system colour is read from the OS on demand (system_accent())
if CONFIG.get("theme") not in THEME_HEX and CONFIG.get("theme") not in (
    "custom",
    "system",
):
    CONFIG["theme"] = DEFAULT_THEME


_cc = str(CONFIG.get("custom_color", DEFAULT_CONFIG["custom_color"]))


if not (_cc.startswith("#") and len(_cc) in (4, 7)):
    CONFIG["custom_color"] = DEFAULT_CONFIG["custom_color"]


# ── Системный акцент ──────────────────────────────────────────────────────────
# Windows: реестр (мгновенно, без доп. зависимостей). Иначе — QPalette.Accent
# (Qt ≥ 6.6: Windows 10/11, macOS и часть Linux-окружений).
# минимальная яркость (HSV) — тёмный акцент не «тонет» на тёмном фоне
_SYS_MIN_V = 190


def _win_accent_hex():
    try:
        import winreg
    except Exception:
        return None
    # (ветка, значение, порядок байт). AccentColor/AccentColorMenu — ABGR,
    # ColorizationColor — ARGB.
    sources = (
        (r"Software\Microsoft\Windows\DWM", "AccentColor", "abgr"),
        (
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Accent",
            "AccentColorMenu",
            "abgr",
        ),
        (r"Software\Microsoft\Windows\DWM", "ColorizationColor", "argb"),
    )
    for sub, name, order in sources:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub) as k:
                v, _ = winreg.QueryValueEx(k, name)
        except OSError:
            continue
        try:
            v = int(v) & 0xFFFFFFFF
        except Exception:
            continue
        if order == "abgr":
            r, g, b = v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF
        else:
            r, g, b = (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}"
    return None


def _qt_accent_hex():
    try:
        role = getattr(QtGui.QPalette.ColorRole, "Accent", None)
        if role is None:  # Qt < 6.6
            return None
        c = QtGui.QGuiApplication.palette().color(role)
        return c.name() if c.isValid() else None
    except Exception:
        return None


def readable_accent(hex_color: str) -> str:
    """Поднимает яркость слишком тёмного цвета, сохраняя оттенок: интерфейс
    рисует тёмный текст на акценте и линию на тёмном фоне."""
    c = QtGui.QColor(hex_color)
    h, s, v, _a = c.getHsv()
    if v >= _SYS_MIN_V:
        return c.name()
    return QtGui.QColor.fromHsv(max(h, 0), s, _SYS_MIN_V).name()


def system_accent():
    """Цвет акцента системы (#rrggbb, уже приведённый к читаемому виду)
    или None, если определить не удалось."""
    raw = _win_accent_hex() if sys.platform == "win32" else None
    if raw is None:
        raw = _qt_accent_hex()
    return readable_accent(raw) if raw else None


@functools.lru_cache(maxsize=256)
def _mix_dark(accent_hex: str, strength: float = 0.07) -> str:
    c = QtGui.QColor(accent_hex)
    r = min(255, int(c.red() * strength) + 6)
    g = min(255, int(c.green() * strength) + 7)
    b = min(255, int(c.blue() * strength) + 9)
    return f"#{r:02x}{g:02x}{b:02x}"


def _surface(accent_hex: str) -> str:
    return _mix_dark(accent_hex, 0.060)


def _rgba(color, a: float) -> str:
    c = QtGui.QColor(color)
    return f"rgba({c.red()},{c.green()},{c.blue()},{int(max(0.0, min(1.0, a)) * 255)})"


def _lerp_color(a: QtGui.QColor, b: QtGui.QColor, t: float) -> QtGui.QColor:
    t = max(0.0, min(1.0, t))
    return QtGui.QColor(*(int(x + (y - x) * t) for x, y in zip(a.getRgb(), b.getRgb())))


def _stereo_r_color(accent_hex: str) -> QtGui.QColor:
    c = QtGui.QColor(accent_hex)
    h, s, v, a = c.getHsv()
    if h < 0:
        return QtGui.QColor(
            max(0, c.red() - 40), max(0, c.green() - 40), max(0, c.blue() - 40), 220
        )
    v2 = max(40, int(v * 0.62))
    s2 = min(255, int(s * 1.05)) if s > 0 else s
    return QtGui.QColor.fromHsv(h, s2, v2, 220)


def make_qss(accent: str) -> str:
    top = _mix_dark(accent, 0.040)
    bot = _mix_dark(accent, 0.014)
    surf = _surface(accent)
    return f"""
    QWidget#root {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 {top}, stop:1 {bot});
    }}
    QLabel {{
        color: #c9d1d9; font-family: 'Segoe UI'; font-size: 12px;
        background: transparent;
    }}
    QLabel#title {{ color: #f0f6fc; font-size: 20px; font-weight: 700;
        letter-spacing: -0.2px; }}
    QLabel#sub {{ color: #7d8590; font-size: 12px; }}
    QLabel#cap {{ color: #6e7681; font-size: 10px; font-weight: 600;
        letter-spacing: 1px; }}
    QFrame#pill {{ background: rgba(255,255,255,10);
        border: 1px solid rgba(255,255,255,20); border-radius: 16px; }}
    QLabel#device {{ color: #9aa4af; font-size: 12px; }}
    QLabel#dot {{ background: {accent}; border-radius: 4px; }}
    QFrame#card {{ background: {surf};
        border: 1px solid rgba(255,255,255,18); border-radius: 16px; }}
    QFrame#panel {{ background: rgba(255,255,255,7);
        border: 1px solid rgba(255,255,255,16); border-radius: 16px; }}
    QFrame#tile {{ background: rgba(255,255,255,7);
        border: 1px solid rgba(255,255,255,16); border-radius: 12px; }}
    QLabel#tilecap {{ color: #6e7681; font-size: 10px; font-weight: 600;
        letter-spacing: 1px; }}
    QLabel#tileval {{ color: #e6edf3;
        font-family: 'Cascadia Mono', 'Consolas', monospace;
        font-size: 14px; font-weight: 600; }}
    QPushButton {{ background: rgba(255,255,255,10); color: #c9d1d9;
        border: 1px solid rgba(255,255,255,26); border-radius: 10px;
        padding: 8px 18px; font-family: 'Segoe UI'; font-size: 12px;
        font-weight: 500; }}
    QPushButton:hover {{ background: rgba(255,255,255,20);
        border-color: {accent}; color: #ffffff; }}
    QPushButton:checked {{ background: {accent}; color: #0d1117;
        border-color: {accent}; font-weight: 600; }}
    QPushButton#primary {{ background: {_rgba(accent, 0.14)}; color: {accent};
        border: 1px solid {_rgba(accent, 0.45)}; border-radius: 10px;
        padding: 9px 20px; min-width: 118px; font-weight: 600; }}
    QPushButton#primary:hover {{ background: {_rgba(accent, 0.26)};
        border-color: {accent}; color: {accent}; }}
    QPushButton#primary:checked {{ background: {accent}; color: #0d1117;
        border-color: {accent}; }}
    QFrame#seg {{ background: rgba(255,255,255,10);
        border: 1px solid rgba(255,255,255,22); border-radius: 11px; }}
    QPushButton#segbtn {{ background: transparent; color: #8b949e; border: none;
        border-radius: 8px; padding: 6px 13px; font-weight: 500;
        font-family: 'Segoe UI'; font-size: 12px;
        min-width: 0px; }}
    QPushButton#segbtn:hover {{ background: rgba(255,255,255,14); color: #ffffff; }}
    QPushButton#segbtn:checked {{ background: {accent}; color: #0d1117;
        font-weight: 700; }}
    QComboBox {{ background: rgba(255,255,255,10); color: #c9d1d9;
        border: 1px solid rgba(255,255,255,22); border-radius: 11px;
        padding: 8px 14px; min-width: 90px; font-family: 'Segoe UI';
        font-size: 12px; combobox-popup: 0; }}
    QComboBox:hover {{ border-color: {accent}; background: rgba(255,255,255,18); }}
    QComboBox::drop-down {{ border: none; width: 24px; }}
    QComboBox QFrame {{ background: transparent; border: none; }}
    QComboBox QAbstractItemView {{ background: #161b22; color: #c9d1d9;
        border: 1px solid #30363d; border-radius: 12px; padding: 4px;
        outline: 0; selection-background-color: transparent;
        selection-color: #c9d1d9; }}
    QComboBox QAbstractItemView::item {{ min-height: 28px; padding: 4px 12px;
        border-radius: 8px; margin: 1px 2px; }}
    QComboBox QAbstractItemView::item:hover,
    QComboBox QAbstractItemView::item:selected {{ background: {accent};
        color: #0d1117; font-weight: 600; }}
    QToolTip {{ background: #161b22; color: #c9d1d9;
        border: 1px solid #30363d; border-radius: 6px; padding: 5px 10px;
        font-family: 'Segoe UI'; font-size: 12px; }}
    """


class LogoMark(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 40)
        self._c = QtGui.QColor(THEME_HEX[DEFAULT_THEME])

    def set_accent(self, color):
        self._c = QtGui.QColor(color)
        self._c.setAlpha(255)
        self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        fill = QtGui.QColor(self._c)
        fill.setAlpha(34)
        edge = QtGui.QColor(self._c)
        edge.setAlpha(110)
        p.setPen(QtGui.QPen(edge, 1))
        p.setBrush(fill)
        p.drawRoundedRect(QtCore.QRectF(0.5, 0.5, 39, 39), 12, 12)
        path = QtGui.QPainterPath()
        steps = 24
        for i in range(steps + 1):
            t = i / steps
            x = 9 + t * 22
            env = math.sin(math.pi * t)
            y = 20 - 9.5 * env * math.sin(t * 2 * math.pi * 2)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        pen = QtGui.QPen(self._c, 2.0)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawPath(path)


class ToggleSwitch(QtWidgets.QAbstractButton):
    TW, TH = 36, 20

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
        )
        f = QtGui.QFont("Segoe UI")
        f.setPixelSize(12)
        self.setFont(f)
        self._accent = QtGui.QColor(THEME_HEX[DEFAULT_THEME])
        self._pos = 0.0
        self._hover = False
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(170)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        self.toggled.connect(self._on_toggled)
        self.setText(text)

    def setText(self, text):
        super().setText(text)
        sh = self.sizeHint()
        self.setMinimumWidth(sh.width())
        self.setFixedHeight(sh.height())
        self.updateGeometry()
        self.update()

    def set_accent(self, color):
        self._accent = QtGui.QColor(color)
        self._accent.setAlpha(255)
        self.update()

    def _on_anim(self, v):
        self._pos = float(v)
        self.update()

    def _on_toggled(self, on):
        target = 1.0 if on else 0.0
        self._anim.stop()
        if not self.isVisible():
            self._pos = target
            self.update()
            return
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(target)
        self._anim.start()

    def sizeHint(self):
        text_w = self.fontMetrics().horizontalAdvance(self.text())
        w = self.TW + 12 + text_w + 8
        return QtCore.QSize(w, 26)

    def minimumSizeHint(self):
        return self.sizeHint()

    def enterEvent(self, ev):
        self._hover = True
        self.update()
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self._hover = False
        self.update()
        super().leaveEvent(ev)

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        h = self.height()
        ty = (h - self.TH) / 2.0
        off_fill = QtGui.QColor(255, 255, 255, 28)
        off_edge = QtGui.QColor(255, 255, 255, 70 if self._hover else 40)
        fill = _lerp_color(off_fill, self._accent, self._pos)
        edge = _lerp_color(off_edge, self._accent, self._pos)
        if not self.isEnabled():
            fill.setAlpha(fill.alpha() // 2)
        p.setPen(QtGui.QPen(edge, 1))
        p.setBrush(fill)
        r = self.TH / 2.0 - 0.5
        p.drawRoundedRect(QtCore.QRectF(0.5, ty + 0.5, self.TW - 1, self.TH - 1), r, r)
        d = 12.0
        kx = 4 + self._pos * (self.TW - d - 8)
        ky = ty + (self.TH - d) / 2.0
        knob = _lerp_color(QtGui.QColor("#c9d1d9"), QtGui.QColor("#0d1117"), self._pos)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(knob)
        p.drawEllipse(QtCore.QRectF(kx, ky, d, d))
        p.setPen(QtGui.QColor("#c9d1d9" if self.isEnabled() else "#6e7681"))
        p.drawText(
            QtCore.QRectF(self.TW + 10, 0, self.width() - self.TW - 10, h),
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
            self.text(),
        )


class Segmented(QtWidgets.QFrame):
    currentIndexChanged = QtCore.Signal(int)

    def __init__(self, items, index=0, parent=None):
        super().__init__(parent)
        self.setObjectName("seg")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed
        )
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        self._btns = []
        self._data = []
        self._idx = -1
        for i, (text, data) in enumerate(items):
            b = QtWidgets.QPushButton(text)
            b.setObjectName("segbtn")
            b.setCheckable(True)
            b.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            b.toggled.connect(lambda on, i=i: on and self._pick(i))
            self._group.addButton(b)
            lay.addWidget(b)
            self._btns.append(b)
            self._data.append(data)
            self._adjust_btn(b)
        if self._btns:
            self._btns[max(0, min(index, len(self._btns) - 1))].setChecked(True)
        self.layout().invalidate()
        self.updateGeometry()

    def _adjust_btn(self, b):
        # Принудительно используем ЖИРНЫЙ шрифт (как в checked QSS),
        # чтобы избежать обрезки текста.
        f = QtGui.QFont("Segoe UI")
        f.setPixelSize(12)
        f.setBold(True)
        fm = QtGui.QFontMetrics(f)
        w = fm.horizontalAdvance(b.text()) + 34
        b.setMinimumWidth(w)
        b.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed
        )
        b.updateGeometry()

    def _readjust_all(self):
        for b in self._btns:
            self._adjust_btn(b)
        self.layout().invalidate()
        self.layout().activate()
        self.updateGeometry()

    def showEvent(self, ev):
        super().showEvent(ev)
        # QSS (min-width: 0px) при первой полировке сбрасывает minimumWidth,
        # поэтому пересчитываем ширину кнопок уже после применения стиля.
        self._readjust_all()
        QtCore.QTimer.singleShot(0, self._readjust_all)

    def changeEvent(self, ev):
        super().changeEvent(ev)
        if ev.type() in (QtCore.QEvent.Type.StyleChange, QtCore.QEvent.Type.FontChange):
            QtCore.QTimer.singleShot(0, self._readjust_all)

    def _pick(self, i):
        if i == self._idx:
            return
        self._idx = i
        self.currentIndexChanged.emit(i)

    def count(self):
        return len(self._btns)

    def currentIndex(self):
        return self._idx

    def setCurrentIndex(self, i):
        if 0 <= i < len(self._btns):
            self._btns[i].setChecked(True)

    def itemData(self, i):
        return self._data[i] if 0 <= i < len(self._data) else None

    def setItemText(self, i, text):
        if 0 <= i < len(self._btns):
            self._btns[i].setText(text)
            self._adjust_btn(self._btns[i])
            self.layout().invalidate()
            self.updateGeometry()


class Swatch(QtWidgets.QPushButton):
    def __init__(self, color, parent=None, custom=False):
        super().__init__(parent)
        self._c = QtGui.QColor(color)
        self._hover = False
        self._custom = custom
        self.setCheckable(True)
        self.setFixedSize(26, 26)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def set_color(self, color):
        self._c = QtGui.QColor(color)
        self.update()

    def enterEvent(self, ev):
        self._hover = True
        self.update()
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self._hover = False
        self.update()
        super().leaveEvent(ev)

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        c = QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)
        r = 8.5 if self.isChecked() else (9.5 if self._hover else 9.0)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        if self._custom:
            grad = QtGui.QConicalGradient(c, 90)
            for i, col in enumerate(
                ("#ff0040", "#ffb000", "#00ff8c", "#19c8ff", "#b26bff", "#ff0040")
            ):
                grad.setColorAt(i / 5.0, QtGui.QColor(col))
            p.setBrush(grad)
            p.drawEllipse(c, r, r)
            p.setBrush(self._c)
            p.drawEllipse(c, r * 0.55, r * 0.55)
        else:
            p.setBrush(self._c)
            p.drawEllipse(c, r, r)
        if self.isChecked():
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.8))
            p.drawEllipse(c, 11.2, 11.2)
        elif self._hover:
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 80), 1.4))
            p.drawEllipse(c, 11.2, 11.2)


class IconButton(QtWidgets.QAbstractButton):
    def __init__(self, kind, checkable=False, parent=None):
        super().__init__(parent)
        self._kind = kind
        self._hover = False
        self._accent = QtGui.QColor(THEME_HEX[DEFAULT_THEME])
        self.setCheckable(checkable)
        self.setFixedSize(32, 32)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def set_accent(self, color):
        self._accent = QtGui.QColor(color)
        self._accent.setAlpha(255)
        self.update()

    def enterEvent(self, ev):
        self._hover = True
        self.update()
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self._hover = False
        self.update()
        super().leaveEvent(ev)

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self._hover:
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(255, 255, 255, 28))
            p.drawRoundedRect(QtCore.QRectF(2, 2, 28, 28), 9, 9)
        k = self._kind
        active = self.isChecked() and k in ("pin", "bg")
        col = (
            self._accent
            if active
            else QtGui.QColor("#e6edf3" if self._hover else "#9aa4af")
        )
        pen = QtGui.QPen(col, 1.8)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        if k == "close":
            p.drawLine(QtCore.QPointF(11, 11), QtCore.QPointF(21, 21))
            p.drawLine(QtCore.QPointF(21, 11), QtCore.QPointF(11, 21))
        elif k == "freeze":
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(col)
            if self.isChecked():
                p.drawPolygon(
                    QtGui.QPolygonF(
                        [
                            QtCore.QPointF(12.0, 9.0),
                            QtCore.QPointF(22.0, 16.0),
                            QtCore.QPointF(12.0, 23.0),
                        ]
                    )
                )
            else:
                p.drawRoundedRect(QtCore.QRectF(11.0, 9.0, 3.6, 14.0), 1.3, 1.3)
                p.drawRoundedRect(QtCore.QRectF(17.4, 9.0, 3.6, 14.0), 1.3, 1.3)
        elif k == "bg":
            p.drawRoundedRect(QtCore.QRectF(9, 9, 14, 14), 3, 3)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(col)
            p.drawRect(QtCore.QRectF(9.9, 9.9, 6.1, 6.1))
            p.drawRect(QtCore.QRectF(16.0, 16.0, 6.1, 6.1))
        elif k == "pin":
            if active:
                p.setBrush(col)
            p.drawRoundedRect(QtCore.QRectF(13, 7.5, 6, 8), 1.5, 1.5)
            p.drawLine(QtCore.QPointF(10.5, 17), QtCore.QPointF(21.5, 17))
            p.drawLine(QtCore.QPointF(16, 17), QtCore.QPointF(16, 24.5))


class StatTile(QtWidgets.QFrame):
    def __init__(self, width: int, parent=None):
        super().__init__(parent)
        self.setObjectName("tile")
        self.setFixedWidth(width)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 11)
        lay.setSpacing(3)
        self.cap = QtWidgets.QLabel()
        self.cap.setObjectName("tilecap")
        self.val = QtWidgets.QLabel()
        self.val.setObjectName("tileval")
        self.val.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lay.addWidget(self.cap)
        lay.addWidget(self.val)

    def set(self, cap: str, val_html: str):
        # не трогаем QLabel, если текст не изменился (экономит relayout/repaint)
        if self.cap.text() != cap:
            self.cap.setText(cap)
        if self.val.text() != val_html:
            self.val.setText(val_html)


def _make_icon(accent) -> QtGui.QIcon:
    accent = QtGui.QColor(accent)
    accent.setAlpha(255)
    icon = QtGui.QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.scale(size / 40.0, size / 40.0)
        edge = QtGui.QColor(accent)
        edge.setAlpha(150)
        p.setPen(QtGui.QPen(edge, 1))
        p.setBrush(QtGui.QColor(_mix_dark(accent.name(), 0.10)))
        p.drawRoundedRect(QtCore.QRectF(0.5, 0.5, 39, 39), 12, 12)
        path = QtGui.QPainterPath()
        for i in range(25):
            t = i / 24
            x = 9 + t * 22
            y = 20 - 9.5 * math.sin(math.pi * t) * math.sin(t * 4 * math.pi)
            path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
        pen = QtGui.QPen(accent, 2.4)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.end()
        icon.addPixmap(pm)
    return icon
