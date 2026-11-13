"""Темы, QSS, цветовые хелперы и кастомные виджеты интерфейса (логотип, свитчи, сегменты и т.д.)."""

from typing import Any
from winreg import HKEYType

import backend  # noqa: F401  (переменные окружения и GL-конфиг — до numpy/Qt)
import functools
import math
import sys
from animations import (
    BUTTON_FEEDBACK_DURATION_MS,
    HOVER_EASING_CURVE,
    ValueAnimation,
)
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from settings import CONFIG, DEFAULT_CONFIG

THEMES: list[tuple[str, str]] = [
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


THEME_HEX: dict[str, str] = dict(THEMES)


DEFAULT_THEME = "mint"


# Можно использовать встроенные, пользовательские и системные темы.
# Пользовательский цвет хранится в CONFIG["custom_color"],
# системный цвет запрашивается у ОС при необходимости (system_accent()).
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


def _win_accent_hex() -> None | str:
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
            v: int = int(v) & 0xFFFFFFFF
        except Exception:
            continue
        if order == "abgr":
            r, g, b = v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF
        else:
            r, g, b = (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}"
    return None


def _qt_accent_hex() -> None | str:
    try:
        role: Any | None = getattr(QtGui.QPalette.ColorRole, "Accent", None)
        if role is None:  # Qt < 6.6
            return None
        c: QtGui.QColor = QtGui.QGuiApplication.palette().color(role)
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


def system_accent() -> str | None:
    """Цвет акцента системы (#rrggbb, уже приведённый к читаемому виду)
    или None, если определить не удалось."""
    raw: None | str = _win_accent_hex() if sys.platform == "win32" else None
    if raw is None:
        raw: None | str = _qt_accent_hex()
    return readable_accent(raw) if raw else None


@functools.lru_cache(maxsize=256)
def _mix_dark(accent_hex: str, strength: float = 0.07) -> str:
    c = QtGui.QColor(accent_hex)
    r: int = min(255, int(c.red() * strength) + 6)
    g: int = min(255, int(c.green() * strength) + 7)
    b: int = min(255, int(c.blue() * strength) + 9)
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
            max(0, c.red() - 40),
            max(0, c.green() - 40),
            max(0, c.blue() - 40),
            220,
        )
    v2: int = max(40, int(v * 0.62))
    s2: int = min(255, int(s * 1.05)) if s > 0 else s
    return QtGui.QColor.fromHsv(h, s2, v2, 220)


def make_qss(accent: str) -> str:
    top: str = _mix_dark(accent, 0.040)
    bot: str = _mix_dark(accent, 0.014)
    surf: str = _surface(accent)
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
    QPushButton:checked {{ background: {accent}; color: #0d1117;
        border-color: {accent}; font-weight: 600; }}
    QPushButton#primary {{ background: {_rgba(accent, 0.14)}; color: {accent};
        border: 1px solid {_rgba(accent, 0.45)}; border-radius: 10px;
        padding: 9px 20px; min-width: 118px; font-weight: 600; }}
    QPushButton#primary:checked {{ background: {accent}; color: #0d1117;
        border-color: {accent}; }}
    QFrame#seg {{ background: rgba(0,0,0,28);
        border: 1px solid rgba(255,255,255,18); border-radius: 12px; }}
    QPushButton#segbtn {{ background: transparent; color: #8b949e; border: none;
        border-radius: 8px; padding: 6px 13px; font-weight: 500;
        font-family: 'Segoe UI'; font-size: 12px;
        min-width: 0px; }}
    QPushButton#segbtn:checked {{ background: {accent}; color: #0d1117;
        font-weight: 700; }}
    QComboBox {{ background: rgba(255,255,255,10); color: #c9d1d9;
        border: 1px solid rgba(255,255,255,22); border-radius: 11px;
        padding: 8px 14px; min-width: 90px; font-family: 'Segoe UI';
        font-size: 12px; combobox-popup: 0; }}
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
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(40, 40)
        self._c = QtGui.QColor(THEME_HEX[DEFAULT_THEME])

    def set_accent(self, color) -> None:
        self._c = QtGui.QColor(color)
        self._c.setAlpha(255)
        self.update()

    def paintEvent(self, ev) -> None:
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
            t: float = i / steps
            x: float = 9 + t * 22
            env: float = math.sin(math.pi * t)
            y: float = 20 - 9.5 * env * math.sin(t * 2 * math.pi * 2)
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

    def __init__(self, text="", parent=None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        f = QtGui.QFont("Segoe UI")
        f.setPixelSize(12)
        self.setFont(f)
        self._accent = QtGui.QColor(THEME_HEX[DEFAULT_THEME])
        self._pos = 0.0
        self._hover_pos = 0.0
        self._anim = ValueAnimation(self, 170, self._on_anim)
        self._hover_anim = ValueAnimation(
            self,
            BUTTON_FEEDBACK_DURATION_MS,
            self._on_hover_anim,
            HOVER_EASING_CURVE,
        )
        self.toggled.connect(self._on_toggled)
        self.setText(text)

    def setText(self, text) -> None:
        super().setText(text)
        sh: QtCore.QSize = self.sizeHint()
        self.setMinimumWidth(sh.width())
        self.setFixedHeight(sh.height())
        self.updateGeometry()
        self.update()

    def set_accent(self, color) -> None:
        self._accent = QtGui.QColor(color)
        self._accent.setAlpha(255)
        self.update()

    def _on_anim(self, v) -> None:
        self._pos = float(v)
        self.update()

    def _on_hover_anim(self, v) -> None:
        self._hover_pos = float(v)
        self.update()

    def _set_hover(self, hovered: bool) -> None:
        self._hover_anim.animate(self._hover_pos, 1.0 if hovered else 0.0)

    def _on_toggled(self, on) -> None:
        target: float = 1.0 if on else 0.0
        self._anim.stop()
        if not self.isVisible():
            self._pos: float = target
            self.update()
            return
        self._anim.animate(self._pos, target)

    def sizeHint(self) -> QtCore.QSize:
        text_w: int = self.fontMetrics().horizontalAdvance(self.text())
        w: int = self.TW + 12 + text_w + 8
        return QtCore.QSize(w, 26)

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()

    def enterEvent(self, ev) -> None:
        self._set_hover(True)
        super().enterEvent(ev)

    def leaveEvent(self, ev) -> None:
        self._set_hover(False)
        super().leaveEvent(ev)

    def paintEvent(self, ev) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        h: int = self.height()
        ty: float = (h - self.TH) / 2.0
        off_fill = QtGui.QColor(255, 255, 255, int(28 + 22 * self._hover_pos))
        off_edge = QtGui.QColor(255, 255, 255, int(40 + 22 * self._hover_pos))
        fill: QtGui.QColor = _lerp_color(off_fill, self._accent, self._pos)
        edge: QtGui.QColor = _lerp_color(off_edge, self._accent, self._pos)
        if not self.isEnabled():
            fill.setAlpha(fill.alpha() // 2)
        p.setPen(QtGui.QPen(edge, 1))
        p.setBrush(fill)
        r: float = self.TH / 2.0 - 0.5
        p.drawRoundedRect(QtCore.QRectF(0.5, ty + 0.5, self.TW - 1, self.TH - 1), r, r)
        d = 12.0
        kx: float = 4 + self._pos * (self.TW - d - 8)
        ky: float = ty + (self.TH - d) / 2.0
        knob: QtGui.QColor = _lerp_color(
            QtGui.QColor("#c9d1d9"), QtGui.QColor("#0d1117"), self._pos
        )
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(knob)
        p.drawEllipse(QtCore.QRectF(kx, ky, d, d))
        p.setPen(QtGui.QColor("#c9d1d9" if self.isEnabled() else "#6e7681"))
        p.drawText(
            QtCore.QRectF(self.TW + 10, 0, self.width() - self.TW - 10, h),
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
            self.text(),
        )


class _SegText(QtWidgets.QWidget):
    """Прозрачный слой поверх кнопок: рисует подписи сегментов с плавной
    сменой цвета (обычный → ховер → выбран)."""

    def __init__(self, seg: "Segmented") -> None:
        super().__init__(seg)
        self._seg = seg
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setStyleSheet("background: transparent; border: none;")
        self.setGeometry(seg.rect())

    def paintEvent(self, ev) -> None:
        self._seg._paint_text(self)


def _mix_color(a: QtGui.QColor, b: QtGui.QColor, t: float) -> QtGui.QColor:
    t = min(1.0, max(0.0, t))
    return QtGui.QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
    )


class Segmented(QtWidgets.QFrame):
    _TXT_BASE = QtGui.QColor("#8b949e")
    _TXT_HOVER = QtGui.QColor("#f0f6fc")
    _TXT_SELECTED = QtGui.QColor("#0d1117")
    _TXT_DISABLED = QtGui.QColor("#6e7681")
    # Текст кнопок прозрачный в QSS — его рисует слой _SegText.
    _BTN_SHEET = (
        "QPushButton#segbtn { color: transparent; }"
        "QPushButton#segbtn:hover { color: transparent; }"
        "QPushButton#segbtn:checked { color: transparent; }"
    )
    currentIndexChanged: QtCore.pyqtSignal = QtCore.Signal(int)

    def __init__(self, items, index=0, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("seg")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        self._btns = []
        self._data = []
        self._idx = -1
        self._animated_selection = False
        self._selection_color = QtGui.QColor(THEME_HEX[DEFAULT_THEME])
        self._selection_rect = QtCore.QRectF()
        self._selection_anim = ValueAnimation(self, 190, self._on_selection_rect)
        # Мягкая «подсветка-пилюля» под курсором: плавно переезжает между кнопками.
        self._hover_rect = QtCore.QRectF()
        self._hover_alpha = 0.0
        self._hover_idx = -1
        self._txt_hover: list[float] = []
        self._txt_sel: list[float] = []
        self._txt_clock = QtCore.QElapsedTimer()
        self._txt_timer = QtCore.QTimer(self)
        self._txt_timer.setInterval(15)
        self._txt_timer.timeout.connect(self._tick_text)
        self._hover_move = ValueAnimation(self, 170, self._on_hover_rect)
        self._hover_fade = ValueAnimation(
            self, 150, self._on_hover_alpha, QtCore.QEasingCurve.Type.InOutSine
        )
        for i, (text, data) in enumerate(items):
            b = QtWidgets.QPushButton(text)
            b.setObjectName("segbtn")
            b.setCheckable(True)
            b.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            b.toggled.connect(lambda on, i=i: on and self._pick(i))
            b.installEventFilter(self)
            b.setStyleSheet(self._BTN_SHEET)
            self._txt_hover.append(0.0)
            self._txt_sel.append(0.0)
            self._group.addButton(b)
            lay.addWidget(b)
            self._btns.append(b)
            self._data.append(data)
            self._adjust_btn(b)
        self._text_layer = _SegText(self)
        if self._btns:
            self._btns[max(0, min(index, len(self._btns) - 1))].setChecked(True)
        self.layout().invalidate()
        self.updateGeometry()

    def _adjust_btn(self, b) -> None:
        # Принудительно используем ЖИРНЫЙ шрифт (как в checked QSS),
        # чтобы избежать обрезки текста.
        f = QtGui.QFont("Segoe UI")
        f.setPixelSize(12)
        f.setBold(True)
        fm = QtGui.QFontMetrics(f)
        w: int = fm.horizontalAdvance(b.text()) + 34
        b.setMinimumWidth(w)
        b.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        b.updateGeometry()

    def _readjust_all(self) -> None:
        for b in self._btns:
            self._adjust_btn(b)
        self.layout().invalidate()
        self.layout().activate()
        self.updateGeometry()

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        # Правило QSS min-width: 0px при первом обновлении сбрасывает minimumWidth,
        # поэтому пересчитываем ширину кнопок уже после применения стиля.
        self._readjust_all()
        self._text_layer.setGeometry(self.rect())
        self._text_layer.raise_()
        QtCore.QTimer.singleShot(0, self._sync_selection_geometry)
        QtCore.QTimer.singleShot(0, self._readjust_all)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        layer = getattr(self, "_text_layer", None)
        if layer is not None:
            layer.setGeometry(self.rect())
            layer.raise_()

    def changeEvent(self, ev) -> None:
        super().changeEvent(ev)
        if ev.type() in (
            QtCore.QEvent.Type.StyleChange,
            QtCore.QEvent.Type.FontChange,
        ):
            QtCore.QTimer.singleShot(0, self._readjust_all)

    def _pick(self, i) -> None:
        if i == self._idx:
            return
        self._idx = i
        self._fade_hover(0.0)
        if self._animated_selection:
            self._move_selection(i)
        if not self._animated_selection or not self.isVisible():
            for k in range(len(self._txt_sel)):
                self._txt_sel[k] = 1.0 if k == i else 0.0
        self._hover_idx = -1
        self._kick_text()
        QtCore.QTimer.singleShot(0, self._refresh_hover)
        self.currentIndexChanged.emit(i)

    def set_animated_selection(
        self, enabled: bool, accent: QtGui.QColor | None = None
    ) -> None:
        self._animated_selection = enabled
        if accent is not None:
            self._selection_color = QtGui.QColor(accent)
        for button in self._btns:
            if enabled:
                button.setStyleSheet(
                    "QPushButton#segbtn { background: transparent; border: none; "
                    "color: transparent; }"
                    "QPushButton#segbtn:hover { background: transparent; "
                    "color: transparent; }"
                    "QPushButton#segbtn:checked { background: transparent; "
                    "color: transparent; font-weight: 700; }"
                )
            else:
                button.setStyleSheet(self._BTN_SHEET)
        if enabled and self._idx >= 0:
            QtCore.QTimer.singleShot(0, self._sync_selection_geometry)
        self.update()

    def set_accent(self, accent: QtGui.QColor) -> None:
        self._selection_color = QtGui.QColor(accent)
        self.update()

    def _button_rect(self, index: int) -> QtCore.QRectF:
        if not 0 <= index < len(self._btns):
            return QtCore.QRectF()
        button = self._btns[index]
        top_left = button.mapTo(self, QtCore.QPoint(0, 0))
        rect = QtCore.QRectF(QtCore.QRect(top_left, button.size()))
        return rect.adjusted(1, 1, -1, -1)

    def _sync_selection_geometry(self) -> None:
        self._readjust_all()
        self._hover_move.stop()
        self._hover_fade.stop()
        self._hover_alpha = 0.0
        self._hover_idx = -1
        QtCore.QTimer.singleShot(0, self._refresh_hover)
        if self._animated_selection and self._idx >= 0:
            self._move_selection(self._idx, False)

    def _on_selection_rect(self, rect: QtCore.QRectF) -> None:
        self._selection_rect = QtCore.QRectF(rect)
        self.update()

    def _move_selection(self, index: int, animate: bool = True) -> None:
        target = self._button_rect(index)
        if target.isEmpty():
            return
        if not animate or self._selection_rect.isEmpty() or not self.isVisible():
            self._selection_rect = target
            self.update()
            return
        self._selection_anim.animate(self._selection_rect, target)

    # --- ховер ---
    def eventFilter(self, obj, ev) -> bool:
        kind = ev.type()
        if kind in (QtCore.QEvent.Type.Enter, QtCore.QEvent.Type.Leave):
            if obj in self._btns:
                QtCore.QTimer.singleShot(0, self._refresh_hover)
        return super().eventFilter(obj, ev)

    def enterEvent(self, ev) -> None:
        super().enterEvent(ev)
        QtCore.QTimer.singleShot(0, self._refresh_hover)

    def leaveEvent(self, ev) -> None:
        super().leaveEvent(ev)
        QtCore.QTimer.singleShot(0, self._refresh_hover)

    def _index_under_cursor(self) -> int:
        """Индекс кнопки под курсором; в щелях и полях рамки — ближайшая,
        чтобы подсветка не мигала. Вне переключателя — -1."""
        if not self.isVisible() or not self.isEnabled():
            return -1
        gpos = QtGui.QCursor.pos()
        w = QtWidgets.QApplication.widgetAt(gpos)
        if w is None or not (w is self or self.isAncestorOf(w)):
            return -1
        p = self.mapFromGlobal(gpos)
        if not self.rect().contains(p):
            return -1
        best, best_d = -1, 1e9
        for i, b in enumerate(self._btns):
            if not b.isEnabled():
                continue
            left = b.mapTo(self, QtCore.QPoint(0, 0)).x()
            right = left + b.width()
            d = 0 if left <= p.x() <= right else min(abs(p.x() - left), abs(p.x() - right))
            if d < best_d:
                best, best_d = i, d
        return best

    def _refresh_hover(self) -> None:
        idx = self._index_under_cursor()
        if idx == self._hover_idx:
            return
        self._hover_idx = idx
        if idx < 0:
            self._fade_hover(0.0)
        else:
            self._hover_to(idx)
        self._kick_text()

    # --- плавный текст ---
    def _kick_text(self) -> None:
        if not self._txt_timer.isActive():
            self._txt_clock.start()
            self._txt_timer.start()
        self._text_layer.update()

    def _tick_text(self) -> None:
        dt = min(0.05, self._txt_clock.restart() / 1000.0)
        k = 1.0 - math.exp(-dt / 0.055)
        busy = False
        for i in range(len(self._btns)):
            targets = (
                (self._txt_hover, 1.0 if (i == self._hover_idx and i != self._idx) else 0.0),
                (self._txt_sel, 1.0 if i == self._idx else 0.0),
            )
            for values, target in targets:
                v = values[i]
                if abs(target - v) > 0.01:
                    values[i] = v + (target - v) * k
                    busy = True
                else:
                    values[i] = target
        self._text_layer.update()
        if not busy:
            self._txt_timer.stop()

    def _paint_text(self, layer: QtWidgets.QWidget) -> None:
        painter = QtGui.QPainter(layer)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        for i, b in enumerate(self._btns):
            if i >= len(self._txt_sel) or not b.isVisible():
                continue
            rect = QtCore.QRectF(QtCore.QRect(b.mapTo(self, QtCore.QPoint(0, 0)), b.size()))
            base = self._TXT_BASE if b.isEnabled() else self._TXT_DISABLED
            color = _mix_color(base, self._TXT_HOVER, self._txt_hover[i])
            color = _mix_color(color, self._TXT_SELECTED, self._txt_sel[i])
            font = QtGui.QFont("Segoe UI")
            font.setPixelSize(12)
            font.setWeight(
                QtGui.QFont.Weight.Bold if i == self._idx else QtGui.QFont.Weight.Medium
            )
            painter.setFont(font)
            painter.setPen(color)
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, b.text())
        painter.end()

    def _hover_to(self, index: int) -> None:
        target = self._button_rect(index)
        if target.isEmpty() or index == self._idx:
            # Выбранная кнопка уже подсвечена — отдельный ховер ей не нужен.
            self._fade_hover(0.0)
            return
        if self._hover_alpha < 0.02 or self._hover_rect.isEmpty():
            self._hover_move.stop()
            self._hover_rect = target
            self.update()
        else:
            self._hover_move.animate(self._hover_rect, target)
        self._fade_hover(1.0)

    def _fade_hover(self, target: float) -> None:
        if abs(self._hover_alpha - target) < 0.001:
            return
        self._hover_fade.animate(self._hover_alpha, target)

    def _on_hover_rect(self, rect: QtCore.QRectF) -> None:
        self._hover_rect = QtCore.QRectF(rect)
        self.update()

    def _on_hover_alpha(self, value) -> None:
        self._hover_alpha = float(value)
        self.update()

    def paintEvent(self, ev) -> None:
        super().paintEvent(ev)
        show_hover = self._hover_alpha > 0.01 and not self._hover_rect.isEmpty()
        show_selection = self._animated_selection and not self._selection_rect.isEmpty()
        if not (show_hover or show_selection):
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if show_hover:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(255, 255, 255, int(24 * self._hover_alpha)))
            painter.drawRoundedRect(self._hover_rect, 8, 8)
        if show_selection:
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 38), 1))
            painter.setBrush(self._selection_color)
            painter.drawRoundedRect(self._selection_rect, 8, 8)

    def count(self) -> int:
        return len(self._btns)

    def currentIndex(self):
        return self._idx

    def setCurrentIndex(self, i) -> None:
        if 0 <= i < len(self._btns):
            self._btns[i].setChecked(True)

    def itemData(self, i):
        return self._data[i] if 0 <= i < len(self._data) else None

    def setItemText(self, i, text) -> None:
        if 0 <= i < len(self._btns):
            self._btns[i].setText(text)
            self._adjust_btn(self._btns[i])
            self._text_layer.update()
            self.layout().invalidate()
            self.updateGeometry()


class Swatch(QtWidgets.QPushButton):
    def __init__(self, color, parent=None, custom=False) -> None:
        super().__init__(parent)
        self._c = QtGui.QColor(color)
        self._hover_progress = 0.0
        self._custom: bool = custom
        self._selection_progress = 1.0
        self._hover_anim = ValueAnimation(
            self,
            BUTTON_FEEDBACK_DURATION_MS,
            self._on_hover_progress,
            HOVER_EASING_CURVE,
        )
        self._selection_anim = ValueAnimation(
            self, 140, self._on_selection_progress
        )
        self.toggled.connect(self._on_checked_changed)
        self.setCheckable(True)
        self.setFixedSize(26, 26)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def _on_selection_progress(self, progress) -> None:
        self._selection_progress = float(progress)
        self.update()

    def _on_hover_progress(self, progress) -> None:
        self._hover_progress = float(progress)
        self.update()

    def _animate_hover(self, hovered: bool) -> None:
        self._hover_anim.animate(
            self._hover_progress, 1.0 if hovered else 0.0
        )

    def _on_checked_changed(self, checked: bool) -> None:
        self._selection_anim.stop()
        if checked:
            self._selection_progress = 0.0
            self._selection_anim.animate(0.0, 1.0)
        else:
            self._selection_progress = 1.0
        self.update()

    def set_color(self, color) -> None:
        self._c = QtGui.QColor(color)
        self.update()

    def enterEvent(self, ev) -> None:
        self._animate_hover(True)
        super().enterEvent(ev)

    def leaveEvent(self, ev) -> None:
        self._animate_hover(False)
        super().leaveEvent(ev)

    def paintEvent(self, ev) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        c = QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)
        r: float = 8.5 if self.isChecked() else 9.0 + 0.5 * self._hover_progress
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        if self._custom:
            grad = QtGui.QConicalGradient(c, 90)
            for i, col in enumerate(
                (
                    "#ff0040",
                    "#ffb000",
                    "#00ff8c",
                    "#19c8ff",
                    "#b26bff",
                    "#ff0040",
                )
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
            progress = self._selection_progress
            ring = QtGui.QColor("#ffffff")
            ring.setAlpha(int(120 + 135 * progress))
            p.setPen(QtGui.QPen(ring, 1.8))
            radius = 9.8 + 1.4 * progress
            p.drawEllipse(c, radius, radius)
        elif self._hover_progress > 0.0:
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            hover_ring = QtGui.QColor(255, 255, 255, int(80 * self._hover_progress))
            p.setPen(QtGui.QPen(hover_ring, 1.4))
            p.drawEllipse(c, 11.2, 11.2)


class IconButton(QtWidgets.QAbstractButton):
    def __init__(self, kind, checkable=False, parent=None) -> None:
        super().__init__(parent)
        self._kind: Any = kind
        self._hover_progress = 0.0
        self._press_progress = 0.0
        self._accent = QtGui.QColor(THEME_HEX[DEFAULT_THEME])
        self._hover_anim = ValueAnimation(
            self,
            BUTTON_FEEDBACK_DURATION_MS,
            self._on_hover_progress,
            HOVER_EASING_CURVE,
        )
        self._press_anim = ValueAnimation(self, 90, self._on_press_progress)
        self.setCheckable(checkable)
        self.setFixedSize(32, 32)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def set_accent(self, color) -> None:
        self._accent = QtGui.QColor(color)
        self._accent.setAlpha(255)
        self.update()

    def _on_hover_progress(self, progress) -> None:
        self._hover_progress = float(progress)
        self.update()

    def _on_press_progress(self, progress) -> None:
        self._press_progress = float(progress)
        self.update()

    def _animate_progress(
        self, anim: ValueAnimation, current: float, target: float
    ) -> None:
        anim.animate(current, target)

    def enterEvent(self, ev) -> None:
        self._animate_progress(self._hover_anim, self._hover_progress, 1.0)
        super().enterEvent(ev)

    def leaveEvent(self, ev) -> None:
        self._animate_progress(self._hover_anim, self._hover_progress, 0.0)
        super().leaveEvent(ev)

    def mousePressEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self._animate_progress(self._press_anim, self._press_progress, 1.0)
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        super().mouseReleaseEvent(ev)
        self._animate_progress(self._press_anim, self._press_progress, 0.0)

    def paintEvent(self, ev) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self._hover_progress > 0.0:
            hover_fill = QtGui.QColor(self._accent)
            hover_fill.setAlpha(int(22 * self._hover_progress))
            hover_edge = QtGui.QColor(self._accent)
            hover_edge.setAlpha(int(110 * self._hover_progress))
            p.setBrush(hover_fill)
            p.setPen(QtGui.QPen(hover_edge, 1.0))
            p.drawRoundedRect(QtCore.QRectF(2, 2, 28, 28), 9, 9)
        if self._press_progress > 0.0:
            p.translate(self.width() / 2.0, self.height() / 2.0)
            scale = 1.0 - 0.06 * self._press_progress
            p.scale(scale, scale)
            p.translate(-self.width() / 2.0, -self.height() / 2.0)
        k: Any = self._kind
        active: bool = self.isChecked() and k in ("pin", "bg")
        col: QtGui.QColor = (
            self._accent
            if active
            else _lerp_color(
                QtGui.QColor("#9aa4af"),
                QtGui.QColor("#e6edf3"),
                self._hover_progress,
            )
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
    def __init__(self, width: int, parent=None) -> None:
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

    def set(self, cap: str, val_html: str) -> None:
        # Не обновляем QLabel без изменения текста, чтобы не пересчитывать
        # компоновку и не перерисовывать виджет.
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
            t: float = i / 24
            x: float = 9 + t * 22
            y: float = 20 - 9.5 * math.sin(math.pi * t) * math.sin(t * 4 * math.pi)
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