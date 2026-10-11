"""
PC Waveform — осциллограмма звука с ПК (WASAPI loopback) by riverya4life (github.com/riverya4life/pc-waveform)
"""

import sys
import threading
import time

import numpy as np
import soundcard as sc
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# ---------------- настройки ----------------
SAMPLERATE = 48000
CHUNK = 256
UPDATE_MS = 1
MAX_N = 8192
TRIGGER_MAX_AGE = 1536
SAMPLE_OPTIONS = [1024, 2048, 4096, 8192]
GAIN_OPTIONS = [1, 2, 4, 8]
TICKS = {1024: 250, 2048: 500, 4096: 500, 8192: 1000}

THEMES = {
    "Мята":    "#00ff8c",
    "Циан":    "#19c8ff",
    "Лазурь":  "#4d8dff",
    "Индиго":  "#7c6cff",
    "Фиолет":  "#b26bff",
    "Розовый": "#ff50a0",
    "Малина":  "#ff2e63",
    "Коралл":  "#ff6b4a",
    "Янтарь":  "#ffb000",
    "Лайм":    "#b6ff2e",
    "Бирюза":  "#2ee6d6",
    "Белый":   "#ebebeb",
}
# -------------------------------------------

# OpenGL + отключение антиалиасинга — критично для FPS
pg.setConfigOptions(useOpenGL=False, antialias=False)

buf = np.zeros(MAX_N * 2, dtype=np.float32)
lock = threading.Lock()
state = {"device": "поиск устройства…", "error": None}


def capture():
    try:
        speaker = sc.default_speaker()
        state["device"] = str(speaker.name)
        mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        with mic.recorder(samplerate=SAMPLERATE, channels=2, blocksize=CHUNK) as rec:
            while True:
                data = rec.record(numframes=CHUNK)
                mono = data.mean(axis=1).astype(np.float32, copy=False)
                n = len(mono)
                with lock:
                    buf[:-n] = buf[n:]
                    buf[-n:] = mono
    except Exception as e:
        state["error"] = str(e)


def _mix_dark(accent_hex: str, strength: float = 0.07) -> str:
    c = QtGui.QColor(accent_hex)
    r = min(255, int(c.red() * strength) + 6)
    g = min(255, int(c.green() * strength) + 7)
    b = min(255, int(c.blue() * strength) + 9)
    return f"#{r:02x}{g:02x}{b:02x}"


def make_qss(accent: str) -> str:
    bg = _mix_dark(accent, 0.065)
    card_bg = _mix_dark(accent, 0.045)
    return f"""
    QWidget#root {{
        background: {bg};
    }}
    QLabel {{
        color: #c9d1d9;
        font-family: 'Segoe UI';
        font-size: 12px;
    }}
    QLabel#title {{
        color: #f0f6fc;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.3px;
    }}
    QLabel#sub, QLabel#device, QLabel#cap {{
        color: #8b949e;
        font-size: 12px;
    }}
    QLabel#stat {{
        color: #e6edf3;
        font-family: 'Cascadia Mono', 'Consolas', monospace;
        font-size: 12px;
        background: {card_bg};
        border: 1px solid #2a3038;
        border-radius: 10px;
        padding: 8px 14px;
    }}
    QFrame#card {{
        background: {card_bg};
        border: 1px solid #21262d;
        border-radius: 14px;
    }}
    QPushButton {{
        background: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 9px;
        padding: 8px 18px;
        font-family: 'Segoe UI';
        font-size: 12px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background: #1c2128;
        border-color: {accent};
        color: #ffffff;
    }}
    QPushButton:pressed {{
        background: #0d1117;
    }}
    QPushButton:checked {{
        background: {accent};
        color: #0d1117;
        border-color: {accent};
        font-weight: 600;
    }}
    QCheckBox {{
        color: #c9d1d9;
        spacing: 9px;
        font-family: 'Segoe UI';
        font-size: 12px;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 5px;
        border: 1.5px solid #30363d;
        background: #161b22;
    }}
    QCheckBox::indicator:hover {{
        border-color: {accent};
    }}
    QCheckBox::indicator:checked {{
        background: {accent};
        border-color: {accent};
    }}
    QComboBox {{
        background: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 9px;
        padding: 7px 14px;
        min-width: 90px;
        font-family: 'Segoe UI';
        font-size: 12px;
        combobox-popup: 0;
    }}
    QComboBox:hover {{
        border-color: {accent};
        background: #1c2128;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox QFrame {{
        background: transparent;
        border: none;
    }}
    QComboBox QAbstractItemView {{
        background: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 11px;
        padding: 4px;
        outline: 0;
        selection-background-color: transparent;
        selection-color: #c9d1d9;
    }}
    QComboBox QAbstractItemView::item {{
        min-height: 28px;
        padding: 4px 12px;
        border-radius: 7px;
        margin: 1px 2px;
    }}
    QComboBox QAbstractItemView::item:hover,
    QComboBox QAbstractItemView::item:selected {{
        background: {accent};
        color: #0d1117;
        font-weight: 600;
    }}
    QToolTip {{
        background: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 5px 10px;
        font-family: 'Segoe UI';
        font-size: 12px;
    }}
    """


def swatch_qss(color):
    return f"""
    QPushButton {{
        background: {color};
        border: 2px solid #090c10;
        border-radius: 12px;
        padding: 0;
    }}
    QPushButton:hover {{
        border: 2px solid #8b949e;
    }}
    QPushButton:checked {{
        border: 2px solid #ffffff;
    }}
    """


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
            return QtCore.Qt.Orientations()

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
        x, y, line_h = rect.x(), rect.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width()
            if next_x > rect.right() + 1 and line_h > 0:
                x = rect.x()
                y += line_h + self._v
                next_x = x + hint.width()
                line_h = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x = next_x + self._h
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y()


def dark_titlebar(widget):
    if sys.platform != "win32":
        return
    try:
        import ctypes
        hwnd = int(widget.winId())
        val = ctypes.c_int(1)
        for attr in (20, 19):
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(val), ctypes.sizeof(val))
    except Exception:
        pass


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("root")
        self.setWindowTitle("PC Waveform — осциллограмма системного звука (github: riverya4life)")
        self.resize(980, 580)
        self.setMinimumSize(500, 500)

        self.n = 4096
        self.frozen = False
        self.last = np.zeros(self.n, dtype=np.float32)
        self.lim = 1.0
        self.prev_tpl = None
        self._kernel = np.ones(8, dtype=np.float32) / 8
        self._x = np.arange(self.n, dtype=np.float32)
        self.frames = 0
        self.t0 = time.perf_counter()
        self.accent = THEMES["Мята"]
        self._current_color = QtGui.QColor(self.accent)
        self._theme_anim = None

        self._build_ui()
        self._apply_theme("Мята", animate=False)
        self.set_n(4096)
        self.set_gain(1)

        self.setWindowOpacity(0.0)
        self._fade = QtCore.QPropertyAnimation(self, b"windowOpacity")
        self._fade.setDuration(280)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._fade.start()

        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(UPDATE_MS)

        self.stats_timer = QtCore.QTimer(self)
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(100)

        shortcut_cls = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut
        shortcut_cls(QtGui.QKeySequence("Space"), self, activated=self.freeze_btn.click)

    # ---------- Анимация темы ----------
    def _animate_theme(self, target_hex: str):
        target = QtGui.QColor(target_hex)
        if self._theme_anim and self._theme_anim.state() == QtCore.QAbstractAnimation.State.Running:
            self._theme_anim.stop()

        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(420)
        anim.setStartValue(self._current_color)
        anim.setEndValue(target)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        anim.valueChanged.connect(self._on_theme_color)
        anim.finished.connect(lambda: setattr(self, "_current_color", target))
        anim.start()
        self._theme_anim = anim

    def _on_theme_color(self, color: QtGui.QColor):
        if not isinstance(color, QtGui.QColor):
            return
        self.accent = color.name()
        self.setStyleSheet(make_qss(self.accent))
        r, g, b = color.red(), color.green(), color.blue()
        self.curve.setPen(pg.mkPen((r, g, b), width=1.6))
        self.glow_curve.setPen(pg.mkPen((r, g, b, 48), width=7))

    def _apply_theme(self, name: str, animate: bool = True):
        if name in self.theme_btns:
            self.theme_btns[name].setChecked(True)
        target = THEMES[name]
        if animate:
            self._animate_theme(target)
        else:
            self.accent = target
            self._current_color = QtGui.QColor(target)
            self.setStyleSheet(make_qss(self.accent))
            c = self._current_color
            self.curve.setPen(pg.mkPen((c.red(), c.green(), c.blue()), width=1.6))
            self.glow_curve.setPen(pg.mkPen((c.red(), c.green(), c.blue(), 48), width=7))

    # ---------- UI ----------
    def _build_ui(self):
        no_focus = QtCore.Qt.FocusPolicy.NoFocus
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)

        header = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        title_box.setSpacing(2)
        title = QtWidgets.QLabel("PC Waveform")
        title.setObjectName("title")
        sub = QtWidgets.QLabel("осциллограмма системного звука")
        sub.setObjectName("sub")
        title_box.addWidget(title)
        title_box.addWidget(sub)
        header.addLayout(title_box)

        self.device_lbl = QtWidgets.QLabel("")
        self.device_lbl.setObjectName("device")
        self.device_lbl.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.device_lbl.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred)
        self.device_err = False
        header.addWidget(self.device_lbl, 1)
        root.addLayout(header)

        card = QtWidgets.QFrame()
        card.setObjectName("card")
        shadow = QtWidgets.QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 5)
        shadow.setColor(QtGui.QColor(0, 0, 0, 70))
        card.setGraphicsEffect(shadow)

        card_l = QtWidgets.QVBoxLayout(card)
        card_l.setContentsMargins(10, 12, 14, 8)

        self.bottom = SpacedAxis("bottom")
        self.left = SpacedAxis("left")
        self.plot = pg.PlotWidget(axisItems={"bottom": self.bottom, "left": self.left})
        self.plot.setBackground(None)
        self.plot.setStyleSheet("background: transparent;")
        self.plot.setMouseEnabled(x=False, y=False)

        pi = self.plot.getPlotItem()
        pi.setMenuEnabled(False)
        pi.hideButtons()
        pi.showGrid(x=True, y=True, alpha=0.09)

        for ax in (self.bottom, self.left):
            ax.setPen(pg.mkPen("#2a3038"))
            ax.setTextPen(pg.mkPen("#8b949e"))
            ax.setTickFont(QtGui.QFont("Segoe UI", 8))
        self.left.setWidth(48)

        self.zero_line = pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen((255, 255, 255, 28), width=1))
        self.plot.addItem(self.zero_line)

        limit_pen = pg.mkPen((255, 255, 255, 45), width=1,
                             style=QtCore.Qt.PenStyle.DashLine)
        self.lim_hi = pg.InfiniteLine(pos=1, angle=0, pen=limit_pen)
        self.lim_lo = pg.InfiniteLine(pos=-1, angle=0, pen=limit_pen)
        self.plot.addItem(self.lim_hi)
        self.plot.addItem(self.lim_lo)

        self.glow_curve = self.plot.plot(pen=None)
        self.curve = self.plot.plot(pen=None)
        self.glow_curve.setVisible(False)

        card_l.addWidget(self.plot)
        root.addWidget(card, 1)

        ctl = FlowLayout(hspacing=18, vspacing=12)

        self.freeze_btn = QtWidgets.QPushButton("Заморозить")
        self.freeze_btn.setCheckable(True)
        self.freeze_btn.setFocusPolicy(no_focus)
        self.freeze_btn.toggled.connect(self.on_freeze)
        ctl.addWidget(self.freeze_btn)

        self.trigger = QtWidgets.QCheckBox("Стабилизация")
        self.trigger.setChecked(True)
        self.trigger.setFocusPolicy(no_focus)
        self.trigger.setToolTip("Триггер по фронту с гистерезисом + захват фазы")
        self.trigger.toggled.connect(lambda _: setattr(self, "prev_tpl", None))
        ctl.addWidget(self.trigger)

        self.glow = QtWidgets.QCheckBox("Свечение")
        self.glow.setChecked(False)
        self.glow.setFocusPolicy(no_focus)
        self.glow.toggled.connect(lambda on: self.glow_curve.setVisible(on))
        ctl.addWidget(self.glow)

        self.n_combo = self._combo(
            "Окно", [f"{v} смп" for v in SAMPLE_OPTIONS],
            SAMPLE_OPTIONS.index(4096), ctl)
        self.n_combo.currentIndexChanged.connect(
            lambda i: self.set_n(SAMPLE_OPTIONS[i]))

        self.gain_combo = self._combo(
            "Усиление", [f"×{g}" for g in GAIN_OPTIONS], 0, ctl)
        self.gain_combo.currentIndexChanged.connect(
            lambda i: self.set_gain(GAIN_OPTIONS[i]))

        self._theme_picker(ctl)
        root.addLayout(ctl)

        stats = FlowLayout(hspacing=12, vspacing=10)
        self.peak_lbl = QtWidgets.QLabel()
        self.rms_lbl = QtWidgets.QLabel()
        self.info_lbl = QtWidgets.QLabel()
        self.fps_lbl = QtWidgets.QLabel()

        for lbl, w in ((self.peak_lbl, 200), (self.rms_lbl, 155),
                       (self.info_lbl, 235), (self.fps_lbl, 105)):
            lbl.setObjectName("stat")
            lbl.setMinimumWidth(w)
            stats.addWidget(lbl)

        root.addLayout(stats)

    def _theme_picker(self, layout):
        holder = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(7)
        cap = QtWidgets.QLabel("Тема")
        cap.setObjectName("cap")
        row.addWidget(cap)
        row.addSpacing(4)

        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        self.theme_btns = {}
        for name, color in THEMES.items():
            b = QtWidgets.QPushButton()
            b.setCheckable(True)
            b.setFixedSize(24, 24)
            b.setToolTip(name)
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            b.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            b.setStyleSheet(swatch_qss(color))
            b.clicked.connect(lambda _=False, nm=name: self._apply_theme(nm))
            group.addButton(b)
            row.addWidget(b)
            self.theme_btns[name] = b
        layout.addWidget(holder)

    def _combo(self, caption, items, index, layout):
        cap = QtWidgets.QLabel(caption)
        cap.setObjectName("cap")
        combo = QtWidgets.QComboBox()
        combo.addItems(items)

        view = QtWidgets.QListView()
        view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        combo.setView(view)

        popup = combo.view().window()
        popup.setWindowFlags(
            popup.windowFlags()
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.NoDropShadowWindowHint)
        popup.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        combo.setCurrentIndex(index)
        combo.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        holder = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(9)
        box.addWidget(cap)
        box.addWidget(combo)
        layout.addWidget(holder)
        return combo

    def set_n(self, n):
        self.n = n
        self._x = np.arange(n, dtype=np.float32)
        self.prev_tpl = None
        self.plot.setXRange(-n * 0.012, n, padding=0)
        self.bottom.setTickSpacing(TICKS[n], TICKS[n] / 2)

    def set_gain(self, g):
        self.plot.setYRange(-1.06 / g, 1.06 / g, padding=0)
        self.left.setTickSpacing(0.5 / g, 0.25 / g)
        self.lim = 1.0 / g
        self.lim_hi.setPos(self.lim)
        self.lim_lo.setPos(-self.lim)

    def on_freeze(self, on):
        self.frozen = on
        self.freeze_btn.setText("Продолжить" if on else "Заморозить")

    def update_frame(self):
        self.frames += 1
        if self.frozen:
            return

        n = self.n
        with lock:
            data = buf[-(n + n):].copy()

        start = n
        x = self._x

        if self.trigger.isChecked():
            found = self.find_trigger(data)
            if found is not None:
                start, frac = found
                x = self._x + (1.0 - frac)

        view = data[start:start + n]
        self.last = view
        shown = np.clip(view, -self.lim, self.lim)

        self.curve.setData(x, shown, skipFiniteCheck=True)
        if self.glow.isChecked():
            self.glow_curve.setData(x, shown, skipFiniteCheck=True)

    def find_trigger(self, data):
        """Упрощённый и ускоренный триггер."""
        n = self.n
        L = len(data)
        half = n // 2

        s = np.convolve(data, self._kernel, mode="same")
        s -= s.mean()

        amp = float(np.max(np.abs(s[half:L - half])))
        if amp < 1e-4:
            return None
        h = 0.15 * amp

        st = np.zeros(L, dtype=np.int8)
        st[s > h] = 1
        st[s < -h] = -1

        last_nz = 0
        for i in range(L):
            if st[i] != 0:
                last_nz = st[i]
            else:
                st[i] = last_nz

        armed = np.flatnonzero((st[:-1] == -1) & (st[1:] == 1)) + 1
        zc = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0)) + 1
        if len(armed) == 0 or len(zc) == 0:
            return None

        pos = np.searchsorted(zc, armed, side="right") - 1
        valid = pos >= 0
        cand = np.unique(zc[pos[valid]])
        cand = cand[(cand >= half) & (cand <= L - half)]
        if len(cand) == 0:
            return None

        recent = cand[cand >= L - half - TRIGGER_MAX_AGE]
        cand = (recent if len(recent) else cand[-1:])[-24:]

        T = min(192, n // 2)
        hT = T // 2
        best = len(cand) - 1

        if (self.prev_tpl is not None and
                len(self.prev_tpl) == T and len(cand) > 1):
            idxs = cand[:, None] + np.arange(-hT, hT, dtype=np.int32)[None, :]
            tpl = data[idxs].astype(np.float32, copy=False)
            tpl -= tpl.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(tpl, axis=1) + 1e-9
            score = (tpl @ self.prev_tpl) / norms
            score += 0.012 * np.linspace(0.0, 1.0, len(cand), dtype=np.float32)
            j = int(np.argmax(score))
            if score[j] >= 0.30:
                best = j

        c = int(cand[best])
        ref = data[c - hT:c + hT].astype(np.float32, copy=False)
        ref = ref - ref.mean()
        self.prev_tpl = ref / (np.linalg.norm(ref) + 1e-9)

        a, b = float(s[c - 1]), float(s[c])
        frac = a / (a - b) if a != b else 0.0
        return c - half, min(max(frac, 0.0), 1.0)

    @staticmethod
    def _db(x):
        return "−∞" if x < 1e-6 else f"{20 * np.log10(x):+.1f}".replace("-", "−")

    def update_stats(self):
        dim = '<span style="color:#8b949e">'
        seg = self.last
        peak = float(np.max(np.abs(seg))) if len(seg) else 0.0
        rms = float(np.sqrt(np.mean(seg * seg))) if len(seg) else 0.0
        peak_color = "#f85149" if peak >= 0.999 else self.accent
        clip = "  CLIP" if peak >= 0.999 else ""

        self.peak_lbl.setText(
            f'{dim}PEAK</span>  <span style="color:{peak_color}">{self._db(peak)} dBFS{clip}</span>')
        self.rms_lbl.setText(f'{dim}RMS</span>  {self._db(rms)} dBFS')
        self.info_lbl.setText(
            f'{dim}ОКНО</span>  {self.n} смп · {self.n / SAMPLERATE * 1000:.1f} мс')

        now = time.perf_counter()
        fps = self.frames / max(now - self.t0, 1e-3)
        self.frames = 0
        self.t0 = now
        self.fps_lbl.setText(
            f'{dim}FPS</span>  ' + ("пауза" if self.frozen else f"{fps:.0f}"))

        has_err = bool(state["error"])
        if has_err != self.device_err:
            self.device_err = has_err
            self.device_lbl.setStyleSheet("color: #f85149;" if has_err else "")
        full = (f'Ошибка: {state["error"]}' if has_err
                else f'{state["device"]}  ·  {SAMPLERATE / 1000:g} кГц')
        self.device_lbl.setToolTip(full)
        self.device_lbl.setText(self.device_lbl.fontMetrics().elidedText(
            full, QtCore.Qt.TextElideMode.ElideRight,
            max(self.device_lbl.width() - 4, 20)))


def main():
    threading.Thread(target=capture, daemon=True).start()
    pg.mkQApp("PC Waveform")
    win = MainWindow()
    win.show()
    dark_titlebar(win)
    pg.exec()


if __name__ == "__main__":
    main()
