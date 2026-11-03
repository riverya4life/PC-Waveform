"""Главное окно приложения (MainWindow)."""

import backend  # noqa: F401
import gc
import math
import sys
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from backend import _BACKEND_ID, _BACKEND_NAME_KEYS, _USE_GL
from settings import (
    CHANNEL_LAYOUTS,
    CH_LAYOUT_TO_N,
    CHUNK,
    CONFIG,
    DEFAULT_CONFIG,
    DEMO_EDGE,
    FPS_AVG_S,
    FPS_OPTIONS,
    GAIN_OPTIONS,
    HZ_EST_SAMPLES,
    LANG,
    LANGUAGES,
    MAX_CH,
    SAMPLERATE,
    SAMPLE_OPTIONS,
    SLOW_STATS_S,
    STATS_MS,
    TICKS,
    UPDATE_MS,
    WATCHDOG_MS,
    _save_config,
    cap_text,
    tr,
)
from theme import (
    DEFAULT_THEME,
    IconButton,
    LogoMark,
    Segmented,
    StatTile,
    Swatch,
    THEMES,
    THEME_HEX,
    ToggleSwitch,
    _make_icon,
    _mix_dark,
    _rgba,
    _stereo_r_color,
    _surface,
    system_accent,
    make_qss,
)
from audio import (
    Capturer,
    _bridge,
    _chunk_pending,
    _ui_gate,
    audio_snapshot,
    audio_snapshot_multi,
    list_loopback_devices,
    set_channel_count,
    state,
)
from trigger import TriggerMixin, _absmax
from widgets import (
    FlowLayout,
    GlowRenderer,
    SpacedAxis,
    StaticShadow,
    WavePlotWidget,
    _gv_cache_mode,
    GradientCurveItem,
    make_gradient_pen,
)
from dsp import make_filter
from analysis import BpmDetector, PeakHold


def _trim_memory(freeze_gc=False):
    gc.collect()
    if freeze_gc and hasattr(gc, "freeze"):
        gc.freeze()
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetProcessWorkingSetSize.argtypes = [
                wintypes.HANDLE,
                ctypes.c_size_t,
                ctypes.c_size_t,
            ]
            k32.SetProcessWorkingSetSize(
                k32.GetCurrentProcess(), ctypes.c_size_t(-1), ctypes.c_size_t(-1)
            )
        except Exception:
            pass


def _demo_edge_window(t: np.ndarray, edge: float = DEMO_EDGE) -> np.ndarray:
    w = np.ones_like(t)
    left = t < edge
    right = t > 1.0 - edge
    if np.any(left):
        w[left] = 0.5 * (1.0 - np.cos(np.pi * t[left] / edge))
    if np.any(right):
        w[right] = 0.5 * (1.0 - np.cos(np.pi * (1.0 - t[right]) / edge))
    return w


def _demo_envelope(t_norm: np.ndarray) -> np.ndarray:
    return _demo_edge_window(t_norm, edge=DEMO_EDGE).astype(np.float32)


class FpsFreezeTile(QtWidgets.QFrame):
    """Плитка: слева FPS, справа кнопка «Заморозить/Продолжить»."""

    def __init__(self, width: int, button, parent=None):
        super().__init__(parent)
        self.setObjectName("tile")
        self.setFixedWidth(width)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 9, 8)
        lay.setSpacing(10)
        self.info = MiniTile()
        lay.addWidget(self.info, 1)
        lay.addWidget(button, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

    def set(self, cap: str, val_html: str):
        self.info.set(cap, val_html)


class MiniTile(QtWidgets.QWidget):
    """Подплитка без рамки (подпись + значение) — часть PairTile."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        self.cap = QtWidgets.QLabel()
        self.cap.setObjectName("tilecap")
        self.val = QtWidgets.QLabel()
        self.val.setObjectName("tileval")
        self.val.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lay.addWidget(self.cap)
        lay.addWidget(self.val)

    def set(self, cap: str, val_html: str):
        if self.cap.text() != cap:
            self.cap.setText(cap)
        if self.val.text() != val_html:
            self.val.setText(val_html)


class PairTile(QtWidgets.QFrame):
    """Одна плитка с двумя значениями: слева RMS, справа частота."""

    def __init__(self, width: int, parent=None):
        super().__init__(parent)
        self.setObjectName("tile")
        self.setFixedWidth(width)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 11)
        lay.setSpacing(12)
        self.left = MiniTile()
        self.right = MiniTile()
        sep = QtWidgets.QFrame()
        sep.setFixedWidth(1)
        sep.setStyleSheet("background: rgba(255,255,255,22); border: none;")
        lay.addWidget(self.left, 1)
        lay.addWidget(sep)
        lay.addWidget(self.right, 1)


class MainWindow(TriggerMixin, QtWidgets.QWidget):
    CTRL_H = 36
    ROW_H = 54
    _RESIZE_MARGIN = 8
    _MIN_W = 160
    _MIN_H = 40
    SOLO_W = 450
    SOLO_H = 150
    CURVE_DS_FACTOR = 1.0
    COL_W = 300  # ширина правой колонки со статистикой (и блока «Поток» внизу)
    SHOW_HEADER = True  # логотип, заголовок и «пилюля» устройства сверху
    SHOW_DEBUG = (
        # отладочная плашка: размер окна и масштаб (F12 — скрыть/показать)
        False
    )
    TRIG_BLEND_TAU = 0.032  # с: плавность смены кадров в режиме стабилизации

    def __init__(self):
        super().__init__()
        self.setObjectName("root")
        self.setWindowTitle(tr("main_title"))
        self.resize(1280, 548)

        self.n = CONFIG.get("n", 4096)
        self.frozen = False

        self.ch_layout = CONFIG.get("ch_layout", "mono")
        self._n_ch = CH_LAYOUT_TO_N.get(self.ch_layout, 1)
        self.channels = "mono" if self._n_ch == 1 else "stereo"
        set_channel_count(self._n_ch)

        self.last = np.zeros(self.n, dtype=np.float32)
        self.last_r = np.zeros(self.n, dtype=np.float32)
        self.lim = 1.0
        self.prev_tpl = None
        self._crossing_smooth = None
        self._trigger_pending_samples = 0
        self._last_total = 0
        self._period_smooth = 0.0
        self._lock_quality = 0.0

        self._fade_on = bool(CONFIG.get("fade", True))
        self._fade_blend = 1.0 if self._fade_on else 0.0
        self._fade_target = self._fade_blend
        self._demo_env_cache = None
        self._demo_agc = 0.0
        self._demo_agc_l = 0.0
        self._demo_agc_r = 0.0

        self._solo = False
        self._card_shadow = None
        self._card_layout = None
        self._card_saved_margins = None
        self._plot_bg_main = None
        self._solo_pinned = False
        self._solo_transparent = True
        self._hud_on = False
        self._solo_drag_mode = None
        self._solo_resize_edges = (False, False, False, False)
        self._solo_origin_global = None
        self._solo_origin_geom = None

        self._fps_setting = int(CONFIG.get("fps", 0))

        self._band_kind = CONFIG.get("bandpass", "off")
        self._filters = [make_filter(self._band_kind) for _ in range(MAX_CH)]
        self._bpm = BpmDetector(SAMPLERATE)
        self._bpm_on = bool(CONFIG.get("bpm_on", True))
        self._bpm_last_total = 0
        self._peak_hold = PeakHold(decay=0.94, smooth=0.5, window=15)
        self._peak_hold_r = PeakHold(decay=0.94, smooth=0.5, window=15)
        self._sm_bufs = {}
        self._sm_out = {}
        self._sm_scale = {}
        self._sm_tmp = None
        self._last_frame_t = 0.0
        self._bpm_scr = np.empty(4096, dtype=np.float32)
        self._show_gradient = bool(CONFIG.get("gradient", False))
        self._show_peak_hold = bool(CONFIG.get("peak_hold", True))
        self._smooth_stab = bool(CONFIG.get("smooth_stab", True))
        self._edge_clip_on = bool(CONFIG.get("edge_clip", True))

        self.multi_curves = []
        self.multi_peak_curves = []
        self.multi_peak_holds = []
        self._multi_bufs = []
        self._filt_out = np.empty(self.n, dtype=np.float32)
        self._alloc_scratch(self.n)

        self._frame_sig = None
        self._was_hidden = False
        self._last_keepalive = 0.0
        self._chunk_driven = False
        self._chunk_connected = False

        self._kernel = np.ones(8, dtype=np.float32) / 8
        self._stats_idle_s = STATS_MS / 1000.0
        self._retry_pending = False
        self._x = np.arange(self.n, dtype=np.float32)
        self.frames = 0
        self._hz_smooth = 0.0
        self.t0 = time.perf_counter()
        self._stats_t = 0.0
        self._fps_measured = 0.0

        self._metrics_total = -(10**9)
        self._peak_val = 0.0
        self._peak_l_val = 0.0
        self._peak_r_val = 0.0
        self._rms_val = 0.0

        self.theme_id = CONFIG.get("theme", DEFAULT_THEME)
        self.theme_btns = {}
        self._theme_dialog = None
        self._sys_last = None
        self._sys_applied = None
        self._sys_timer = None
        self.accent = self._theme_color(self.theme_id)
        self._current_color = QtGui.QColor(self.accent)
        self._theme_anim = None
        self._stereo_r_hex = _stereo_r_color(self.accent).name()
        self._solid_pen = None
        self._gradient_pen = None

        self._capturer = Capturer()
        self._loading = True
        self._layout_applied = False
        self._set_chunk_driven(int(CONFIG.get("fps", 0)) == 0)

        self._build_ui()
        self._apply_theme(self.theme_id, animate=False)
        self._apply_ch_layout(self.ch_layout)
        self.set_n(self.n)
        self.set_gain(CONFIG.get("gain", 1))
        if CONFIG.get("glow", False):
            self.glow_renderer.set_enabled(True)
            if self._n_ch == 2:
                self.glow_renderer_r.set_enabled(True)
        self._populate_devices()
        self._loading = False
        self._apply_demo_state()
        self._fit_pill()

        self.setWindowOpacity(0.0)
        self._win_fade = QtCore.QPropertyAnimation(self, b"windowOpacity")
        self._win_fade.setDuration(280)
        self._win_fade.setStartValue(0.0)
        self._win_fade.setEndValue(1.0)
        self._win_fade.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._win_fade.start()

        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(UPDATE_MS)
        self._apply_fps()

        self.stats_timer = QtCore.QTimer(self)
        self.stats_timer.timeout.connect(self._stats_fallback)
        self.stats_timer.start(STATS_MS)

        shortcut_cls = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut
        shortcut_cls(QtGui.QKeySequence("Space"), self, activated=self.freeze_btn.click)
        self._solo_esc = shortcut_cls(
            QtGui.QKeySequence("Escape"), self, activated=self._exit_solo
        )
        self._solo_esc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._solo_esc.setEnabled(False)

    def _toggle_debug(self):
        self.SHOW_DEBUG = not self.SHOW_DEBUG
        if self.SHOW_HEADER:
            self.dbg_lbl.setVisible(self.SHOW_DEBUG)
        self._update_debug()
        self._fit_pill()

    def _update_debug(self):
        if not self.SHOW_DEBUG or not self.SHOW_HEADER:
            return
        w, h = self.width(), self.height()
        dpr = self.devicePixelRatioF()
        dpi = self.logicalDpiX()
        txt = (
            f"{w}×{h} · {round(dpr * 100)}% · "
            f"{round(w * dpr)}×{round(h * dpr)} px · {dpi:.0f} dpi"
        )
        if self.dbg_lbl.text() != txt:
            self.dbg_lbl.setText(txt)

    def showEvent(self, ev):
        super().showEvent(ev)
        wh = self.windowHandle()
        if wh is not None and not getattr(self, "_dbg_hooked", False):
            self._dbg_hooked = True
            try:
                wh.screenChanged.connect(lambda *_: self._update_debug())
            except Exception:
                pass
        self._update_debug()
        self._fit_pill()

    def _alloc_scratch(self, n):
        f = np.float32
        self._scr_t = np.empty(n * 2, f)
        self._fade_buf = np.empty(n, f)
        self._disp_buf = np.empty(n, f)
        self._filt_out = np.empty(n, f)
        self._xs = np.empty(n, f)
        self._dec_x = np.empty(n, f)
        self._dec_y = np.empty(n, f)

        self._multi_bufs = (
            [np.empty(n * 2, f) for _ in range(self._n_ch)] if self._n_ch > 2 else []
        )

        if self._n_ch == 2:
            self._scr_l = np.empty(n * 2, f)
            self._scr_r = np.empty(n * 2, f)
            self._fade_buf_r = np.empty(n, f)
            self._disp_buf_r = np.empty(n, f)
            self._dec_yr = np.empty(n, f)
        else:
            self._scr_l = self._scr_r = None
            self._fade_buf_r = self._disp_buf_r = self._dec_yr = None

    @staticmethod
    def _decimate(y, ds, m, out):
        if ds <= 1:
            return y
        return np.mean(y[: m * ds].reshape(m, ds), axis=1, out=out[:m])

    def _trim_if_hidden(self):
        if self._ui_hidden():
            _trim_memory()

    def _fit_pill(self):
        pass

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._update_debug()
        self._fit_pill()

    def _save_settings(self):
        CONFIG["lang"] = LANG["current"]
        CONFIG["theme"] = self.theme_id
        CONFIG["n"] = self.n
        g_idx = self.gain_combo.currentIndex()
        if 0 <= g_idx < len(GAIN_OPTIONS):
            CONFIG["gain"] = GAIN_OPTIONS[g_idx]
        CONFIG["stabilization"] = self.trigger.isChecked()
        CONFIG["glow"] = self.glow.isChecked()
        CONFIG["ch_layout"] = self.ch_layout
        CONFIG["fade"] = self._fade_on
        CONFIG["fps"] = int(self._fps_setting)
        CONFIG["bandpass"] = self._band_kind
        CONFIG["gradient"] = self._show_gradient
        CONFIG["peak_hold"] = self._show_peak_hold
        CONFIG["bpm_on"] = self._bpm_on
        CONFIG["edge_clip"] = self._edge_clip_on
        idx = self.device_combo.currentIndex()
        if idx >= 0:
            CONFIG["device_id"] = self.device_combo.itemData(idx)
        _save_config(CONFIG)

    def _populate_devices(self):
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        devices = list_loopback_devices()
        if not devices:
            self.device_combo.addItem(tr("no_devices"), None)
            self.device_combo.setEnabled(False)
            self.device_combo.blockSignals(False)
            return
        preferred = CONFIG.get("device_id")
        select_idx = 0
        for i, (name, did) in enumerate(devices):
            self.device_combo.addItem(name, did)
            if preferred is not None and str(did) == str(preferred):
                select_idx = i
        self.device_combo.setEnabled(True)
        self.device_combo.setCurrentIndex(select_idx)
        self.device_combo.blockSignals(False)
        self._on_device_change(select_idx)

    def _on_device_change(self, idx):
        if idx < 0:
            return
        did = self.device_combo.itemData(idx)
        if did is None:
            return
        self.prev_tpl = None
        self._crossing_smooth = None
        self._last_total = state.get("total_samples", 0)
        self._bpm_last_total = self._last_total
        self._capturer.start(did)
        if not getattr(self, "_loading", False):
            CONFIG["device_id"] = did
            _save_config(CONFIG)

    def _on_language_change(self, idx):
        if idx < 0:
            return
        code = self.lang_combo.itemData(idx)
        if not code or code == LANG["current"]:
            return
        LANG["current"] = code
        CONFIG["lang"] = code
        _save_config(CONFIG)
        self._retranslate_ui()

    @staticmethod
    def _fps_label(v: int) -> str:
        if v == 0:
            return tr("fps_unlimited")
        if v == -1:
            return tr("fps_monitor")
        return f"{v} FPS"

    def _refresh_fps_combo(self):
        if not hasattr(self, "fps_combo"):
            return
        self.fps_combo.blockSignals(True)
        self.fps_combo.clear()
        for v in FPS_OPTIONS:
            self.fps_combo.addItem(self._fps_label(v), v)
        try:
            idx = FPS_OPTIONS.index(self._fps_setting)
        except ValueError:
            idx = 0
        self.fps_combo.setCurrentIndex(idx)
        self.fps_combo.blockSignals(False)
        self.fps_combo.setToolTip(tr("fps_tip"))

    def _on_fps_change(self, idx):
        if idx < 0:
            return
        v = self.fps_combo.itemData(idx)
        if v is None or v == self._fps_setting:
            return
        self._fps_setting = int(v)
        self._apply_fps()
        if not getattr(self, "_loading", False):
            CONFIG["fps"] = self._fps_setting
            _save_config(CONFIG)

    def _apply_fps(self):
        interval = UPDATE_MS
        v = int(getattr(self, "_fps_setting", 0))
        self._set_chunk_driven(v == 0)
        if v == 0:
            interval = WATCHDOG_MS
        elif v == -1:
            scr = self.screen()
            if scr is not None:
                hz = float(scr.refreshRate() or 0.0)
                if hz > 1.0:
                    interval = max(1, int(round(1000.0 / hz)))
        elif v > 0:
            interval = max(1, int(round(1000.0 / v)))
        t = getattr(self, "timer", None)
        if t is not None:
            t.stop()
            t.setTimerType(
                QtCore.Qt.TimerType.CoarseTimer
                if v == 0
                else QtCore.Qt.TimerType.PreciseTimer
            )
            t.start(interval)
        self._stats_idle_s = max(STATS_MS, 2 * interval) / 1000.0

    def _set_chunk_driven(self, on: bool):
        self._chunk_driven = bool(on)
        if on and not self._chunk_connected:
            _bridge.chunk.connect(self._on_audio_chunk)
            self._chunk_connected = True
        elif not on and self._chunk_connected:
            try:
                _bridge.chunk.disconnect(self._on_audio_chunk)
            except Exception:
                pass
            self._chunk_connected = False

    def _on_tick(self):
        _chunk_pending[0] = False
        self.update_frame()

    def _on_audio_chunk(self):
        _chunk_pending[0] = False
        self.update_frame()

    @staticmethod
    def _freeze_text(on: bool) -> str:
        return "  " + (tr("resume") if on else tr("freeze"))

    @staticmethod
    def _freeze_icon(play: bool, color) -> QtGui.QIcon:
        size, dpr = 18, 2
        pm = QtGui.QPixmap(size * dpr, size * dpr)
        pm.setDevicePixelRatio(dpr)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(color))
        if play:
            p.drawPolygon(
                QtGui.QPolygonF(
                    [
                        QtCore.QPointF(5.5, 3.0),
                        QtCore.QPointF(15.0, 9.0),
                        QtCore.QPointF(5.5, 15.0),
                    ]
                )
            )
        else:
            p.drawRoundedRect(QtCore.QRectF(4.0, 3.5, 3.6, 11.0), 1.3, 1.3)
            p.drawRoundedRect(QtCore.QRectF(10.4, 3.5, 3.6, 11.0), 1.3, 1.3)
        p.end()
        return QtGui.QIcon(pm)

    def _update_freeze_icon(self):
        if not hasattr(self, "freeze_btn"):
            return
        color = "#0d1117" if self.frozen else self.accent
        self.freeze_btn.setIcon(self._freeze_icon(self.frozen, color))

    def _retranslate_ui(self):
        self.setWindowTitle(tr("main_title"))
        self.sub_lbl.setText(tr("subtitle"))
        self.freeze_btn.setText(self._freeze_text(self.frozen))
        self.trigger.setToolTip(tr("stabilization_tip"))
        if hasattr(self, "fade_sw"):
            self.fade_sw.setToolTip(tr("fade_tip"))
        if getattr(self, "_settings_dialog", None):
            for sw in (
                self.trigger,
                self.glow,
                getattr(self, "fade_sw", None),
                getattr(self, "gradient_sw", None),
                getattr(self, "peak_hold_sw", None),
                getattr(self, "bpm_sw", None),
                getattr(self, "smooth_sw", None),
                getattr(self, "edge_clip_sw", None),
            ):
                if sw is not None:
                    sw.setText("")
            if getattr(self, "_cap_stab", None):
                self._cap_stab.setText(tr("stabilization"))
            if getattr(self, "_cap_glow", None):
                self._cap_glow.setText(tr("glow"))
            if getattr(self, "_cap_fade", None):
                self._cap_fade.setText(tr("fade"))
            if getattr(self, "_cap_edge", None):
                self._cap_edge.setText(tr("edge_clip"))
        if self.cap_window:
            self.cap_window.setText(tr("window"))
        if self.cap_gain:
            self.cap_gain.setText(tr("gain"))
        self.cap_stream.setText(tr("stream"))
        if self.cap_theme:
            self.cap_theme.setText(tr("theme"))
        if self.cap_lang:
            self.cap_lang.setText(tr("language"))
        if self.cap_channels:
            self.cap_channels.setText(tr("ch_layout"))
        if hasattr(self, "theme_btn"):
            self.theme_btn.setText(tr("theme"))
        if hasattr(self, "settings_btn"):
            self.settings_btn.setText(tr("settings"))
            self.settings_btn.setToolTip(tr("settings_tip"))
        if hasattr(self, "cap_settings") and self.cap_settings:
            self.cap_settings.setText(cap_text("settings"))
        if hasattr(self, "edge_clip_sw"):
            self.edge_clip_sw.setToolTip(tr("edge_clip_tip"))

        if hasattr(self, "ch_layout_combo"):
            for i, (_key, _n, label) in enumerate(CHANNEL_LAYOUTS):
                self.ch_layout_combo.setItemText(i, tr(label))

        self.plot.setToolTip(tr("solo_tip"))
        idx = self.n_combo.currentIndex()
        self.n_combo.blockSignals(True)
        self.n_combo.clear()
        for v in SAMPLE_OPTIONS:
            self.n_combo.addItem(f"{v} {tr('samples')}")
        self.n_combo.setCurrentIndex(idx)
        self.n_combo.blockSignals(False)
        if hasattr(self, "theme_btns"):
            for tid, btn in self.theme_btns.items():
                if tid == "custom":
                    btn.setToolTip(tr("theme_custom_tip"))
                elif tid == "system":
                    btn.setToolTip(tr("theme_system_tip"))
                else:
                    btn.setToolTip(tr(f"theme_{tid}"))
        if not self.device_combo.isEnabled() and self.device_combo.count() == 1:
            self.device_combo.blockSignals(True)
            self.device_combo.setItemText(0, tr("no_devices"))
            self.device_combo.blockSignals(False)
        if self._solo:
            self.card.setWindowTitle(tr("solo_title"))
        if hasattr(self, "fps_combo"):
            self._refresh_fps_combo()
        if self.cap_fps:
            self.cap_fps.setText(tr("fps"))
        self.bpm_tile.setToolTip(tr("bpm_tip"))
        self._apply_hud_texts()
        self._fit_pill()
        self.update_stats()

    def _reapply_ranges(self):
        n = self.n
        self.plot.setXRange(-n * 0.012, n, padding=0)
        g = 1.0 / self.lim if self.lim else 1.0
        self.plot.setYRange(-1.06 / g, 1.06 / g, padding=0)

    def _enter_solo(self):
        if self._solo:
            return
        self._solo = True
        size = QtCore.QSize(self.SOLO_W, self.SOLO_H)
        if self._card_layout is not None:
            self._card_saved_margins = self._card_layout.contentsMargins()
            self._card_layout.setContentsMargins(0, 0, 0, 0)
        pi = self.plot.getPlotItem()
        pi.hideAxis("bottom")
        pi.hideAxis("left")
        pi.showGrid(x=True, y=True, alpha=0.05)
        self.lim_hi.setVisible(False)
        self.lim_lo.setVisible(False)
        try:
            pi.getViewBox().disableAutoRange()
        except Exception:
            pass
        self._plot_bg_main = pg.mkColor(_surface(self.accent))
        self.plot.setBackground(self._plot_bg())
        self.plot.setStyleSheet("background: transparent; border: none;")
        self.card.hide()
        self._root_layout.removeWidget(self.card)
        self.card.setParent(None)
        self._apply_solo_flags()
        self.card.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.card.setWindowTitle(tr("solo_title"))
        self.card.setStyleSheet(self._solo_card_qss())
        if self._card_shadow is not None:
            self._card_shadow.sync()
        self.card.setMouseTracking(True)
        self.plot.setMouseTracking(True)
        try:
            self.plot.viewport().setMouseTracking(True)
        except Exception:
            pass
        self._apply_plot_render_mode()
        self.plot.setBackground(self._plot_bg())
        self.card.resize(size)
        self.card.show()
        self.card.raise_()
        self.card.activateWindow()
        self.hide()
        self._style_hud()
        self._hud.show()
        self._layout_hud()
        self._hud.raise_()
        self._hud_on = False
        self._hud_fx.setOpacity(0.0)
        self._hud_show(True)
        QtCore.QTimer.singleShot(0, self._reapply_ranges)
        self._solo_esc.setEnabled(True)

    def _exit_solo(self):
        if not self._solo:
            return
        self._solo = False
        self._solo_drag_mode = None
        self._hud_timer.stop()
        self._hud_anim.stop()
        self._hud_on = False
        self._hud.hide()
        self.card.hide()
        self.card.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, False
        )
        self.card.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, False)
        self.card.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.card.setWindowTitle("")
        self.card.setStyleSheet("")
        self.card.unsetCursor()
        if self._card_shadow is not None:
            QtCore.QTimer.singleShot(0, self._card_shadow.sync)
        if self._card_layout is not None and self._card_saved_margins is not None:
            self._card_layout.setContentsMargins(self._card_saved_margins)
        pi = self.plot.getPlotItem()
        pi.showAxis("bottom")
        pi.showAxis("left")
        pi.showGrid(x=True, y=True, alpha=0.07)
        self.lim_hi.setVisible(True)
        self.lim_lo.setVisible(True)
        self.left.setWidth(48)
        if self._plot_bg_main is not None:
            self.plot.setBackground(self._plot_bg_main)
        self._apply_plot_render_mode()
        self._top_layout.insertWidget(0, self.card, 1)
        self.plot.setStyleSheet("background: transparent;")
        self.show()
        self.card.show()
        self.raise_()
        self.activateWindow()
        QtCore.QTimer.singleShot(0, self._reapply_ranges)
        self._solo_esc.setEnabled(False)

    def _plot_bg(self):
        if self._solo and self._solo_transparent:
            return pg.mkColor((0, 0, 0, 3))
        return pg.mkColor(_surface(self.accent))

    def _solo_card_qss(self) -> str:
        if self._solo_transparent:
            bg, border = "rgba(0,0,0,3)", _rgba(self.accent, 0.35)
        else:
            bg, border = _surface(self.accent), _rgba(self.accent, 0.55)
        return f"""
            QFrame#card {{ background: {bg}; border: 1px solid {border}; border-radius: 0; }}
            QToolTip {{ background: #161b22; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 5px 10px; font-family: 'Segoe UI'; font-size: 12px; }}
        """

    def _apply_solo_flags(self):
        flags = QtCore.Qt.WindowType.Window | QtCore.Qt.WindowType.FramelessWindowHint
        if self._solo_pinned:
            flags |= QtCore.Qt.WindowType.WindowStaysOnTopHint
        self.card.setWindowFlags(flags)

    def _apply_plot_render_mode(self):
        transparent = self._solo and self._solo_transparent
        self.plot.transparent_bg = transparent
        try:
            self.plot.setCacheMode(_gv_cache_mode(transparent))
        except Exception:
            pass
        if not _USE_GL:
            self.plot.viewport().update()
            return
        try:
            vp = self.plot.viewport()
            if not hasattr(vp, "setUpdateBehavior"):
                return
            cls = type(vp)
            ub = getattr(cls, "UpdateBehavior", cls)
            vp.setUpdateBehavior(
                ub.NoPartialUpdate if transparent else ub.PartialUpdate
            )
            vp.setAutoFillBackground(False)
            vp.update()
        except Exception:
            pass

    def _on_solo_bg(self, on):
        self._solo_transparent = bool(on)
        if self._solo:
            self._apply_plot_render_mode()
            self.plot.setBackground(self._plot_bg())
            self.card.setStyleSheet(self._solo_card_qss())
            self.plot.viewport().update()

    def _on_solo_pin(self, on):
        self._solo_pinned = bool(on)
        if self._solo:
            self._apply_solo_flags()
            self.card.show()
            self.card.raise_()
            self.card.activateWindow()
            self._layout_hud()
            self._hud.raise_()

    def _build_solo_hud(self):
        hud = QtWidgets.QFrame(self.card)
        hud.setObjectName("hud")
        hud.setMouseTracking(True)
        hud.setFixedHeight(40)
        lay = QtWidgets.QHBoxLayout(hud)
        lay.setContentsMargins(14, 4, 8, 4)
        lay.setSpacing(8)
        self.solo_dot = QtWidgets.QLabel()
        self.solo_dot.setObjectName("sdot")
        self.solo_dot.setFixedSize(8, 8)
        self.solo_title = QtWidgets.QLabel(tr("brand"), hud)
        self.solo_title.setObjectName("stitle")
        self.solo_title.hide()
        self.solo_read = QtWidgets.QLabel("")
        self.solo_read.setObjectName("sread")
        for w in (self.solo_dot, self.solo_title, self.solo_read):
            w.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.solo_freeze = IconButton("freeze", checkable=True)
        self.solo_freeze.clicked.connect(lambda: self.freeze_btn.click())
        self.solo_bg = IconButton("bg", checkable=True)
        self.solo_bg.setChecked(self._solo_transparent)
        self.solo_bg.toggled.connect(self._on_solo_bg)
        self.solo_pin = IconButton("pin", checkable=True)
        self.solo_pin.toggled.connect(self._on_solo_pin)
        self.solo_close = IconButton("close")
        self.solo_close.clicked.connect(self._exit_solo)
        lay.addWidget(self.solo_dot, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.solo_read)
        lay.addStretch(1)
        lay.addWidget(self.solo_freeze)
        lay.addWidget(self.solo_bg)
        lay.addWidget(self.solo_pin)
        lay.addWidget(self.solo_close)
        self._hud = hud
        self._hud_lay = lay
        self._solo_vals = ("−∞", "−∞", "0.00Hz", "0.00Hz")
        self._hud_fx = QtWidgets.QGraphicsOpacityEffect(hud)
        self._hud_fx.setOpacity(0.0)
        hud.setGraphicsEffect(self._hud_fx)
        self._hud_anim = QtCore.QPropertyAnimation(self._hud_fx, b"opacity", self)
        self._hud_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._hud_timer = QtCore.QTimer(self)
        self._hud_timer.setSingleShot(True)
        self._hud_timer.timeout.connect(lambda: self._hud_show(False))
        hud.hide()
        self._apply_hud_texts()
        self._style_hud()

    def _apply_hud_texts(self):
        if not hasattr(self, "_hud"):
            return
        self.solo_freeze.setToolTip(tr("solo_freeze"))
        self.solo_bg.setToolTip(tr("solo_bg"))
        self.solo_pin.setToolTip(tr("solo_pin"))
        self.solo_close.setToolTip(tr("solo_exit"))
        self.solo_title.setText(tr("brand"))

    def _style_hud(self):
        if not hasattr(self, "_hud"):
            return
        self._hud.setStyleSheet(f"""
            QFrame#hud {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(0,0,0,175), stop:1 rgba(0,0,0,0)); border: none; }}
            QLabel {{ color: #c9d1d9; font-family: 'Segoe UI'; font-size: 12px; background: transparent; }}
            QLabel#stitle {{ color: #f0f6fc; font-size: 13px; font-weight: 700; }}
            QLabel#sread {{ color: #8b949e; font-family: 'Cascadia Mono', 'Consolas', monospace; font-size: 12px; }}
            QLabel#sdot {{ background: {self.accent}; border-radius: 4px; }}
        """)
        for b in (self.solo_freeze, self.solo_bg, self.solo_pin, self.solo_close):
            b.set_accent(self.accent)

    def _layout_hud(self):
        self._hud.setGeometry(0, 0, self.card.width(), self._hud.height())
        self._fit_hud()

    def _fit_hud(self):
        w = self.card.width()
        sp = self._hud_lay.spacing()
        m = self._hud_lay.contentsMargins()
        btns = 4 * 32 + 3 * sp
        avail = w - m.left() - m.right() - btns - sp
        dot_w = 8 + sp
        fm = self.solo_read.fontMetrics()
        p, r, _hz_full, hz_short = self._solo_vals
        cands = [
            f"PEAK {p}  ·  RMS {r} dBFS  ·  {hz_short}",
            f"PEAK {p} · RMS {r} · {hz_short}",
            f"{p} · {r} dB · {hz_short}",
            f"PEAK {p} · RMS {r}",
            f"{p} · {r} dB",
            hz_short,
        ]
        left = avail - dot_w - 10
        text = ""
        for t in cands:
            if fm.horizontalAdvance(t) <= left:
                text = t
                break
        self.solo_title.setVisible(False)
        self.solo_read.setVisible(bool(text))
        self.solo_read.setText(text)

    def _hud_show(self, on: bool):
        if not self._solo:
            return
        self._hud_on = on
        if on:
            self._fit_hud()
        self._hud_anim.stop()
        self._hud_anim.setStartValue(self._hud_fx.opacity())
        self._hud_anim.setEndValue(1.0 if on else 0.0)
        self._hud_anim.setDuration(140 if on else 320)
        self._hud_anim.start()
        self._hud.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, not on
        )
        if on:
            self._hud_timer.start(2400)
        else:
            self._hud_timer.stop()

    def _poke_hud(self):
        if not self._hud_on:
            self._hud_show(True)
        else:
            self._hud_timer.start(2400)

    @staticmethod
    def _global_pos(ev):
        if hasattr(ev, "globalPosition"):
            return ev.globalPosition().toPoint()
        return ev.globalPos()

    def _solo_hit_edges(self, gp):
        geo = self.card.geometry()
        if not geo.contains(gp):
            return (False, False, False, False)
        local = gp - geo.topLeft()
        m = self._RESIZE_MARGIN
        return (
            local.x() <= m,
            local.x() >= geo.width() - m,
            local.y() <= m,
            local.y() >= geo.height() - m,
        )

    def _solo_update_cursor(self, gp):
        l, r, t, b = self._solo_hit_edges(gp)
        cur = QtCore.Qt.CursorShape.ArrowCursor
        if (l and t) or (r and b):
            cur = QtCore.Qt.CursorShape.SizeFDiagCursor
        elif (r and t) or (l and b):
            cur = QtCore.Qt.CursorShape.SizeBDiagCursor
        elif l or r:
            cur = QtCore.Qt.CursorShape.SizeHorCursor
        elif t or b:
            cur = QtCore.Qt.CursorShape.SizeVerCursor
        self.card.setCursor(cur)

    def _solo_begin_drag(self, gp):
        l, r, t, b = self._solo_hit_edges(gp)
        geo = self.card.geometry()
        self._solo_origin_global = gp
        self._solo_origin_geom = QtCore.QRect(geo)
        if l or r or t or b:
            self._solo_drag_mode = "resize"
            self._solo_resize_edges = (l, r, t, b)
        else:
            self._solo_drag_mode = "move"

    def _solo_apply_drag(self, gp):
        if self._solo_drag_mode is None or self._solo_origin_global is None:
            return
        dx = gp.x() - self._solo_origin_global.x()
        dy = gp.y() - self._solo_origin_global.y()
        g = self._solo_origin_geom
        if self._solo_drag_mode == "move":
            self.card.move(g.topLeft() + QtCore.QPoint(dx, dy))
            return
        l, r, t, b = self._solo_resize_edges
        x, y, w, h = g.x(), g.y(), g.width(), g.height()
        minw, minh = self._MIN_W, self._MIN_H
        if l:
            nx = x + dx
            nw = w - dx
            if nw < minw:
                nx = x + w - minw
                nw = minw
            x, w = nx, nw
        if r:
            w = max(minw, w + dx)
        if t:
            ny = y + dy
            nh = h - dy
            if nh < minh:
                ny = y + h - minh
                nh = minh
            y, h = ny, nh
        if b:
            h = max(minh, h + dy)
        self.card.setGeometry(x, y, w, h)

    def eventFilter(self, obj, ev):
        et = ev.type()
        is_plot = obj is self.plot or (
            hasattr(self.plot, "viewport") and obj is self.plot.viewport()
        )
        is_card = obj is self.card
        if not (is_plot or is_card):
            return super().eventFilter(obj, ev)
        if is_card and self._solo:
            if et == QtCore.QEvent.Type.Resize:
                self._layout_hud()
            elif et == QtCore.QEvent.Type.Leave:
                self._hud_show(False)
        if is_card and et == QtCore.QEvent.Type.Close:
            if self._solo:
                ev.ignore()
                self._exit_solo()
                return True
            return super().eventFilter(obj, ev)
        if is_plot and et == QtCore.QEvent.Type.MouseButtonPress:
            if ev.button() == QtCore.Qt.MouseButton.RightButton:
                if self._solo:
                    self._exit_solo()
                else:
                    self._enter_solo()
                return True
        if self._solo:
            if et == QtCore.QEvent.Type.MouseButtonPress:
                if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                    gp = self._global_pos(ev)
                    self._solo_begin_drag(gp)
                    return True
            elif et == QtCore.QEvent.Type.MouseMove:
                gp = self._global_pos(ev)
                self._poke_hud()
                if self._solo_drag_mode is None:
                    self._solo_update_cursor(gp)
                else:
                    self._solo_apply_drag(gp)
                return True
            elif et == QtCore.QEvent.Type.MouseButtonRelease:
                if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                    self._solo_drag_mode = None
                    self._solo_update_cursor(self._global_pos(ev))
                    return True
        return super().eventFilter(obj, ev)

    def _animate_theme(self, target_hex: str):
        target = QtGui.QColor(target_hex)
        if (
            self._theme_anim is not None
            and self._theme_anim.state() == QtCore.QAbstractAnimation.State.Running
        ):
            self._theme_anim.stop()
        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(420)
        anim.setStartValue(self._current_color)
        anim.setEndValue(target)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        anim.valueChanged.connect(self._on_theme_color)

        def _done():
            self._current_color = target
            self._update_app_icon(target)

        anim.finished.connect(_done)
        anim.start()
        self._theme_anim = anim

    def _update_app_icon(self, color=None):
        icon = _make_icon(color or self.accent)
        self.setWindowIcon(icon)
        self.card.setWindowIcon(icon)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)

    def _rebuild_pens(self):
        c = QtGui.QColor(self._current_color)
        c.setAlpha(255)
        quiet = QtGui.QColor(c)
        quiet.setAlpha(40)
        loud = QtGui.QColor(c).lighter(120)
        self._solid_pen = pg.mkPen(c, width=1.7)
        self._gradient_pen = make_gradient_pen(quiet, loud, 1.7)
        self.curve.setPen(
            self._gradient_pen if self._show_gradient else self._solid_pen
        )

        c2 = _stereo_r_color(c.name())
        self._stereo_r_hex = c2.name()
        quiet2 = QtGui.QColor(c2)
        quiet2.setAlpha(40)
        loud2 = QtGui.QColor(c2).lighter(120)
        self.curve_r.setPen(
            make_gradient_pen(quiet2, loud2, 1.5)
            if self._show_gradient
            else pg.mkPen(c2, width=1.5)
        )

        if hasattr(self, "peak_hold_curve"):
            ph_l = QtGui.QColor(c2)
            ph_l.setAlpha(140)
            self.peak_hold_curve.setPen(pg.mkPen(ph_l, width=1.0))
            ph_r = QtGui.QColor(c2)
            ph_r.setAlpha(140)
            self.peak_hold_curve_r.setPen(pg.mkPen(ph_r, width=1.0))

        # Многоканальные кривые
        h, s, v, a = c.getHsv()
        for i, mc in enumerate(self.multi_curves):
            factor = 1.0 - (i / max(self._n_ch - 1, 1)) * 0.6
            v_mod = int(v * factor)
            col = QtGui.QColor.fromHsv(h, s, max(40, v_mod))

            if self._show_gradient:
                q = QtGui.QColor(col)
                q.setAlpha(60)
                lo = QtGui.QColor(col).lighter(130)
                mc.setPen(make_gradient_pen(q, lo, 1.3))
            else:
                mc.setPen(pg.mkPen(col, width=1.3))

            # Пик-холд для этого канала берет тот же цвет, что и в стерео/моно
            if i < len(self.multi_peak_curves):
                ph_col = QtGui.QColor(c2)
                ph_col.setAlpha(140)
                self.multi_peak_curves[i].setPen(pg.mkPen(ph_col, width=1.0))

    def _apply_theme_color(self, color: QtGui.QColor):
        c = QtGui.QColor(color)
        c.setAlpha(255)
        self._current_color = c
        self._rebuild_pens()
        self.glow_renderer.set_color(color)
        c2 = _stereo_r_color(c.name())
        self.glow_renderer_r.set_color(c2)
        self.logo.set_accent(c)
        self._update_freeze_icon()
        self.trigger.set_accent(c)
        self.glow.set_accent(c)
        if hasattr(self, "fade_sw"):
            self.fade_sw.set_accent(c)
        for attr in (
            "gradient_sw",
            "peak_hold_sw",
            "bpm_sw",
            "smooth_sw",
            "edge_clip_sw",
        ):
            w = getattr(self, attr, None)
            if w is not None:
                w.set_accent(c)

        if self._solo:
            self.card.setStyleSheet(self._solo_card_qss())
            self._style_hud()

    def _on_theme_color(self, color: QtGui.QColor):
        if not isinstance(color, QtGui.QColor):
            return
        self.accent = color.name()
        self.setStyleSheet(make_qss(self.accent))
        self._apply_theme_color(color)
        self.plot.setBackground(self._plot_bg())
        prev = getattr(self, "_theme_preview", None)
        if prev is not None:
            prev.setStyleSheet(f"background: {self.accent}; border-radius: 6px;")
        dlg = getattr(self, "_theme_dialog", None)
        if dlg is not None and dlg.isVisible():
            dlg.setStyleSheet(self._theme_dialog_qss())

    def _theme_color(self, theme_id: str) -> str:
        if theme_id == "system":
            c = system_accent() or self._sys_last or THEME_HEX[DEFAULT_THEME]
            self._sys_last = c
            return c
        if theme_id == "custom":
            return str(CONFIG.get("custom_color", DEFAULT_CONFIG["custom_color"]))
        return THEME_HEX.get(theme_id, THEME_HEX[DEFAULT_THEME])

    SYS_POLL_MS = 2000

    def _sync_system_watch(self):
        if self._sys_timer is None:
            tm = QtCore.QTimer(self)
            tm.setTimerType(QtCore.Qt.TimerType.VeryCoarseTimer)
            tm.setInterval(self.SYS_POLL_MS)
            tm.timeout.connect(self._poll_system_accent)
            self._sys_timer = tm
        if self.theme_id == "system":
            if not self._sys_timer.isActive():
                self._sys_timer.start()
        else:
            self._sys_timer.stop()

    def _poll_system_accent(self):
        if self.theme_id != "system":
            return
        new = self._theme_color("system")
        if new == self._sys_applied:
            return
        self._sys_applied = new
        self._animate_theme(new)
        if getattr(self, "_theme_dialog", None):
            self._refresh_theme_dialog_texts()

    def _apply_theme(self, theme_id: str, animate: bool = True):
        btns = getattr(self, "theme_btns", None)
        if btns and theme_id in btns:
            btns[theme_id].setChecked(True)
        self.theme_id = theme_id
        target = self._theme_color(theme_id)
        self._sys_applied = target if theme_id == "system" else None
        self._sync_system_watch()
        if animate:
            self._animate_theme(target)
        else:
            self.accent = target
            self._current_color = QtGui.QColor(target)
            self.setStyleSheet(make_qss(self.accent))
            self._apply_theme_color(self._current_color)
            self.plot.setBackground(self._plot_bg())
            self._update_app_icon()
        if getattr(self, "_theme_dialog", None) and self._theme_dialog.isVisible():
            self._refresh_theme_dialog_texts()
        if not getattr(self, "_loading", False):
            CONFIG["theme"] = theme_id
            _save_config(CONFIG)

    def _build_ui(self):
        no_focus = QtCore.Qt.FocusPolicy.NoFocus
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)
        self._root_layout = root

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(14)
        self.logo = LogoMark()
        header.addWidget(self.logo, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.title_holder = QtWidgets.QWidget()
        title_box = QtWidgets.QVBoxLayout(self.title_holder)
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(1)
        title = QtWidgets.QLabel(tr("brand"))
        title.setObjectName("title")
        self.sub_lbl = QtWidgets.QLabel(tr("subtitle"))
        self.sub_lbl.setObjectName("sub")
        title_box.addWidget(title)
        title_box.addWidget(self.sub_lbl)
        header.addWidget(self.title_holder)
        header.addStretch(1)
        self.dbg_lbl = QtWidgets.QLabel("")
        self.dbg_lbl.setObjectName("dbg")
        self.dbg_lbl.setStyleSheet(
            "QLabel#dbg { color: #d29922; background: rgba(210,153,34,24);"
            " border: 1px solid rgba(210,153,34,90); border-radius: 10px;"
            " padding: 3px 10px; font-family: 'Cascadia Mono', 'Consolas', monospace;"
            " font-size: 11px; }"
        )
        self.dbg_lbl.setToolTip(
            "Debug: размер окна (логические px) · масштаб экрана · физические px · DPI\n"
            "F12 — скрыть/показать"
        )
        self.dbg_lbl.setVisible(self.SHOW_DEBUG)
        header.addWidget(self.dbg_lbl, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.pill = QtWidgets.QFrame()
        self.pill.setObjectName("pill")
        self.pill.setFixedHeight(32)
        pill_l = QtWidgets.QHBoxLayout(self.pill)
        pill_l.setContentsMargins(14, 0, 14, 0)
        pill_l.setSpacing(9)
        self.dot = QtWidgets.QLabel()
        self.dot.setObjectName("dot")
        self.dot.setFixedSize(8, 8)
        self.device_lbl = QtWidgets.QLabel("")
        self.device_lbl.setObjectName("device")
        self.device_lbl.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred
        )
        self.device_err = False
        pill_l.addWidget(self.dot, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        pill_l.addWidget(self.device_lbl, 1)
        self.pill.setMinimumWidth(220)
        self.pill.setMaximumWidth(480)
        self.pill.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )
        header.addWidget(self.pill, 1, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._header_layout = header
        if self.SHOW_HEADER:
            root.addLayout(header)
        else:
            for w in (self.logo, self.title_holder, self.pill, self.dbg_lbl):
                w.setParent(self)
                w.hide()

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(14)
        self._top_layout = top
        root.addLayout(top, 1)

        self.card = QtWidgets.QFrame()
        self.card.setObjectName("card")
        if not _USE_GL:
            self._card_shadow = StaticShadow(self.card, self)
        card_l = QtWidgets.QVBoxLayout(self.card)
        card_l.setContentsMargins(10, 14, 16, 8)
        self._card_layout = card_l
        self.card.setMinimumSize(300, 160)
        self.bottom = SpacedAxis("bottom")
        self.left = SpacedAxis("left")
        self.plot = WavePlotWidget(axisItems={"bottom": self.bottom, "left": self.left})
        self.plot.setBackground(self._plot_bg())
        self.plot.setStyleSheet("background: transparent;")
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setToolTip(tr("solo_tip"))
        pi = self.plot.getPlotItem()
        pi.setMenuEnabled(False)
        pi.hideButtons()
        pi.showGrid(x=True, y=True, alpha=0.07)
        try:
            pi.getViewBox().disableAutoRange()
        except Exception:
            pass
        for ax in (self.bottom, self.left):
            ax.setPen(pg.mkPen("#262c34"))
            ax.setTextPen(pg.mkPen("#6e7681"))
            ax.setTickFont(QtGui.QFont("Segoe UI", 8))
        self.left.setWidth(48)

        self.zero_line = pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen((255, 255, 255, 28), width=1)
        )
        self.zero_line.setZValue(-100)
        self.plot.addItem(self.zero_line)

        limit_pen = pg.mkPen(
            (255, 255, 255, 45), width=1, style=QtCore.Qt.PenStyle.DashLine
        )
        self.lim_hi = pg.InfiniteLine(pos=1, angle=0, pen=limit_pen)
        self.lim_lo = pg.InfiniteLine(pos=-1, angle=0, pen=limit_pen)
        self.lim_hi.setZValue(-100)
        self.lim_lo.setZValue(-100)
        self.plot.addItem(self.lim_hi)
        self.plot.addItem(self.lim_lo)

        sec_color = _stereo_r_color(self.accent)
        sec_color.setAlpha(140)
        self.peak_hold_curve = pg.PlotCurveItem(pen=pg.mkPen(sec_color, width=1.0))
        self.peak_hold_curve.setZValue(-0.7)
        self.peak_hold_curve.setVisible(self._show_peak_hold)
        self.plot.addItem(self.peak_hold_curve)
        self.peak_hold_curve_r = pg.PlotCurveItem(pen=pg.mkPen(sec_color, width=1.0))
        self.peak_hold_curve_r.setZValue(-0.75)
        self.peak_hold_curve_r.setVisible(self._show_peak_hold and self._n_ch == 2)
        self.plot.addItem(self.peak_hold_curve_r)

        self.curve = GradientCurveItem()
        self.curve.setZValue(0)
        self.plot.addItem(self.curve)
        self.curve_r = GradientCurveItem()
        self.curve_r.setZValue(-0.5)
        self.curve_r.setVisible(self._n_ch == 2)
        self.plot.addItem(self.curve_r)

        self.glow_renderer = GlowRenderer(pi)
        self.glow_renderer_r = GlowRenderer(pi)
        if CONFIG.get("glow", False) and self._n_ch == 2:
            self.glow_renderer_r.set_enabled(True)

        card_l.addWidget(self.plot)
        top.addWidget(self.card, 1)
        self.plot.installEventFilter(self)
        try:
            self.plot.viewport().installEventFilter(self)
        except Exception:
            pass
        self.card.installEventFilter(self)
        self._build_solo_hud()

        self.freeze_btn = QtWidgets.QPushButton(self._freeze_text(False))
        self.freeze_btn.setObjectName("primary")
        self.freeze_btn.setCheckable(True)
        self.freeze_btn.setFocusPolicy(no_focus)
        self.freeze_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.freeze_btn.setFixedSize(140, self.CTRL_H)
        self.freeze_btn.setIconSize(QtCore.QSize(16, 16))
        self.freeze_btn.toggled.connect(self.on_freeze)

        self.theme_btn = QtWidgets.QPushButton(tr("theme"))
        self.theme_btn.setFocusPolicy(no_focus)
        self.theme_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.theme_btn.setFixedSize((self.COL_W - 12) // 2, self.CTRL_H)
        self.theme_btn.clicked.connect(self._open_theme_dialog)
        self.cap_theme = None

        self.settings_btn = QtWidgets.QPushButton(tr("settings"))
        self.settings_btn.setFocusPolicy(no_focus)
        self.settings_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.settings_btn.setFixedSize((self.COL_W - 12) // 2, self.CTRL_H)
        self.settings_btn.setToolTip(tr("settings_tip"))
        self.settings_btn.clicked.connect(self._open_settings_dialog)
        self.cap_settings = None

        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.setFixedHeight(self.CTRL_H)
        self.device_combo.setFocusPolicy(no_focus)
        self.device_combo.currentIndexChanged.connect(self._on_device_change)
        self._style_combo_popup(self.device_combo)
        holder, self.cap_stream = self._caption_holder("stream", self.device_combo)
        holder.setFixedWidth(self.COL_W)
        self.stream_holder = holder

        self.trigger = ToggleSwitch(tr("stabilization"), self)
        self.trigger.setChecked(CONFIG.get("stabilization", True))
        self.trigger.setToolTip(tr("stabilization_tip"))
        self.trigger.toggled.connect(self._on_trigger_toggle)
        self.trigger.hide()

        self.glow = ToggleSwitch(tr("glow"), self)
        self.glow.setChecked(CONFIG.get("glow", False))
        self.glow.toggled.connect(self._on_glow_toggle)
        self.glow.hide()

        self.fade_sw = ToggleSwitch(tr("fade"), self)
        self.fade_sw.setChecked(self._fade_on)
        self.fade_sw.setToolTip(tr("fade_tip"))
        self.fade_sw.toggled.connect(self._on_fade_toggle)
        self.fade_sw.hide()

        self.gradient_sw = ToggleSwitch("", self)
        self.gradient_sw.setChecked(self._show_gradient)
        self.gradient_sw.toggled.connect(self._on_gradient_toggle)
        self.gradient_sw.hide()

        self.peak_hold_sw = ToggleSwitch("", self)
        self.peak_hold_sw.setChecked(self._show_peak_hold)
        self.peak_hold_sw.toggled.connect(self._on_peak_hold_toggle)
        self.peak_hold_sw.hide()

        self.bpm_sw = ToggleSwitch("", self)
        self.bpm_sw.setChecked(self._bpm_on)
        self.bpm_sw.toggled.connect(self._on_bpm_toggle)
        self.bpm_sw.hide()

        self.smooth_sw = ToggleSwitch("", self)
        self.smooth_sw.setChecked(self._smooth_stab)
        self.smooth_sw.setEnabled(self.trigger.isChecked())
        self.smooth_sw.toggled.connect(self._on_smooth_toggle)
        self.trigger.toggled.connect(self.smooth_sw.setEnabled)
        self.smooth_sw.hide()

        self.edge_clip_sw = ToggleSwitch("", self)
        self.edge_clip_sw.setChecked(self._edge_clip_on)
        self.edge_clip_sw.setEnabled(self.trigger.isChecked())
        self.edge_clip_sw.toggled.connect(self._on_edge_clip_toggle)
        self.trigger.toggled.connect(self.edge_clip_sw.setEnabled)
        self.edge_clip_sw.hide()

        self.band_combo = QtWidgets.QComboBox(self)
        self.band_combo.addItem(tr("band_off"), "off")
        self.band_combo.addItem(tr("band_low"), "low")
        self.band_combo.addItem(tr("band_mid"), "mid")
        self.band_combo.addItem(tr("band_high"), "high")
        try:
            _bi = ["off", "low", "mid", "high"].index(self._band_kind)
        except ValueError:
            _bi = 0
        self.band_combo.setCurrentIndex(_bi)
        self.band_combo.setFixedHeight(self.CTRL_H)
        self.band_combo.setFocusPolicy(no_focus)
        self.band_combo.setToolTip(tr("bandpass_tip"))
        self.band_combo.currentIndexChanged.connect(self._on_band_change)
        self._style_combo_popup(self.band_combo)
        self.band_combo.hide()

        self.ch_layout_combo = QtWidgets.QComboBox(self)
        for key, _n, label in CHANNEL_LAYOUTS:
            self.ch_layout_combo.addItem(tr(label), key)
        try:
            _idx = [k for k, _, _ in CHANNEL_LAYOUTS].index(self.ch_layout)
        except ValueError:
            _idx = 0
        self.ch_layout_combo.setCurrentIndex(_idx)
        self.ch_layout_combo.setFixedHeight(self.CTRL_H)
        self.ch_layout_combo.setFocusPolicy(no_focus)
        self.ch_layout_combo.currentIndexChanged.connect(self._on_layout_change)
        self._style_combo_popup(self.ch_layout_combo)
        self.ch_layout_combo.hide()
        self.cap_channels = None

        n_idx = (
            SAMPLE_OPTIONS.index(CONFIG.get("n", 4096))
            if CONFIG.get("n") in SAMPLE_OPTIONS
            else SAMPLE_OPTIONS.index(4096)
        )
        self.n_combo = QtWidgets.QComboBox(self)
        self.n_combo.addItems([f"{v} {tr('samples')}" for v in SAMPLE_OPTIONS])
        self.n_combo.setCurrentIndex(n_idx)
        self.n_combo.setFixedHeight(self.CTRL_H)
        self.n_combo.setFocusPolicy(no_focus)
        self._style_combo_popup(self.n_combo)
        self.n_combo.currentIndexChanged.connect(
            lambda i: self.set_n(SAMPLE_OPTIONS[i])
        )
        self.n_combo.hide()

        g_cfg = CONFIG.get("gain", 1)
        g_idx = GAIN_OPTIONS.index(g_cfg) if g_cfg in GAIN_OPTIONS else 0
        self.gain_combo = Segmented([(f"×{g}", g) for g in GAIN_OPTIONS], g_idx, self)
        self.gain_combo.setFixedHeight(self.CTRL_H)
        self.gain_combo.currentIndexChanged.connect(
            lambda i: self.set_gain(GAIN_OPTIONS[i])
        )
        self.gain_combo.hide()

        self.lang_combo = QtWidgets.QComboBox(self)
        self.lang_combo.setMinimumWidth(140)
        self.lang_combo.setFixedHeight(self.CTRL_H)
        self.lang_combo.setFocusPolicy(no_focus)
        for label, code in LANGUAGES:
            self.lang_combo.addItem(label, code)
        for i in range(self.lang_combo.count()):
            if self.lang_combo.itemData(i) == LANG["current"]:
                self.lang_combo.setCurrentIndex(i)
                break
        self.lang_combo.currentIndexChanged.connect(self._on_language_change)
        self._style_combo_popup(self.lang_combo)
        self.lang_combo.hide()

        fps_idx = (
            FPS_OPTIONS.index(self._fps_setting)
            if self._fps_setting in FPS_OPTIONS
            else 0
        )
        self.fps_combo = QtWidgets.QComboBox(self)
        self.fps_combo.setMinimumWidth(160)
        self.fps_combo.setFixedHeight(self.CTRL_H)
        self.fps_combo.setFocusPolicy(no_focus)
        for v in FPS_OPTIONS:
            self.fps_combo.addItem(self._fps_label(v), v)
        self.fps_combo.setCurrentIndex(fps_idx)
        self.fps_combo.setToolTip(tr("fps_tip"))
        self.fps_combo.currentIndexChanged.connect(self._on_fps_change)
        self._style_combo_popup(self.fps_combo)
        self.fps_combo.hide()
        self.cap_fps = None

        self.cap_window = None
        self.cap_gain = None
        self.cap_lang = None
        self._settings_dialog = None

        col = QtWidgets.QVBoxLayout()
        col.setSpacing(12)
        self._stats_layout = col
        self.peak_tile = StatTile(self.COL_W)
        self.bpm_tile = StatTile(self.COL_W)
        self.rms_hz_tile = PairTile(self.COL_W)
        self.rms_tile = self.rms_hz_tile.left
        self.hz_tile = self.rms_hz_tile.right
        self.info_tile = StatTile(self.COL_W)
        self.fps_tile = FpsFreezeTile(self.COL_W, self.freeze_btn)
        self.hz_tile.setToolTip(tr("hz_tip"))
        self.bpm_tile.setToolTip(tr("bpm_tip"))
        for tile in (
            self.peak_tile,
            self.bpm_tile,
            self.rms_hz_tile,
            self.info_tile,
            self.fps_tile,
        ):
            tile.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
            )
            tile.setFixedHeight(54)
            col.addWidget(tile)
        col.addSpacing(6)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.addWidget(self.theme_btn)
        btn_row.addWidget(self.settings_btn)
        col.addLayout(btn_row)
        col.addWidget(self.stream_holder)
        col.addStretch(1)
        top.addLayout(col, 0)

        self.bknd_tile = StatTile(140)
        self.bknd_tile.setParent(self)
        self.bknd_tile.hide()

    @staticmethod
    def _style_combo_popup(combo):
        view = QtWidgets.QListView()
        view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        combo.setView(view)
        try:
            popup = combo.view().window()
            if popup is None:
                return
            popup.setWindowFlags(
                popup.windowFlags()
                | QtCore.Qt.WindowType.FramelessWindowHint
                | QtCore.Qt.WindowType.NoDropShadowWindowHint
            )
            popup.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        except Exception:
            pass

    def _caption_holder(self, caption_key, widget):
        cap = QtWidgets.QLabel(cap_text(caption_key))
        cap.setObjectName("cap")
        cap.setFixedHeight(12)
        holder = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addWidget(cap)
        box.addWidget(widget)
        return holder, cap

    def _settings_dialog_qss(self) -> str:
        a = self.accent
        top = _mix_dark(a, 0.040)
        bot = _mix_dark(a, 0.014)
        return f"""
            QDialog {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {top}, stop:1 {bot}); }}
            QLabel {{ background: transparent; color: #c9d1d9; }}
            QLabel#stTitle {{ color: #f0f6fc; font-size: 16px; font-weight: 700; }}
            QLabel#stCap {{ color: #6e7681; font-size: 10px; font-weight: 600; letter-spacing: 1px; padding: 2px 4px 6px 4px; }}
            QLabel#stRowLabel {{ color: #e6edf3; font-size: 13px; font-weight: 500; }}
            QFrame#stCard {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; }}
            QFrame#stSep {{ background: rgba(255,255,255,0.06); max-height: 1px; min-height: 1px; border: none; }}
            QPushButton#stDone {{ background: {a}; color: #0d1117; border: none; border-radius: 10px; padding: 10px 28px; font-weight: 700; font-size: 13px; }}
            QPushButton#stDone:hover {{ background: {_rgba(a, 0.85)}; }}
            QComboBox {{ background: rgba(255,255,255,0.08); color: #c9d1d9; border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; padding: 6px 12px; min-width: 120px; min-height: 28px; }}
            QComboBox:hover {{ border-color: {a}; }}
            QComboBox::drop-down {{ border: none; width: 22px; }}
            QComboBox QAbstractItemView {{ background: #161b22; color: #c9d1d9; border: 1px solid #30363d; border-radius: 10px; selection-background-color: {a}; selection-color: #0d1117; }}
            QFrame#seg {{ background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; }}
            QPushButton#segbtn {{ background: transparent; color: #8b949e; border: none; border-radius: 8px; padding: 6px 12px; font-weight: 500; font-family: 'Segoe UI'; font-size: 12px; min-width: 0px; }}
            QPushButton#segbtn:hover {{ background: rgba(255,255,255,0.10); color: #ffffff; }}
            QPushButton#segbtn:checked {{ background: {a}; color: #0d1117; font-weight: 700; }}
        """

    def _settings_row(self, label_key, widget, store_cap=None):
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(14, 10, 14, 10)
        h.setSpacing(12)
        lbl = QtWidgets.QLabel(tr(label_key))
        lbl.setObjectName("stRowLabel")
        h.addWidget(lbl, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        h.addStretch(1)
        h.addWidget(widget, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        if store_cap is not None:
            setattr(self, store_cap, lbl)
        return row

    def _settings_sep(self):
        line = QtWidgets.QFrame()
        line.setObjectName("stSep")
        line.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        return line

    def _build_settings_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(tr("settings"))
        dlg.setModal(True)
        dlg.setFixedWidth(820)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(24, 22, 24, 20)
        lay.setSpacing(14)

        title = QtWidgets.QLabel(tr("settings"))
        title.setObjectName("stTitle")
        lay.addWidget(title)

        def _section(title_key):
            wrap = QtWidgets.QVBoxLayout()
            wrap.setSpacing(6)
            cap = QtWidgets.QLabel(tr(title_key))
            cap.setObjectName("stCap")
            wrap.addWidget(cap)
            card = QtWidgets.QFrame()
            card.setObjectName("stCard")
            card_l = QtWidgets.QVBoxLayout(card)
            card_l.setContentsMargins(0, 4, 0, 4)
            card_l.setSpacing(0)
            wrap.addWidget(card)
            return wrap, card_l, cap

        w1, c1, self._st_cap_general = _section("settings_general")
        c1.addWidget(self._settings_row("language", self.lang_combo, "cap_lang"))
        c1.addWidget(self._settings_sep())
        c1.addWidget(
            self._settings_row("ch_layout", self.ch_layout_combo, "cap_channels")
        )
        c1.addWidget(self._settings_sep())
        c1.addWidget(self._settings_row("fps", self.fps_combo, "cap_fps"))

        w2, c2, self._st_cap_signal = _section("settings_signal")
        c2.addWidget(self._settings_row("window", self.n_combo, "cap_window"))
        c2.addWidget(self._settings_sep())
        c2.addWidget(self._settings_row("gain", self.gain_combo, "cap_gain"))
        c2.addWidget(self._settings_sep())
        c2.addWidget(self._settings_row("bandpass", self.band_combo))

        w3, c3, self._st_cap_effects = _section("settings_effects")
        c3.addWidget(self._settings_row("stabilization", self.trigger, "_cap_stab"))
        c3.addWidget(self._settings_sep())
        c3.addWidget(self._settings_row("smooth_stab", self.smooth_sw))
        c3.addWidget(self._settings_sep())
        c3.addWidget(self._settings_row("edge_clip", self.edge_clip_sw, "_cap_edge"))
        c3.addWidget(self._settings_sep())
        c3.addWidget(self._settings_row("glow", self.glow, "_cap_glow"))
        c3.addWidget(self._settings_sep())
        c3.addWidget(self._settings_row("fade", self.fade_sw, "_cap_fade"))
        c3.addWidget(self._settings_sep())
        c3.addWidget(self._settings_row("gradient", self.gradient_sw))
        c3.addWidget(self._settings_sep())
        c3.addWidget(self._settings_row("peak_hold", self.peak_hold_sw))
        c3.addWidget(self._settings_sep())
        c3.addWidget(self._settings_row("bpm", self.bpm_sw))

        cols = QtWidgets.QHBoxLayout()
        cols.setSpacing(24)

        col_left = QtWidgets.QVBoxLayout()
        col_left.setSpacing(14)
        col_left.addLayout(w1)
        col_left.addLayout(w2)
        col_left.addStretch(1)

        col_right = QtWidgets.QVBoxLayout()
        col_right.setSpacing(14)
        col_right.addLayout(w3)
        col_right.addStretch(1)

        cols.addLayout(col_left, 1)
        cols.addLayout(col_right, 1)
        lay.addLayout(cols)

        for w in (
            self.lang_combo,
            self.n_combo,
            self.gain_combo,
            self.ch_layout_combo,
            self.fps_combo,
            self.trigger,
            self.glow,
            self.fade_sw,
            self.band_combo,
            self.gradient_sw,
            self.peak_hold_sw,
            self.bpm_sw,
            self.smooth_sw,
            self.edge_clip_sw,
        ):
            w.setVisible(True)
            w.show()

        for w in (self.lang_combo, self.n_combo, self.fps_combo, self.band_combo):
            w.setFixedWidth(190)

        self.ch_layout_combo.setMinimumWidth(190)
        self.gain_combo.setMinimumWidth(210)

        for sw in (
            self.trigger,
            self.glow,
            self.fade_sw,
            self.gradient_sw,
            self.peak_hold_sw,
            self.bpm_sw,
            self.smooth_sw,
            self.edge_clip_sw,
        ):
            sw.setText("")
            sw.setMinimumWidth(sw.TW + 4)
            sw.setFixedHeight(26)

        close_btn = QtWidgets.QPushButton(tr("theme_close"))
        close_btn.setObjectName("stDone")
        close_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)

        dlg.setStyleSheet(self._settings_dialog_qss())
        self._settings_dialog = dlg
        self._settings_dlg_title = title
        self._settings_dlg_close = close_btn

    def _open_settings_dialog(self):
        if not getattr(self, "_settings_dialog", None):
            self._build_settings_dialog()
        for w in (
            self.lang_combo,
            self.n_combo,
            self.gain_combo,
            self.ch_layout_combo,
            self.fps_combo,
            self.trigger,
            self.glow,
            self.fade_sw,
            self.band_combo,
            self.gradient_sw,
            self.peak_hold_sw,
            self.bpm_sw,
            self.smooth_sw,
            self.edge_clip_sw,
        ):
            w.setVisible(True)
            w.show()
        self._settings_dialog.setWindowTitle(tr("settings"))
        self._settings_dlg_title.setText(tr("settings"))
        self._settings_dlg_close.setText(tr("theme_close"))
        if getattr(self, "_st_cap_general", None):
            self._st_cap_general.setText(tr("settings_general"))
        if getattr(self, "_st_cap_signal", None):
            self._st_cap_signal.setText(tr("settings_signal"))
        if getattr(self, "_st_cap_effects", None):
            self._st_cap_effects.setText(tr("settings_effects"))
        if self.cap_lang:
            self.cap_lang.setText(tr("language"))
        if self.cap_channels:
            self.cap_channels.setText(tr("ch_layout"))
        if self.cap_window:
            self.cap_window.setText(tr("window"))
        if self.cap_gain:
            self.cap_gain.setText(tr("gain"))
        if getattr(self, "_cap_stab", None):
            self._cap_stab.setText(tr("stabilization"))
        if getattr(self, "_cap_glow", None):
            self._cap_glow.setText(tr("glow"))
        if getattr(self, "_cap_fade", None):
            self._cap_fade.setText(tr("fade"))
        if getattr(self, "_cap_edge", None):
            self._cap_edge.setText(tr("edge_clip"))
        for sw in (
            self.trigger,
            self.glow,
            self.fade_sw,
            self.gradient_sw,
            self.peak_hold_sw,
            self.bpm_sw,
            self.smooth_sw,
            self.edge_clip_sw,
        ):
            sw.setText("")
        self.trigger.setToolTip(tr("stabilization_tip"))
        self.fade_sw.setToolTip(tr("fade_tip"))
        self.edge_clip_sw.setToolTip(tr("edge_clip_tip"))
        self.band_combo.setToolTip(tr("bandpass_tip"))
        self.bpm_tile.setToolTip(tr("bpm_tip"))
        self.band_combo.setItemText(0, tr("band_off"))
        self.band_combo.setItemText(1, tr("band_low"))
        self.band_combo.setItemText(2, tr("band_mid"))
        self.band_combo.setItemText(3, tr("band_high"))

        if self.ch_layout_combo:
            for i, (_key, _n, label) in enumerate(CHANNEL_LAYOUTS):
                self.ch_layout_combo.setItemText(i, tr(label))

        for sw in (
            self.trigger,
            self.glow,
            self.fade_sw,
            self.gradient_sw,
            self.peak_hold_sw,
            self.bpm_sw,
            self.smooth_sw,
            self.edge_clip_sw,
        ):
            sw.set_accent(self.accent)
        idx = self.n_combo.currentIndex()
        self.n_combo.blockSignals(True)
        self.n_combo.clear()
        self.n_combo.addItems([f"{v} {tr('samples')}" for v in SAMPLE_OPTIONS])
        self.n_combo.setCurrentIndex(idx)
        self.n_combo.blockSignals(False)
        self._refresh_fps_combo()
        if self.cap_fps:
            self.cap_fps.setText(tr("fps"))
        self._settings_dialog.setStyleSheet(self._settings_dialog_qss())
        self._settings_dialog.adjustSize()
        self._settings_dialog.exec()

    def _theme_dialog_qss(self) -> str:
        a = self.accent
        top = _mix_dark(a, 0.040)
        bot = _mix_dark(a, 0.014)
        return f"""
            QDialog {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {top}, stop:1 {bot}); }}
            QLabel {{ background: transparent; color: #c9d1d9; }}
            QLabel#thTitle {{ color: #f0f6fc; font-size: 15px; font-weight: 700; }}
            QLabel#thCap {{ color: #6e7681; font-size: 10px; font-weight: 600; letter-spacing: 1px; }}
            QLabel#thSub {{ color: #8b949e; font-size: 12px; }}
            QFrame#thCard {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; }}
            QPushButton#thCustom {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.10); border-radius: 12px; color: #c9d1d9; text-align: left; padding: 10px 14px; font-size: 12px; }}
            QPushButton#thCustom:hover {{ background: rgba(255,255,255,0.08); border-color: {a}; }}
            QPushButton#thDone {{ background: {a}; color: #0d1117; border: none; border-radius: 10px; padding: 10px 28px; font-weight: 700; font-size: 13px; }}
            QPushButton#thDone:hover {{ background: {_rgba(a, 0.85)}; }}
            QFrame#thPreview {{ background: {a}; border-radius: 6px; }}
        """

    def _build_theme_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(tr("theme"))
        dlg.setModal(True)
        dlg.setFixedWidth(360)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(16)

        head = QtWidgets.QHBoxLayout()
        head.setSpacing(12)
        title = QtWidgets.QLabel(tr("theme"))
        title.setObjectName("thTitle")
        head.addWidget(title)
        head.addStretch(1)
        self._theme_preview = QtWidgets.QFrame()
        self._theme_preview.setObjectName("thPreview")
        self._theme_preview.setFixedSize(28, 28)
        head.addWidget(self._theme_preview)
        lay.addLayout(head)

        card = QtWidgets.QFrame()
        card.setObjectName("thCard")
        card_l = QtWidgets.QVBoxLayout(card)
        card_l.setContentsMargins(16, 14, 16, 16)
        card_l.setSpacing(12)

        cap = QtWidgets.QLabel(tr("theme_presets"))
        cap.setObjectName("thCap")
        card_l.addWidget(cap)

        grid_w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        group = QtWidgets.QButtonGroup(dlg)
        group.setExclusive(True)
        self.theme_btns = {}
        cols = 6
        for i, (tid, color) in enumerate(THEMES):
            cell = QtWidgets.QWidget()
            cell_l = QtWidgets.QVBoxLayout(cell)
            cell_l.setContentsMargins(0, 0, 0, 0)
            cell_l.setSpacing(4)
            cell_l.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            b = Swatch(color)
            b.setToolTip(tr(f"theme_{tid}"))
            b.setFixedSize(34, 34)
            b.clicked.connect(lambda _=False, t=tid: self._apply_theme(t))
            group.addButton(b)
            name = QtWidgets.QLabel(tr(f"theme_{tid}"))
            name.setObjectName("thSub")
            name.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            name.setStyleSheet("font-size: 10px; color: #6e7681;")
            cell_l.addWidget(b, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
            cell_l.addWidget(name)
            grid.addWidget(cell, i // cols, i % cols)
            self.theme_btns[tid] = b
            b._name_lbl = name

        card_l.addWidget(grid_w)
        lay.addWidget(card)

        sys_hex = system_accent()
        self._theme_system_btn = QtWidgets.QPushButton()
        self._theme_system_btn.setObjectName("thCustom")
        self._theme_system_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._theme_system_btn.clicked.connect(lambda: self._apply_theme("system"))
        system_inner = QtWidgets.QHBoxLayout(self._theme_system_btn)
        system_inner.setContentsMargins(4, 2, 8, 2)
        system_inner.setSpacing(12)
        self.system_swatch = Swatch(sys_hex or THEME_HEX[DEFAULT_THEME])
        self.system_swatch.setFixedSize(30, 30)
        self.system_swatch.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        system_inner.addWidget(self.system_swatch)
        system_txt_box = QtWidgets.QVBoxLayout()
        system_txt_box.setSpacing(1)
        self._theme_system_title = QtWidgets.QLabel(tr("theme_system"))
        self._theme_system_title.setStyleSheet(
            "font-size: 12px; font-weight: 600; color: #e6edf3; background: transparent;"
        )
        self._theme_system_hex = QtWidgets.QLabel(sys_hex or tr("theme_system_na"))
        self._theme_system_hex.setStyleSheet(
            "font-size: 11px; color: #8b949e; background: transparent; font-family: 'Cascadia Mono', 'Consolas', monospace;"
        )
        system_txt_box.addWidget(self._theme_system_title)
        system_txt_box.addWidget(self._theme_system_hex)
        system_inner.addLayout(system_txt_box)
        system_inner.addStretch(1)
        group.addButton(self.system_swatch)
        self.theme_btns["system"] = self.system_swatch
        self._theme_system_btn.setEnabled(sys_hex is not None)
        lay.addWidget(self._theme_system_btn)

        self._theme_custom_btn = QtWidgets.QPushButton()
        self._theme_custom_btn.setObjectName("thCustom")
        self._theme_custom_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._theme_custom_btn.clicked.connect(self._on_custom_theme)
        custom_inner = QtWidgets.QHBoxLayout(self._theme_custom_btn)
        custom_inner.setContentsMargins(4, 2, 8, 2)
        custom_inner.setSpacing(12)
        cc = CONFIG.get("custom_color", DEFAULT_CONFIG["custom_color"])
        self.custom_swatch = Swatch(cc, custom=True)
        self.custom_swatch.setFixedSize(30, 30)
        self.custom_swatch.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        custom_inner.addWidget(self.custom_swatch)
        custom_txt_box = QtWidgets.QVBoxLayout()
        custom_txt_box.setSpacing(1)
        self._theme_custom_title = QtWidgets.QLabel(tr("theme_custom"))
        self._theme_custom_title.setStyleSheet(
            "font-size: 12px; font-weight: 600; color: #e6edf3; background: transparent;"
        )
        self._theme_custom_hex = QtWidgets.QLabel(cc)
        self._theme_custom_hex.setStyleSheet(
            "font-size: 11px; color: #8b949e; background: transparent; font-family: 'Cascadia Mono', 'Consolas', monospace;"
        )
        custom_txt_box.addWidget(self._theme_custom_title)
        custom_txt_box.addWidget(self._theme_custom_hex)
        custom_inner.addLayout(custom_txt_box)
        custom_inner.addStretch(1)
        group.addButton(self.custom_swatch)
        self.theme_btns["custom"] = self.custom_swatch
        lay.addWidget(self._theme_custom_btn)

        self._theme_dlg_label = QtWidgets.QLabel("")
        self._theme_dlg_label.setObjectName("thSub")
        self._theme_dlg_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._theme_dlg_label)

        close_btn = QtWidgets.QPushButton(tr("theme_close"))
        close_btn.setObjectName("thDone")
        close_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)

        dlg.setStyleSheet(self._theme_dialog_qss())
        self._theme_dialog = dlg
        self._theme_dlg_title = title
        self._theme_dlg_close = close_btn

    def _refresh_theme_dialog_texts(self):
        if not getattr(self, "_theme_dialog", None):
            return
        self._theme_dialog.setWindowTitle(tr("theme"))
        self._theme_dlg_title.setText(tr("theme"))
        self._theme_dlg_close.setText(tr("theme_close"))
        self._theme_custom_title.setText(tr("theme_custom"))
        self._theme_system_title.setText(tr("theme_system"))
        sys_hex = system_accent()
        self._theme_system_btn.setEnabled(sys_hex is not None)
        self._theme_system_btn.setToolTip(tr("theme_system_tip"))
        self._theme_system_hex.setText(sys_hex or tr("theme_system_na"))
        if sys_hex:
            self.system_swatch.set_color(sys_hex)
        for tid, btn in self.theme_btns.items():
            if tid == "custom":
                btn.setToolTip(tr("theme_custom_tip"))
            elif tid == "system":
                btn.setToolTip(tr("theme_system_tip"))
            else:
                btn.setToolTip(tr(f"theme_{tid}"))
                if hasattr(btn, "_name_lbl"):
                    btn._name_lbl.setText(tr(f"theme_{tid}"))
        if self.theme_id in self.theme_btns:
            self.theme_btns[self.theme_id].setChecked(True)
        if self.theme_id == "custom":
            name = tr("theme_custom")
            hex_c = CONFIG.get("custom_color", "")
        elif self.theme_id == "system":
            name = tr("theme_system")
            hex_c = self._theme_color("system")
        else:
            name = tr(f"theme_{self.theme_id}")
            hex_c = THEME_HEX.get(self.theme_id, "")
        self._theme_dlg_label.setText(f"{name}  ·  {hex_c}")
        self._theme_custom_hex.setText(
            str(CONFIG.get("custom_color", DEFAULT_CONFIG["custom_color"]))
        )
        chip = self._theme_color(self.theme_id)
        self._theme_preview.setStyleSheet(f"background: {chip}; border-radius: 6px;")
        self._theme_dialog.setStyleSheet(self._theme_dialog_qss())

    def _open_theme_dialog(self):
        if not getattr(self, "_theme_dialog", None):
            self._build_theme_dialog()
        self._refresh_theme_dialog_texts()
        self._theme_dialog.adjustSize()
        self._theme_dialog.exec()

    def _on_custom_theme(self):
        initial = QtGui.QColor(
            CONFIG.get("custom_color", DEFAULT_CONFIG["custom_color"])
        )
        parent = self._theme_dialog if getattr(self, "_theme_dialog", None) else self
        color = QtWidgets.QColorDialog.getColor(initial, parent, tr("theme_custom"))
        if not color.isValid():
            if self.theme_id == "custom":
                self.custom_swatch.setChecked(True)
            return
        hex_color = color.name()
        CONFIG["custom_color"] = hex_color
        self.custom_swatch.set_color(hex_color)
        self._theme_custom_hex.setText(hex_color)
        self._apply_theme("custom")
        if getattr(self, "_theme_dialog", None):
            self._refresh_theme_dialog_texts()

    def set_n(self, n):
        self.n = n
        self._alloc_scratch(n)
        self._x = np.arange(n, dtype=np.float32)
        self.last = np.zeros(n, dtype=np.float32)
        self.last_r = np.zeros(n, dtype=np.float32)
        self.prev_tpl = None
        self._crossing_smooth = None
        self._period_smooth = 0.0
        self._lock_quality = 0.0
        self._sm_bufs.clear()
        self._peak_hold.reset()
        self._peak_hold_r.reset()
        for ph in self.multi_peak_holds:
            ph.reset()
        if self._fade_on or self._fade_blend > 0.01:
            self._rebuild_demo_caches()
        self.plot.setXRange(-n * 0.012, n, padding=0)
        self.bottom.setTickSpacing(TICKS[n], TICKS[n] / 2)
        self._reapply_ranges()
        if not getattr(self, "_loading", False):
            CONFIG["n"] = n
            _save_config(CONFIG)

    def set_gain(self, g):
        self.plot.setYRange(-1.06 / g, 1.06 / g, padding=0)
        self.left.setTickSpacing(0.5 / g, 0.25 / g)
        self.lim = 1.0 / g
        self.lim_hi.setPos(self.lim)
        self.lim_lo.setPos(-self.lim)
        if not getattr(self, "_loading", False):
            CONFIG["gain"] = g
            _save_config(CONFIG)

    def on_freeze(self, on):
        self.frozen = on
        if not on:
            _ui_gate[0] = True
        self.freeze_btn.setText(self._freeze_text(on))
        self._update_freeze_icon()
        self.solo_freeze.setChecked(on)

    def _on_trigger_toggle(self, on):
        self.prev_tpl = None
        self._crossing_smooth = None
        self._period_smooth = 0.0
        self._lock_quality = 0.0
        CONFIG["stabilization"] = bool(on)
        _save_config(CONFIG)

    def _on_glow_toggle(self, on: bool):
        self.glow_renderer.set_enabled(on)
        self.glow_renderer_r.set_enabled(on and self._n_ch == 2)
        CONFIG["glow"] = bool(on)
        _save_config(CONFIG)

    def _on_fade_toggle(self, on: bool):
        self._fade_on = bool(on)
        self._fade_target = 1.0 if self._fade_on else 0.0
        if self._fade_on:
            self._rebuild_demo_caches()
        CONFIG["fade"] = self._fade_on
        _save_config(CONFIG)

    def _on_gradient_toggle(self, on):
        self._show_gradient = bool(on)
        self._rebuild_pens()
        CONFIG["gradient"] = self._show_gradient
        _save_config(CONFIG)

    def _on_peak_hold_toggle(self, on):
        self._show_peak_hold = bool(on)
        self.peak_hold_curve.setVisible(self._show_peak_hold and self._n_ch <= 2)
        self.peak_hold_curve_r.setVisible(self._show_peak_hold and self._n_ch == 2)
        if not self._show_peak_hold:
            self.peak_hold_curve.clear()
            self.peak_hold_curve_r.clear()
            for c in self.multi_peak_curves:
                c.setVisible(False)
        CONFIG["peak_hold"] = self._show_peak_hold
        _save_config(CONFIG)

    def _on_smooth_toggle(self, on):
        self._smooth_stab = bool(on)
        self._sm_bufs.clear()
        self._hold_samples = 0
        CONFIG["smooth_stab"] = self._smooth_stab
        _save_config(CONFIG)

    def _on_edge_clip_toggle(self, on):
        self._edge_clip_on = bool(on)
        self._sm_bufs.clear()
        self._sm_scale.clear()
        CONFIG["edge_clip"] = self._edge_clip_on
        _save_config(CONFIG)

    def _on_bpm_toggle(self, on):
        self._bpm_on = bool(on)
        CONFIG["bpm_on"] = self._bpm_on
        _save_config(CONFIG)

    def _on_band_change(self, idx):
        if idx < 0:
            return
        key = self.band_combo.itemData(idx)
        if not key:
            return
        self._band_kind = key
        self._filters = [make_filter(key) for _ in range(MAX_CH)]
        CONFIG["bandpass"] = key
        _save_config(CONFIG)

    def _on_layout_change(self, idx):
        if idx < 0:
            return
        key = self.ch_layout_combo.itemData(idx)
        if key:
            self._apply_ch_layout(key)

    def _apply_ch_layout(self, layout: str):
        if layout == self.ch_layout and self._layout_applied:
            return
        self.ch_layout = layout
        self._n_ch = CH_LAYOUT_TO_N.get(layout, 1)
        self.channels = "mono" if self._n_ch == 1 else "stereo"
        set_channel_count(self._n_ch)
        self._alloc_scratch(self.n)

        for c in self.multi_curves:
            try:
                self.plot.removeItem(c)
            except Exception:
                pass
        self.multi_curves = []

        for c in self.multi_peak_curves:
            try:
                self.plot.removeItem(c)
            except Exception:
                pass
        self.multi_peak_curves = []
        self.multi_peak_holds = []

        if self._n_ch > 2:
            for i in range(self._n_ch):
                c = GradientCurveItem()
                c.setZValue(-0.1 - 0.01 * i)
                self.plot.addItem(c)
                self.multi_curves.append(c)

                ph = PeakHold(decay=0.94, smooth=0.5, window=15)
                self.multi_peak_holds.append(ph)

                ph_c = pg.PlotCurveItem(pen=None)
                ph_c.setZValue(-0.8 - 0.01 * i)
                ph_c.setVisible(self._show_peak_hold)
                self.plot.addItem(ph_c)
                self.multi_peak_curves.append(ph_c)

            self.curve.clear()
            self.curve_r.clear()
            self.curve.setVisible(False)
            self.curve_r.setVisible(False)
            self.peak_hold_curve.setVisible(False)
            self.peak_hold_curve_r.setVisible(False)
        else:
            self.curve.setVisible(True)
            self.curve_r.setVisible(self._n_ch == 2)
            self.peak_hold_curve.setVisible(self._show_peak_hold)
            self.peak_hold_curve_r.setVisible(self._show_peak_hold and self._n_ch == 2)

        self.trigger.setEnabled(True)

        if self._n_ch > 2:
            if self.glow_renderer._item is not None:
                self.glow_renderer._item.clear()
            if self.glow_renderer_r._item is not None:
                self.glow_renderer_r._item.clear()

        self.prev_tpl = None
        self._crossing_smooth = None
        self._period_smooth = 0.0
        self._lock_quality = 0.0
        self._last_total = state.get("total_samples", 0)
        self._bpm_last_total = self._last_total
        self._metrics_total = -(10**9)
        self._sm_bufs.clear()
        self._peak_hold.reset()
        self._peak_hold_r.reset()
        for ph in self.multi_peak_holds:
            ph.reset()

        for f in self._filters:
            if f is not None:
                f.reset()

        self.glow_renderer_r.set_enabled(
            self.glow_renderer.is_enabled() and self._n_ch == 2
        )

        self._layout_applied = True
        if not getattr(self, "_loading", False):
            CONFIG["ch_layout"] = layout
            _save_config(CONFIG)
        self._rebuild_pens()

    def _rebuild_demo_caches(self):
        n = self.n
        if n <= 0:
            self._demo_env_cache = None
            return
        t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
        self._demo_env_cache = _demo_envelope(t)

    def _apply_demo_state(self):
        if self._fade_on or self._fade_blend > 0.01:
            self._rebuild_demo_caches()
        else:
            self._demo_env_cache = None

    def _ui_hidden(self) -> bool:
        w = self.card if self._solo else self
        return w.isMinimized() or not w.isVisible()

    def _keepalive(self):
        if self._solo:
            t = time.perf_counter()
            if t - self._last_keepalive >= 0.1:
                self._last_keepalive = t
                self.plot.viewport().update()

    def _trigger_position(self, data, shift, need, n):
        if not self._smooth_stab:
            found = self.find_trigger(data, shift)
            self._hold_samples = 0
            return found

        force_search = shift <= 0 or self._crossing_smooth is None
        if force_search:
            self._trigger_pending_samples = 0
        else:
            self._trigger_pending_samples += shift

        trigger_interval = max(2 * CHUNK, SAMPLERATE // 60)
        if force_search or self._trigger_pending_samples >= trigger_interval:
            self._trigger_pending_samples = 0
            found = self.find_trigger(data, shift)
            if found is None:
                return self._hold_trigger(shift, need, n)
            self._hold_samples = 0
            return found

        found = self._hold_trigger(shift, need, n)
        if found is None:
            self._trigger_pending_samples = 0
            found = self.find_trigger(data, shift)
            if found is not None:
                self._hold_samples = 0
        return found

    def update_frame(self):
        if self._ui_hidden():
            if not self._was_hidden:
                QtCore.QTimer.singleShot(2500, self._trim_if_hidden)
            self._was_hidden = True
            _ui_gate[0] = False
            return

        animating = False
        if abs(self._fade_blend - self._fade_target) > 1e-4:
            animating = True
            self._fade_blend += (self._fade_target - self._fade_blend) * 0.08
            if abs(self._fade_blend - self._fade_target) < 0.002:
                self._fade_blend = self._fade_target
                if self._fade_blend < 0.01 and not self._fade_on:
                    self._demo_env_cache = None

        if self.frozen:
            _ui_gate[0] = False
            self._keepalive()
            return

        n = self.n
        stereo = self.channels == "stereo"
        multi = self._n_ch > 2
        trig_on = self.trigger.isChecked()
        vb = self.plot.getViewBox()
        width_px = float(vb.width())

        sig = (
            n,
            self.lim,
            stereo,
            trig_on,
            round(self._fade_blend, 3),
            self.glow_renderer.is_enabled(),
            self.glow_renderer_r.is_enabled(),
            round(width_px, 1),
            round(vb.height(), 1),
            self._n_ch,
            self._show_gradient,
            self._show_peak_hold,
            self._band_kind,
            self._edge_clip_on,
        )

        if (
            not animating
            and not self._was_hidden
            and sig == self._frame_sig
            and state["total_samples"] == self._last_total
        ):
            self._keepalive()
            return
        self._frame_sig = sig
        if self._was_hidden:
            self._was_hidden = False
            self.prev_tpl = None
            self._crossing_smooth = None
            self._sm_bufs.clear()
        self.frames += 1

        now_t = time.perf_counter()
        dt = min(max(now_t - self._last_frame_t, 1e-3), 0.05)
        self._last_frame_t = now_t
        blend_a = 1.0 - math.exp(-dt / self.TRIG_BLEND_TAU)
        blend_on = trig_on and self._smooth_stab

        if self._edge_clip_on:
            # Включено: размер амплитуды в зависимости от усиления, пики бьются об края "прям жёстко"
            clip_lim = float("inf") if blend_on else self.lim
            blend_lim = self.lim
        else:
            # Выключено: обычный режим (за основу взята бывшая включенная настройка)
            clip_lim = self.lim
            blend_lim = self.lim

        if not blend_on and self._sm_bufs:
            self._sm_bufs.clear()

        use_fade = (
            self._fade_blend > 0.01
            and self._demo_env_cache is not None
            and self._demo_env_cache.size == n
        )

        env = None
        if use_fade:
            b = float(self._fade_blend)
            env = (
                self._demo_env_cache
                if b >= 0.999
                else 1.0 + (self._demo_env_cache - 1.0) * b
            )

        ds = 1
        if width_px > 1.0:
            ds = max(1, int(n / (width_px * self.CURVE_DS_FACTOR)))

        if ds > 1:
            m = n // ds
            xd = np.add(self._x[: m * ds : ds], 0.5 * (ds - 1), out=self._dec_x[:m])
        else:
            m = n
            xd = self._x

        main_yl = None
        main_yr = None

        if multi:
            need = n * 2 if trig_on else n
            total = audio_snapshot_multi(
                need, self._n_ch, self._multi_bufs, self._scr_t if trig_on else None
            )
            shift = total - self._last_total
            self._last_total = total
            start = 0
            if trig_on:
                start = n
                x = self._x
                found = self._trigger_position(self._scr_t[:need], shift, need, n)
                if found is not None:
                    start, x_off = found
                    start = max(0, min(start, need - n))
                    x = np.add(self._x, x_off, out=self._xs[:n])
                if ds > 1:
                    xd = np.add(x[: m * ds : ds], 0.5 * (ds - 1), out=self._dec_x[:m])
                else:
                    xd = x

            for i, curve_item in enumerate(self.multi_curves):
                raw = self._multi_bufs[i][start : start + n]
                if env is not None:
                    np.multiply(raw, env, out=self._fade_buf[:n])
                    src = self._fade_buf[:n]
                else:
                    src = raw
                filt = self._filters[i] if i < len(self._filters) else None
                if filt is not None:
                    filt.process(src, self._filt_out[:n])
                    filtered = self._filt_out[:n]
                else:
                    filtered = src
                np.clip(filtered, -clip_lim, clip_lim, out=self._disp_buf[:n])
                yl = self._decimate(self._disp_buf[:n], ds, m, self._dec_y)
                if blend_on:
                    yl = self._blend_frame(i, yl, blend_a, blend_lim)

                if _absmax(yl) < 1e-6:
                    curve_item.setVisible(False)
                    if i < len(self.multi_peak_curves):
                        self.multi_peak_curves[i].setVisible(False)
                else:
                    curve_item.setVisible(True)
                    curve_item.setData(xd, yl, skipFiniteCheck=True)

                    if self._show_peak_hold and i < len(self.multi_peak_curves):
                        self.multi_peak_curves[i].setVisible(True)
                        env_ph = self.multi_peak_holds[i].update(yl, dt)
                        self.multi_peak_curves[i].setData(
                            xd, env_ph, skipFiniteCheck=True
                        )
                    elif i < len(self.multi_peak_curves):
                        self.multi_peak_curves[i].setVisible(False)

                if i == 0:
                    main_yl = yl

            self.last = self._multi_bufs[0][start : start + n]
            self.last_r = (
                self._multi_bufs[1][start : start + n] if self._n_ch > 1 else self.last
            )

        else:
            for ph_c in self.multi_peak_curves:
                ph_c.setVisible(False)

            n2 = n + n
            need = n2 if trig_on else n
            want_mid = trig_on or not stereo
            total = audio_snapshot(
                need,
                self._scr_t if want_mid else None,
                self._scr_l if stereo else None,
                self._scr_r if stereo else None,
            )
            shift = total - self._last_total
            self._last_total = total
            data_t = self._scr_t[:need]

            if trig_on:
                start = n
                x = self._x
                found = self._trigger_position(data_t, shift, need, n)
                if found is not None:
                    start, x_off = found
                    start = max(0, min(start, need - n))
                    x = np.add(self._x, x_off, out=self._xs[:n])
            else:
                start = 0
                x = self._x

            if ds > 1:
                xd = np.add(x[: m * ds : ds], 0.5 * (ds - 1), out=self._dec_x[:m])
            else:
                xd = x

            if stereo:
                raw_l = self._scr_l[start : start + n]
                raw_r = self._scr_r[start : start + n]
                if env is not None:
                    np.multiply(raw_l, env, out=self._fade_buf[:n])
                    np.multiply(raw_r, env, out=self._fade_buf_r[:n])
                    view_l = self._fade_buf[:n]
                    view_r = self._fade_buf_r[:n]
                else:
                    view_l = raw_l
                    view_r = raw_r
                self.last = view_l
                self.last_r = view_r

                if self._filters[0] is not None:
                    self._filters[0].process(view_l, self._filt_out[:n])
                    np.clip(
                        self._filt_out[:n], -clip_lim, clip_lim, out=self._disp_buf[:n]
                    )
                else:
                    np.clip(view_l, -clip_lim, clip_lim, out=self._disp_buf[:n])

                if self._filters[1] is not None:
                    self._filters[1].process(view_r, self._filt_out[:n])
                    np.clip(
                        self._filt_out[:n],
                        -clip_lim,
                        clip_lim,
                        out=self._disp_buf_r[:n],
                    )
                else:
                    np.clip(view_r, -clip_lim, clip_lim, out=self._disp_buf_r[:n])

                yl = self._decimate(self._disp_buf[:n], ds, m, self._dec_y)
                yr = self._decimate(self._disp_buf_r[:n], ds, m, self._dec_yr)
                if blend_on:
                    yl = self._blend_frame("l", yl, blend_a, blend_lim)
                    yr = self._blend_frame("r", yr, blend_a, blend_lim)
                self.curve.setData(xd, yl, skipFiniteCheck=True)
                self.curve_r.setData(xd, yr, skipFiniteCheck=True)
                self.glow_renderer.update(xd, yl, vb, self.lim)
                self.glow_renderer_r.update(xd, yr, vb, self.lim)
                main_yl = yl
                main_yr = yr
            else:
                raw = data_t[start : start + n]
                if env is not None:
                    np.multiply(raw, env, out=self._fade_buf[:n])
                    view = self._fade_buf[:n]
                else:
                    view = raw
                self.last = view
                self.last_r = view

                if self._filters[0] is not None:
                    self._filters[0].process(view, self._filt_out[:n])
                    np.clip(
                        self._filt_out[:n], -clip_lim, clip_lim, out=self._disp_buf[:n]
                    )
                else:
                    np.clip(view, -clip_lim, clip_lim, out=self._disp_buf[:n])

                yl = self._decimate(self._disp_buf[:n], ds, m, self._dec_y)
                if blend_on:
                    yl = self._blend_frame("l", yl, blend_a, blend_lim)
                self.curve.setData(xd, yl, skipFiniteCheck=True)
                if self.curve_r.isVisible():
                    self.curve_r.clear()
                self.glow_renderer.update(xd, yl, vb, self.lim)
                main_yl = yl

        if self._show_peak_hold and not multi and main_yl is not None:
            self._draw_peak(self.peak_hold_curve, self._peak_hold, main_yl, xd, dt)
            if stereo and main_yr is not None:
                self._draw_peak(
                    self.peak_hold_curve_r, self._peak_hold_r, main_yr, xd, dt
                )
            elif self.peak_hold_curve_r.isVisible():
                self.peak_hold_curve_r.setVisible(False)
        elif not self._show_peak_hold:
            if self.peak_hold_curve.isVisible():
                self.peak_hold_curve.clear()
            if self.peak_hold_curve_r.isVisible():
                self.peak_hold_curve_r.clear()

        self.update_stats()

    @staticmethod
    def _draw_peak(curve, ph, y, xd, dt):
        env = ph.update(y, dt)
        if _absmax(y) < 1e-6:
            if curve.isVisible():
                curve.setVisible(False)
            return
        if not curve.isVisible():
            curve.setVisible(True)
        curve.setData(xd, env, skipFiniteCheck=True)

    def _blend_frame(self, key, y, a, lim=None):
        """Смешивание кадров с компенсацией амплитуды.
        Восстанавливает исходный размер волны, а затем обрезает по lim,
        создавая эффект 'удара об края' (плоские вершины)."""
        buf = self._sm_bufs.get(key)
        if buf is None or buf.shape != y.shape:
            buf = self._sm_bufs[key] = np.array(y, dtype=np.float32)
            res = buf
        elif a >= 0.999:
            buf[:] = y
            res = y
        else:
            tmp = self._sm_tmp
            if tmp is None or tmp.shape[0] < y.shape[0]:
                tmp = self._sm_tmp = np.empty(max(y.shape[0], 1024), dtype=np.float32)
            t = tmp[: y.shape[0]]
            np.subtract(y, buf, out=t)
            t *= a
            buf += t
            res = buf

        if lim is None:
            return res

        out = self._sm_out.get(key)
        if out is None or out.shape != res.shape:
            out = self._sm_out[key] = np.empty(res.shape, dtype=np.float32)

        if res is y:
            out[:] = y
        else:
            pk_b = _absmax(res)
            pk_y = _absmax(y)
            if pk_b > 1e-6 and pk_y > 1e-6:
                s = min(max(pk_y / pk_b, 0.5), 2.0)
                ps = self._sm_scale.get(key, s)
                if s > ps:
                    ps = s
                else:
                    ps = ps + (s - ps) * min(1.0, a * 2.0)
                self._sm_scale[key] = ps
                np.multiply(res, ps, out=out)
            else:
                out[:] = res

        np.clip(out, -lim, lim, out=out)
        return out

    @staticmethod
    def _db(x):
        return "−∞" if x < 1e-6 else f"{20 * np.log10(x):+.1f}".replace("-", "−")

    _NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

    def _hz_parts(self, hz: float):
        if hz < 1.0:
            return "0.00Hz", None, 0
        txt = f"{hz:.2f}Hz" if hz < 1000.0 else f"{hz / 1000.0:.2f}kHz"
        midi = 69.0 + 12.0 * math.log2(hz / 440.0)
        n = int(round(midi))
        cents = int((midi - n) * 100.0)
        return txt, f"{self._NOTE_NAMES[n % 12]}{n // 12 - 1}", cents

    def _format_hz(self, hz: float) -> str:
        return self._hz_parts(hz)[0]

    def _format_hz_plain(self, hz: float) -> str:
        txt, name, cents = self._hz_parts(hz)
        if name is None:
            return txt
        sign = "-" if cents < 0 else ""
        return f"{txt} | {name} - {sign}{abs(cents)} Cents"

    def _try_reconnect(self):
        self._retry_pending = False
        if not state["error"]:
            return
        try:
            self._on_device_change(self.device_combo.currentIndex())
        except Exception:
            pass

    def _stats_fallback(self):
        if (not _ui_gate[0]) and not self.frozen and not self._ui_hidden():
            _ui_gate[0] = True
        if state["error"] and not self._retry_pending:
            self._retry_pending = True
            QtCore.QTimer.singleShot(2000, self._try_reconnect)
        idle = getattr(self, "_stats_idle_s", STATS_MS / 1000.0)
        if time.perf_counter() - getattr(self, "_stats_t", 0.0) > idle:
            self.update_stats()

    def _feed_bpm(self):
        total_s = state["total_samples"]
        new_n = total_s - self._bpm_last_total
        if new_n <= 0:
            return
        if new_n > 4096:
            new_n = 4096
        audio_snapshot(new_n, out_m=self._bpm_scr[:new_n])
        self._bpm.feed(self._bpm_scr[:new_n])
        self._bpm_last_total = total_s

    def update_stats(self):
        now = time.perf_counter()
        if now - getattr(self, "_stats_t", 0.0) < 1.0 / 30.0:
            return
        self._stats_t = now
        if self._ui_hidden():
            self.frames = 0
            self.t0 = now
            return

        if self._bpm_on:
            self._feed_bpm()
            if self._bpm.maybe_update():
                bpm_val = (
                    f"{self._bpm.bpm:.0f}"
                    if (self._bpm.bpm > 30.0 and self._bpm.confidence > 0.1)
                    else "—"
                )
                self.bpm_tile.set(tr("bpm"), bpm_val)

        total_s = state["total_samples"]
        if (
            not self.frozen
            and total_s - getattr(self, "_metrics_total", -(10**9)) >= HZ_EST_SAMPLES
        ):
            self._metrics_total = total_s
            seg = self.last
            if self.channels == "stereo" and len(self.last_r):
                if len(seg):
                    self._peak_l_val = _absmax(seg)
                    self._peak_r_val = _absmax(self.last_r)
                    self._peak_val = max(self._peak_l_val, self._peak_r_val)
                    self._rms_val = math.sqrt(
                        0.5
                        * (
                            float(np.dot(seg, seg)) / len(seg)
                            + float(np.dot(self.last_r, self.last_r)) / len(self.last_r)
                        )
                    )
                else:
                    self._peak_val = self._peak_l_val = 0.0
                    self._peak_r_val = self._rms_val = 0.0
            else:
                if len(seg):
                    self._peak_val = _absmax(seg)
                    self._rms_val = math.sqrt(float(np.dot(seg, seg)) / len(seg))
                else:
                    self._peak_val = self._rms_val = 0.0

            hz_raw = self._estimate_bass_hz()
            if hz_raw > 1.0:
                self._hz_silent_since = None
                if self._hz_smooth < 1.0:
                    self._hz_smooth = hz_raw
                else:
                    rel = abs(hz_raw - self._hz_smooth) / max(self._hz_smooth, 1.0)
                    k = 0.7 if rel > 0.08 else (0.4 if rel > 0.01 else 0.25)
                    self._hz_smooth = self._hz_smooth * (1.0 - k) + hz_raw * k
            else:
                t_s = getattr(self, "_hz_silent_since", None)
                if t_s is None:
                    self._hz_silent_since = now
                elif now - t_s > 0.6:
                    self._hz_smooth = 0.0

        dim = '<span style="color:#6e7681">'
        peak = self._peak_val
        rms = self._rms_val
        if self.channels == "stereo":
            peak_l = self._peak_l_val
            peak_r = self._peak_r_val
            clip_l = peak_l >= 0.999
            clip_r = peak_r >= 0.999
            col_l = "#f85149" if clip_l else self.accent
            col_r = "#f85149" if clip_r else self._stereo_r_hex
            mark_l = f" {tr('clip')}" if clip_l else ""
            mark_r = f" {tr('clip')}" if clip_r else ""
            self.peak_tile.set(
                tr("peak") + " · dBFS",
                f'<span style="color:{col_l}">L {self._db(peak_l)}{mark_l}</span>'
                f"{dim} · </span>"
                f'<span style="color:{col_r}">R {self._db(peak_r)}{mark_r}</span>',
            )
            self.rms_tile.set(tr("rms"), f"{self._db(rms)} dBFS")
        else:
            peak_color = "#f85149" if peak >= 0.999 else self.accent
            clip = f"  {tr('clip')}" if peak >= 0.999 else ""
            self.peak_tile.set(
                tr("peak"),
                f'<span style="color:{peak_color}">{self._db(peak)} dBFS{clip}</span>',
            )
            self.rms_tile.set(tr("rms"), f"{self._db(rms)} dBFS")

        hz_txt = self._format_hz(self._hz_smooth)
        if self._solo:
            vals = (
                self._db(peak),
                self._db(rms),
                self._format_hz_plain(self._hz_smooth),
                self._hz_parts(self._hz_smooth)[0],
            )
            if vals != self._solo_vals:
                self._solo_vals = vals
                if self._hud_on:
                    self._fit_hud()
        self.hz_tile.set(tr("hz_label"), hz_txt)
        if self.hz_tile.toolTip() != tr("hz_tip"):
            self.hz_tile.setToolTip(tr("hz_tip"))

        dt = now - self.t0
        if dt >= FPS_AVG_S:
            self._fps_measured = self.frames / max(dt, 1e-3)
            self.frames = 0
            self.t0 = now
        fps = getattr(self, "_fps_measured", 0.0)
        self.fps_tile.set(
            tr("fps_label"), tr("paused") if self.frozen else f"{fps:.0f}"
        )

        if now - getattr(self, "_slow_t", 0.0) < SLOW_STATS_S:
            return
        self._slow_t = now
        self.info_tile.set(
            tr("window_label"),
            f'{self.n} {tr("samples")} {dim}·</span> '
            f'{self.n / SAMPLERATE * 1000:.1f} {tr("ms")}',
        )
        be_name = tr(_BACKEND_NAME_KEYS.get(_BACKEND_ID, "backend_raster"))
        self.bknd_tile.set(tr("backend_label"), be_name)
        if self.bknd_tile.toolTip() != tr("backend_tip"):
            self.bknd_tile.setToolTip(tr("backend_tip"))
        has_err = bool(state["error"])
        if has_err != self.device_err:
            self.device_err = has_err
            self.device_lbl.setStyleSheet("color: #f85149;" if has_err else "")
            self.dot.setStyleSheet(
                "background: #f85149; border-radius: 4px;" if has_err else ""
            )
        if has_err:
            full = f'{tr("error_prefix")} {state["error"]}'
        else:
            name = state["device"] or tr("searching")
            full = f'{name}  ·  {SAMPLERATE / 1000:g} {tr("khz")}'
        if self.pill.toolTip() != full:
            self.pill.setToolTip(full)
        elided = self.device_lbl.fontMetrics().elidedText(
            full,
            QtCore.Qt.TextElideMode.ElideRight,
            max(self.device_lbl.width() - 4, 20),
        )
        if self.device_lbl.text() != elided:
            self.device_lbl.setText(elided)

    def closeEvent(self, event):
        self._save_settings()
        try:
            self._capturer.stop()
        except Exception:
            pass
        super().closeEvent(event)
