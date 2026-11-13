"""Главное окно приложения (MainWindow)."""

from typing import Any

import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from backend import _USE_GL
from animations import (
    SOLO_TRANSITION_DURATION_MS,
    animate_button,
    fade_in,
    fade_out,
)
from settings import (
    CHANNEL_LAYOUTS,
    CH_LAYOUT_TO_N,
    BPM_HISTORY_SAMPLES,
    CONFIG,
    FPS_OPTIONS,
    GAIN_OPTIONS,
    LANG,
    LANGUAGES,
    MAX_CH,
    SAMPLERATE,
    SAMPLE_OPTIONS,
    STATS_MS,
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
    ToggleSwitch,
    _rgba,
    _stereo_r_color,
    _surface,
)
from audio import (
    Capturer,
    _bridge,
    _chunk_pending,
    list_loopback_devices,
    set_channel_count,
    state,
)
from trigger import TriggerMixin
from widgets import (
    SOLO_HIT_ALPHA,
    GlowRenderer,
    SpacedAxis,
    StaticShadow,
    WavePlotWidget,
    _gv_cache_mode,
    GradientCurveItem,
)
from dsp import Biquad, make_filter
from analysis import BpmDetector, PeakHold
from w_dialogs import WindowDialogsMixin
from w_render import WindowRenderMixin
from w_support import (
    _trim_memory,
)


class FpsFreezeTile(QtWidgets.QFrame):
    """Плитка: слева FPS, справа кнопка «Заморозить/Продолжить»."""

    def __init__(self, width: int, button, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("tile")
        self.setFixedWidth(width)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 9, 8)
        lay.setSpacing(10)
        self.info = MiniTile()
        lay.addWidget(self.info, 1)
        lay.addWidget(button, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

    def set(self, cap: str, val_html: str) -> None:
        self.info.set(cap, val_html)


class MiniTile(QtWidgets.QWidget):
    """Подплитка без рамки (подпись + значение) — часть PairTile."""

    def __init__(self, parent=None) -> None:
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

    def set(self, cap: str, val_html: str) -> None:
        if self.cap.text() != cap:
            self.cap.setText(cap)
        if self.val.text() != val_html:
            self.val.setText(val_html)


class PairTile(QtWidgets.QFrame):
    """Одна плитка с двумя значениями: слева RMS, справа частота."""

    def __init__(self, width: int, parent=None) -> None:
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


class MainWindow(
    TriggerMixin,
    WindowDialogsMixin,
    WindowRenderMixin,
    QtWidgets.QWidget,
):
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

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("root")
        self.setWindowTitle(tr("main_title"))
        self.resize(1280, 548)

        self.n = CONFIG.get("n", 4096)
        self.frozen = False

        self.ch_layout = CONFIG.get("ch_layout", "mono")
        self._n_ch: int = CH_LAYOUT_TO_N.get(self.ch_layout, 1)
        self.channels: str = "mono" if self._n_ch == 1 else "stereo"
        set_channel_count(self._n_ch)

        self.last: np.ndarray[Any, np.dtype[np.floating]] = np.zeros(
            self.n, dtype=np.float32
        )
        self.last_r: np.ndarray[Any, np.dtype[np.floating]] = np.zeros(
            self.n, dtype=np.float32
        )
        self.lim = 1.0
        self._gain_display_scale = 1.0
        self.prev_tpl = None
        self._crossing_smooth = None
        self._trigger_pending_samples = 0
        self._last_total = 0
        self._period_smooth = 0.0
        self._lock_quality = 0.0

        self._fade_on = bool(CONFIG.get("fade", True))
        self._fade_blend: float = 1.0 if self._fade_on else 0.0
        self._fade_target: float = self._fade_blend
        self._demo_env_cache = None
        self._demo_agc = 0.0
        self._demo_agc_l = 0.0
        self._demo_agc_r = 0.0

        self._solo = False
        self._solo_transition = None
        self._card_shadow = None
        self._card_layout = None
        self._card_saved_margins = None
        self._plot_bg_main = None
        self._solo_pinned = bool(CONFIG.get("solo_pinned", False))
        self._solo_transparent = True
        self._hud_on = False
        self._solo_drag_mode = None
        self._solo_resize_edges = (False, False, False, False)
        self._solo_origin_global = None
        self._solo_origin_geom = None
        # Запоминаем положение окон при переключении «меню ⇄ соло».
        self._solo_saved_geom: QtCore.QRect | None = None
        _g = CONFIG.get("solo_geom")
        if _g:
            self._solo_saved_geom = QtCore.QRect(*_g)
        self._main_saved_pos: QtCore.QPoint | None = None

        self._fps_setting = int(CONFIG.get("fps", 0))

        self._band_kind = CONFIG.get("bandpass", "off")
        self._filters: list[None | Biquad] = [
            make_filter(self._band_kind) for _ in range(MAX_CH)
        ]
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
        self._bpm_scr: np.ndarray[Any, np.dtype[np.floating]] = np.empty(
            BPM_HISTORY_SAMPLES, dtype=np.float32
        )
        self._show_gradient = bool(CONFIG.get("gradient", False))
        self._show_peak_hold = bool(CONFIG.get("peak_hold", True))
        self._smooth_stab = bool(CONFIG.get("smooth_stab", True))
        self._edge_clip_on = bool(CONFIG.get("edge_clip", True))

        self.multi_curves = []
        self.multi_peak_curves = []
        self.multi_peak_holds = []
        self._multi_bufs = []
        self._filt_out: np.ndarray[Any, np.dtype[np.floating]] = np.empty(
            self.n, dtype=np.float32
        )
        self._alloc_scratch(self.n)

        self._frame_sig = None
        self._was_hidden = False
        self._last_keepalive = 0.0
        self._chunk_driven = False
        self._chunk_connected = False

        self._kernel = np.ones(8, dtype=np.float32) / 8
        self._stats_idle_s: float = STATS_MS / 1000.0
        self._retry_pending = False
        self._x: np.ndarray[Any, np.dtype[np.floating]] = np.arange(
            self.n, dtype=np.float32
        )
        self.frames = 0
        self._hz_smooth = 0.0
        self.t0: float = time.perf_counter()
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
        self.accent: str = self._theme_color(self.theme_id)
        self._current_color = QtGui.QColor(self.accent)
        self._theme_anim = None
        self._stereo_r_hex: str = _stereo_r_color(self.accent).name()
        self._solid_pen = None
        self._gradient_pen = None

        self._capturer = Capturer()
        self._loading = True
        self._layout_applied = False
        self._set_chunk_driven(int(CONFIG.get("fps", 0)) == 0)

        self._build_ui()
        self._apply_theme(self.theme_id)
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

    def _toggle_debug(self) -> None:
        self.SHOW_DEBUG: bool = not self.SHOW_DEBUG
        if self.SHOW_HEADER:
            self.dbg_lbl.setVisible(self.SHOW_DEBUG)
        self._update_debug()
        self._fit_pill()

    def _update_debug(self) -> None:
        if not self.SHOW_DEBUG or not self.SHOW_HEADER:
            return
        w, h = self.width(), self.height()
        dpr: float = self.devicePixelRatioF()
        dpi: int = self.logicalDpiX()
        txt: str = (
            f"{w}×{h} · {round(dpr * 100)}% · "
            f"{round(w * dpr)}×{round(h * dpr)} px · {dpi:.0f} dpi"
        )
        if self.dbg_lbl.text() != txt:
            self.dbg_lbl.setText(txt)

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        wh: QtGui.QWindow | None = self.windowHandle()
        if wh is not None and not getattr(self, "_dbg_hooked", False):
            self._dbg_hooked = True
            try:
                wh.screenChanged.connect(lambda *_: self._update_debug())
            except Exception:
                pass
        self._update_debug()
        self._fit_pill()

    def _alloc_scratch(self, n) -> None:
        f = np.float32
        self._scr_t: np.ndarray[Any, np.dtype[np.floating]] = np.empty(n * 2, f)
        self._fade_buf: np.ndarray[Any, np.dtype[np.floating]] = np.empty(n, f)
        self._disp_buf: np.ndarray[Any, np.dtype[np.floating]] = np.empty(n, f)
        self._filt_out: np.ndarray[Any, np.dtype[np.floating]] = np.empty(n, f)
        self._xs: np.ndarray[Any, np.dtype[np.floating]] = np.empty(n, f)
        self._dec_x: np.ndarray[Any, np.dtype[np.floating]] = np.empty(n, f)
        self._dec_y: np.ndarray[Any, np.dtype[np.floating]] = np.empty(n, f)

        self._multi_bufs: list[np.ndarray[Any, np.dtype[np.floating]]] = (
            [np.empty(n * 2, f) for _ in range(self._n_ch)] if self._n_ch > 2 else []
        )

        if self._n_ch == 2:
            self._scr_l: np.ndarray[Any, np.dtype[np.floating]] = np.empty(n * 2, f)
            self._scr_r: np.ndarray[Any, np.dtype[np.floating]] = np.empty(n * 2, f)
            self._fade_buf_r: np.ndarray[Any, np.dtype[np.floating]] = np.empty(n, f)
            self._disp_buf_r: np.ndarray[Any, np.dtype[np.floating]] = np.empty(n, f)
            self._dec_yr: np.ndarray[Any, np.dtype[np.floating]] = np.empty(n, f)
        else:
            self._scr_l = self._scr_r = None
            self._fade_buf_r = self._disp_buf_r = self._dec_yr = None

    @staticmethod
    def _decimate(y, ds, m, out):
        if ds <= 1:
            return y
        return np.mean(y[: m * ds].reshape(m, ds), axis=1, out=out[:m])

    def _trim_if_hidden(self) -> None:
        if self._ui_hidden():
            _trim_memory()

    def _fit_pill(self) -> None:
        pass

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._update_debug()
        self._fit_pill()

    def _save_settings(self) -> None:
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
        idx: int = self.device_combo.currentIndex()
        if idx >= 0:
            CONFIG["device_id"] = self.device_combo.itemData(idx)
        CONFIG["solo_pinned"] = self._solo_pinned
        if self._solo:
            self._solo_saved_geom = QtCore.QRect(self.card.geometry())
        g = self._solo_saved_geom
        if g is not None:
            CONFIG["solo_geom"] = (g.x(), g.y(), g.width(), g.height())
        _save_config(CONFIG)

    def _populate_devices(self) -> None:
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
                select_idx: int = i
        self.device_combo.setEnabled(True)
        self.device_combo.setCurrentIndex(select_idx)
        self.device_combo.blockSignals(False)
        self._on_device_change(select_idx)

    def _on_device_change(self, idx) -> None:
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

    def _on_language_change(self, idx) -> None:
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

    def _refresh_fps_combo(self) -> None:
        if not hasattr(self, "fps_combo"):
            return
        self.fps_combo.blockSignals(True)
        self.fps_combo.clear()
        for v in FPS_OPTIONS:
            self.fps_combo.addItem(self._fps_label(v), v)
        try:
            idx: int = FPS_OPTIONS.index(self._fps_setting)
        except ValueError:
            idx = 0
        self.fps_combo.setCurrentIndex(idx)
        self.fps_combo.blockSignals(False)
        self.fps_combo.setToolTip(tr("fps_tip"))

    def _on_fps_change(self, idx) -> None:
        if idx < 0:
            return
        v = self.fps_combo.itemData(idx)
        if v is None or v == self._fps_setting:
            return
        self._fps_setting = int(v)
        self._apply_fps()
        if not getattr(self, "_loading", False):
            CONFIG["fps"] = self._fps_setting
            CONFIG["fps_customized"] = True
            _save_config(CONFIG)

    def _apply_fps(self) -> None:
        interval: int = UPDATE_MS
        v = int(getattr(self, "_fps_setting", 0))
        self._set_chunk_driven(v == 0)
        if v == 0:
            interval: int = WATCHDOG_MS
        elif v == -1:
            scr: QtGui.QScreen | None = self.screen()
            if scr is not None:
                hz = float(scr.refreshRate() or 0.0)
                if hz > 1.0:
                    interval: int = max(1, int(round(1000.0 / hz)))
        elif v > 0:
            interval: int = max(1, int(round(1000.0 / v)))
        t: Any | None = getattr(self, "timer", None)
        if t is not None:
            t.stop()
            t.setTimerType(
                QtCore.Qt.TimerType.CoarseTimer
                if v == 0
                else QtCore.Qt.TimerType.PreciseTimer
            )
            t.start(interval)
        self._stats_idle_s: float = max(STATS_MS, 2 * interval) / 1000.0

    def _set_chunk_driven(self, on: bool) -> None:
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

    def _on_tick(self) -> None:
        _chunk_pending[0] = False
        self.update_frame()

    def _on_audio_chunk(self) -> None:
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

    def _update_freeze_icon(self) -> None:
        if not hasattr(self, "freeze_btn"):
            return
        color: str = "#0d1117" if self.frozen else self.accent
        self.freeze_btn.setIcon(self._freeze_icon(self.frozen, color))

    def _retranslate_ui(self) -> None:
        self.setWindowTitle(tr("main_title"))
        if hasattr(self, "sub_lbl"):
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
        idx: int = self.n_combo.currentIndex()
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

    def _reapply_ranges(self) -> None:
        n = self.n
        self.plot.setXRange(-n * 0.012, n, padding=0)
        scale = getattr(self, "_gain_display_scale", self.lim)
        self.plot.setYRange(-1.06 * scale, 1.06 * scale, padding=0)

    def _enter_solo(self) -> None:
        if self._solo or self._solo_transition is not None:
            return
        self._solo = True
        size = QtCore.QSize(self.SOLO_W, self.SOLO_H)
        if self._card_layout is not None:
            self._card_saved_margins: QtCore.QMargins = (
                self._card_layout.contentsMargins()
            )
            self._card_layout.setContentsMargins(0, 0, 0, 0)
        pi = self.plot.getPlotItem()
        pi.hideAxis("bottom")
        pi.hideAxis("left")
        pi.showGrid(x=True, y=True, alpha=0.08)
        self.lim_hi.setVisible(False)
        self.lim_lo.setVisible(False)
        try:
            pi.getViewBox().disableAutoRange()
        except Exception:
            pass
        self._plot_bg_main: QtGui.QColor = pg.mkColor(_surface(self.accent))
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
        self._place_solo_card(size)
        self.card.setWindowOpacity(0.0)
        self.card.show()
        self.card.raise_()
        self.card.activateWindow()
        if not (self.isMaximized() or self.isFullScreen()):
            self._main_saved_pos = QtCore.QPoint(self.pos())
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
        self._solo_transition = fade_in(self.card, SOLO_TRANSITION_DURATION_MS)
        self._solo_transition.finished.connect(self._clear_solo_transition)

    def _persist_solo_geom(self) -> None:
        g = self._solo_saved_geom
        if g is not None:
            CONFIG["solo_geom"] = (g.x(), g.y(), g.width(), g.height())
            _save_config(CONFIG)

    def _place_solo_card(self, default_size: QtCore.QSize) -> None:
        """Соло-окно: при первом открытии — по центру экрана, дальше — туда,
        где пользователь оставил его в прошлый раз."""
        saved = self._solo_saved_geom
        if saved is not None:
            for scr in QtGui.QGuiApplication.screens():
                visible = scr.availableGeometry().intersected(saved)
                if visible.width() >= min(saved.width(), 120) and visible.height() >= min(
                    saved.height(), 60
                ):
                    self.card.setGeometry(saved)
                    return
        scr = self.screen() or QtGui.QGuiApplication.primaryScreen()
        rect = QtCore.QRect(QtCore.QPoint(0, 0), default_size)
        if scr is not None:
            rect.moveCenter(scr.availableGeometry().center())
        self.card.setGeometry(rect)

    def _exit_solo(self) -> None:
        if not self._solo or self._solo_transition is not None:
            return
        self._solo_saved_geom = QtCore.QRect(self.card.geometry())
        self._persist_solo_geom()
        self._solo_transition = fade_out(self.card)
        self._solo_transition.finished.connect(self._finish_exit_solo)

    def _finish_exit_solo(self) -> None:
        self._solo_transition = None
        self._solo = False
        self._solo_drag_mode = None
        self._hud_timer.stop()
        self._hud_anim.stop()
        self._hud_on = False
        self._hud.hide()
        self.card.hide()
        self.card.setWindowOpacity(1.0)
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
        pi.showGrid(x=True, y=True, alpha=0.11)
        self.lim_hi.setVisible(True)
        self.lim_lo.setVisible(True)
        self.left.setWidth(48)
        if self._plot_bg_main is not None:
            self.plot.setBackground(self._plot_bg_main)
        self._apply_plot_render_mode()
        self._top_layout.insertWidget(0, self.card, 1)
        self.plot.setStyleSheet("background: transparent;")
        self.setWindowOpacity(0.0)
        if self._main_saved_pos is not None and not (
            self.isMaximized() or self.isFullScreen()
        ):
            self.move(self._main_saved_pos)
        self.show()
        self.card.show()
        self.raise_()
        self.activateWindow()
        self._solo_transition = fade_in(self, SOLO_TRANSITION_DURATION_MS)
        self._solo_transition.finished.connect(self._clear_solo_transition)
        QtCore.QTimer.singleShot(0, self._reapply_ranges)
        self._solo_esc.setEnabled(False)

    def _clear_solo_transition(self) -> None:
        self._solo_transition = None

    def _plot_bg(self) -> QtGui.QColor:
        if self._solo and self._solo_transparent:
            return pg.mkColor((0, 0, 0, SOLO_HIT_ALPHA))
        return pg.mkColor(_surface(self.accent))

    def _solo_card_qss(self) -> str:
        if self._solo_transparent:
            # Фон карточки полностью прозрачный: клики ловит слой графика (альфа 1/255).
            bg, border = "transparent", _rgba(self.accent, 0.35)
        else:
            bg, border = _surface(self.accent), _rgba(self.accent, 0.55)
        return f"""
            QFrame#card {{ background: {bg}; border: 1px solid {border}; border-radius: 0; }}
            QToolTip {{ background: #161b22; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 5px 10px; font-family: 'Segoe UI'; font-size: 12px; }}
        """

    def _apply_solo_flags(self) -> None:
        flags: QtCore.Qt.WindowType = (
            QtCore.Qt.WindowType.Window | QtCore.Qt.WindowType.FramelessWindowHint
        )
        if self._solo_pinned:
            flags |= QtCore.Qt.WindowType.WindowStaysOnTopHint
        self.card.setWindowFlags(flags)

    def _apply_plot_render_mode(self) -> None:
        transparent: bool = self._solo and self._solo_transparent
        self.plot.transparent_bg = transparent
        try:
            self.plot.setCacheMode(_gv_cache_mode(transparent))
        except Exception:
            pass
        if not _USE_GL:
            self.plot.viewport().update()
            return
        try:
            vp: QtWidgets.QWidget | None = self.plot.viewport()
            if not hasattr(vp, "setUpdateBehavior"):
                return
            cls = type(vp)
            ub: Any | QtWidgets.QWidget | type[None] = getattr(
                cls, "UpdateBehavior", cls
            )
            vp.setUpdateBehavior(
                ub.NoPartialUpdate if transparent else ub.PartialUpdate
            )
            vp.setAutoFillBackground(False)
            vp.update()
        except Exception:
            pass

    def _on_solo_bg(self, on) -> None:
        self._solo_transparent = bool(on)
        if self._solo:
            self._apply_plot_render_mode()
            self.plot.setBackground(self._plot_bg())
            self.card.setStyleSheet(self._solo_card_qss())
            self.plot.viewport().update()

    def _on_solo_pin(self, on) -> None:
        self._solo_pinned = bool(on)
        CONFIG["solo_pinned"] = self._solo_pinned
        _save_config(CONFIG)
        if self._solo:
            self._apply_solo_flags()
            self.card.show()
            self.card.raise_()
            self.card.activateWindow()
            self._layout_hud()
            self._hud.raise_()

    def _build_solo_hud(self) -> None:
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
        self.solo_freeze: IconButton[str] = IconButton("freeze", checkable=True)
        self.solo_freeze.clicked.connect(lambda: self.freeze_btn.click())
        self.solo_bg: IconButton[str] = IconButton("bg", checkable=True)
        self.solo_bg.setChecked(self._solo_transparent)
        self.solo_bg.toggled.connect(self._on_solo_bg)
        self.solo_pin: IconButton[str] = IconButton("pin", checkable=True)
        self.solo_pin.setChecked(self._solo_pinned)
        self.solo_pin.toggled.connect(self._on_solo_pin)
        self.solo_close: IconButton[str] = IconButton("close")
        self.solo_close.clicked.connect(self._exit_solo)
        lay.addWidget(self.solo_dot, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.solo_read)
        lay.addStretch(1)
        lay.addWidget(self.solo_freeze)
        lay.addWidget(self.solo_bg)
        lay.addWidget(self.solo_pin)
        lay.addWidget(self.solo_close)
        self._hud: QtWidgets.QFrame = hud
        self._hud_lay: QtWidgets.QHBoxLayout = lay
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

    def _apply_hud_texts(self) -> None:
        if not hasattr(self, "_hud"):
            return
        self.solo_freeze.setToolTip(tr("solo_freeze"))
        self.solo_bg.setToolTip(tr("solo_bg"))
        self.solo_pin.setToolTip(tr("solo_pin"))
        self.solo_close.setToolTip(tr("solo_exit"))
        self.solo_title.setText(tr("brand"))

    def _style_hud(self) -> None:
        if not hasattr(self, "_hud"):
            return
        self._hud.setStyleSheet(f"""
            QFrame#hud {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(0,0,0,175), stop:1 rgba(0,0,0,0)); border: none; }}
            QLabel {{ color: #c9d1d9; font-family: 'Segoe UI'; font-size: 12px; background: transparent; }}
            QLabel#stitle {{ color: #f0f6fc; font-size: 13px; font-weight: 700; }}
            QLabel#sread {{ color: #8b949e; font-family: 'Cascadia Mono', 'Consolas', monospace; font-size: 12px; }}
            QLabel#sdot {{ background: {self.accent}; border-radius: 4px; }}
        """)
        for b in (
            self.solo_freeze,
            self.solo_bg,
            self.solo_pin,
            self.solo_close,
        ):
            b.set_accent(self.accent)

    def _layout_hud(self) -> None:
        self._hud.setGeometry(0, 0, self.card.width(), self._hud.height())
        self._fit_hud()

    def _fit_hud(self) -> None:
        w: int = self.card.width()
        sp: int = self._hud_lay.spacing()
        m: QtCore.QMargins = self._hud_lay.contentsMargins()
        btns: int = 4 * 32 + 3 * sp
        avail: int = w - m.left() - m.right() - btns - sp
        dot_w: int = 8 + sp
        fm: QtGui.QFontMetrics = self.solo_read.fontMetrics()
        p, r, _hz_full, hz_short = self._solo_vals
        cands: list[str] = [
            f"PEAK {p}  ·  RMS {r} dBFS  ·  {hz_short}",
            f"PEAK {p} · RMS {r} · {hz_short}",
            f"{p} · {r} dB · {hz_short}",
            f"PEAK {p} · RMS {r}",
            f"{p} · {r} dB",
            hz_short,
        ]
        left: int = avail - dot_w - 10
        text: str = ""
        for t in cands:
            if fm.horizontalAdvance(t) <= left:
                text: str = t
                break
        self.solo_title.setVisible(False)
        self.solo_read.setVisible(bool(text))
        self.solo_read.setText(text)

    def _hud_show(self, on: bool) -> None:
        if not self._solo:
            return
        self._hud_on: bool = on
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

    def _poke_hud(self) -> None:
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
        geo: QtCore.QRect = self.card.geometry()
        if not geo.contains(gp):
            return (False, False, False, False)
        local = gp - geo.topLeft()
        m: int = self._RESIZE_MARGIN
        return (
            local.x() <= m,
            local.x() >= geo.width() - m,
            local.y() <= m,
            local.y() >= geo.height() - m,
        )

    def _solo_update_cursor(self, gp) -> None:
        l, r, t, b = self._solo_hit_edges(gp)
        cur: QtCore.Qt.CursorShape = QtCore.Qt.CursorShape.ArrowCursor
        if (l and t) or (r and b):
            cur: QtCore.Qt.CursorShape = QtCore.Qt.CursorShape.SizeFDiagCursor
        elif (r and t) or (l and b):
            cur: QtCore.Qt.CursorShape = QtCore.Qt.CursorShape.SizeBDiagCursor
        elif l or r:
            cur: QtCore.Qt.CursorShape = QtCore.Qt.CursorShape.SizeHorCursor
        elif t or b:
            cur: QtCore.Qt.CursorShape = QtCore.Qt.CursorShape.SizeVerCursor
        self.card.setCursor(cur)

    def _solo_begin_drag(self, gp) -> None:
        l, r, t, b = self._solo_hit_edges(gp)
        geo: QtCore.QRect = self.card.geometry()
        self._solo_origin_global = gp
        self._solo_origin_geom = QtCore.QRect(geo)
        if l or r or t or b:
            self._solo_drag_mode = "resize"
            self._solo_resize_edges = (l, r, t, b)
        else:
            self._solo_drag_mode = "move"

    def _solo_apply_drag(self, gp) -> None:
        if self._solo_drag_mode is None or self._solo_origin_global is None:
            return
        dx = gp.x() - self._solo_origin_global.x()
        dy = gp.y() - self._solo_origin_global.y()
        g: None | QtCore.QRect = self._solo_origin_geom
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
                nx: int = x + w - minw
                nw: int = minw
            x, w = nx, nw
        if r:
            w: int = max(minw, w + dx)
        if t:
            ny = y + dy
            nh = h - dy
            if nh < minh:
                ny: int = y + h - minh
                nh: int = minh
            y, h = ny, nh
        if b:
            h: int = max(minh, h + dy)
        self.card.setGeometry(x, y, w, h)

    def eventFilter(self, obj, ev):
        et = ev.type()
        if (
            obj in (self.pill, self.device_lbl, self.dot)
            and et == QtCore.QEvent.Type.MouseButtonPress
            and ev.button() == QtCore.Qt.MouseButton.LeftButton
            and state.get("error")
        ):
            self._show_audio_error()
            return True
        plot: Any | None = getattr(self, "plot", None)
        card: Any | None = getattr(self, "card", None)
        if plot is None or card is None:
            return super().eventFilter(obj, ev)
        is_plot: bool = obj is plot or (
            hasattr(plot, "viewport") and obj is plot.viewport()
        )
        is_card: bool = obj is card
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

    SYS_POLL_MS = 2000

    def _build_ui(self) -> None:
        no_focus: QtCore.Qt.FocusPolicy = QtCore.Qt.FocusPolicy.NoFocus
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._outer_layout: QtWidgets.QVBoxLayout = outer

        self.content = QtWidgets.QWidget(self)
        root = QtWidgets.QVBoxLayout(self.content)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)
        self._root_layout: QtWidgets.QVBoxLayout = root
        outer.addWidget(self.content, 1)

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
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.pill.installEventFilter(self)
        self.device_lbl.installEventFilter(self)
        self.dot.installEventFilter(self)
        self.device_err = False
        pill_l.addWidget(self.dot, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        pill_l.addWidget(self.device_lbl, 1)
        self.pill.setMinimumWidth(220)
        self.pill.setMaximumWidth(480)
        self.pill.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        header.addWidget(self.pill, 1, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._header_layout: QtWidgets.QHBoxLayout = header
        if self.SHOW_HEADER:
            root.addLayout(header)
        else:
            for w in (self.pill, self.dbg_lbl):
                w.setParent(self)
                w.hide()

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(14)
        self._top_layout: QtWidgets.QHBoxLayout = top
        root.addLayout(top, 1)

        self.card = QtWidgets.QFrame()
        self.card.setObjectName("card")
        if not _USE_GL:
            self._card_shadow = StaticShadow(self.card, self)
        card_l = QtWidgets.QVBoxLayout(self.card)
        card_l.setContentsMargins(10, 14, 16, 8)
        self._card_layout: QtWidgets.QVBoxLayout = card_l
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
        pi.showGrid(x=True, y=True, alpha=0.11)
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
            pos=0, angle=0, pen=pg.mkPen((255, 255, 255, 48), width=1)
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

        sec_color: QtGui.QColor = _stereo_r_color(self.accent)
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

        self.glow_renderer: GlowRenderer[None | pg.PlotItem | Any] = GlowRenderer(pi)
        self.glow_renderer_r: GlowRenderer[None | pg.PlotItem | Any] = GlowRenderer(pi)
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
        self.stream_holder: QtWidgets.QWidget = holder

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
        self._sync_edge_clip_enabled()
        self.edge_clip_sw.toggled.connect(self._on_edge_clip_toggle)
        # «Об края» работает только вместе с плавной стабилизацией.
        self.trigger.toggled.connect(self._sync_edge_clip_enabled)
        self.smooth_sw.toggled.connect(self._sync_edge_clip_enabled)
        self.edge_clip_sw.hide()

        self.band_combo = QtWidgets.QComboBox(self)
        self.band_combo.addItem(tr("band_off"), "off")
        self.band_combo.addItem(tr("band_low"), "low")
        self.band_combo.addItem(tr("band_mid"), "mid")
        self.band_combo.addItem(tr("band_high"), "high")
        try:
            _bi: int = ["off", "low", "mid", "high"].index(self._band_kind)
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
            _idx: int = [k for k, _, _ in CHANNEL_LAYOUTS].index(self.ch_layout)
        except ValueError:
            _idx = 0
        self.ch_layout_combo.setCurrentIndex(_idx)
        self.ch_layout_combo.setFixedHeight(self.CTRL_H)
        self.ch_layout_combo.setFocusPolicy(no_focus)
        self.ch_layout_combo.currentIndexChanged.connect(self._on_layout_change)
        self._style_combo_popup(self.ch_layout_combo)
        self.ch_layout_combo.hide()
        self.cap_channels = None

        n_idx: int = (
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
        g_idx: int = GAIN_OPTIONS.index(g_cfg) if g_cfg in GAIN_OPTIONS else 0
        self.gain_combo: Segmented[list[tuple[str, int]]] = Segmented(
            [(f"×{g}", g) for g in GAIN_OPTIONS], g_idx, self
        )
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

        fps_idx: int = (
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
        self._stats_layout: QtWidgets.QVBoxLayout = col
        self.peak_tile = StatTile(self.COL_W)
        self.bpm_tile = StatTile(self.COL_W)
        self.rms_hz_tile = PairTile(self.COL_W)
        self.rms_tile: MiniTile = self.rms_hz_tile.left
        self.hz_tile: MiniTile = self.rms_hz_tile.right
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
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
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
        self._main_button_animations = (
            animate_button(self.theme_btn),
            animate_button(self.settings_btn, intro_delay_ms=90),
            animate_button(self.freeze_btn, intro_delay_ms=180),
        )

    _NOTE_NAMES = (
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    )

    def closeEvent(self, event) -> None:
        self._save_settings()
        try:
            self._capturer.stop()
        except Exception:
            pass
        super().closeEvent(event)