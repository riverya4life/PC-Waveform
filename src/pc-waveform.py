"""
PC Waveform — осциллограмма звука с ПК (WASAPI loopback) by riverya4life
Optimized: device selection, fast trigger, layered glow, smooth stabilization,
solo mode (RMB on plot, solid dark background, axes hidden),
i18n: ru / uk / en (default ru).
Settings: INI in %APPDATA%/PC Waveform/settings.ini
(lang, theme, window, gain, stabilization, glow, device, fade, fps).
Backend auto: OpenGL, если доступен, иначе растр.
Соло-режим: прозрачный фон прямо на GL (альфа-буфер + постоянная перерисовка).

UI v2: новый визуальный стиль — логотип, «пилюля» устройства, панель
управления с группами, переключатели-свитчи, сегментированные контролы,
круглые свотчи тем, карточки статистики.

«Затухание» (вкл. по умолчанию): мягкое затухание краёв окна + AGC/tanh.

Оптимизация: зеркальный кольцевой буфер (запись O(chunk) вместо сдвига
всего буфера), кадр рисуется только при появлении новых данных (в режиме
«без ограничения» — по сигналу от аудио-потока, без поллинга таймером),
свечение — один даунсэмплинг на канал, preallocated scratch-буферы,
in-place fade/clip, лёгкий триггер — без аллокаций в hot-path.
"""

import configparser
import math
import os
import sys
import threading
import time

_FORCED = os.environ.get("PCWAVEFORM_BACKEND", "").lower()
if _FORCED == "vulkan":
    os.environ.setdefault("QSG_RHI_BACKEND", "vulkan")
    os.environ.setdefault("QT_QUICK_BACKEND", "vulkan")

import numpy as np
import soundcard as sc
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# ---------------- настройки ----------------
SAMPLERATE = 48000
CHUNK = 256
UPDATE_MS = 1
WATCHDOG_MS = 16        # «без ограничения»: кадры идут от аудио-потока, таймер — страховка
MAX_N = 32768
TRIGGER_MAX_AGE = 1536
SAMPLE_OPTIONS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
GAIN_OPTIONS = [1, 2, 4, 8]
# FPS: 0 = максимум, -1 = по герцовке монитора, иначе — фикс. значение
FPS_OPTIONS = [0, -1, 30, 60, 75, 90, 120, 144, 165, 240]
CHANNEL_MODES = ["mono", "stereo"]
TICKS = {
    256: 64, 512: 128, 1024: 250, 2048: 500,
    4096: 500, 8192: 1000, 16384: 2000, 32768: 4000,
}

# Параметры демо-огибающей: только мягкое затухание по краям (без горбов).
DEMO_EDGE = 0.12        # доля окна на плавное затухание краёв


def _config_dir() -> str:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
            os.path.expanduser("~"), ".config"
        )
    return os.path.join(base, "PC Waveform")


CONFIG_PATH = os.path.join(_config_dir(), "settings.ini")

DEFAULT_CONFIG = {
    "lang": "ru", "theme": "mint", "n": 4096, "gain": 1,
    "stabilization": True, "glow": False, "channels": "mono",
    "device_id": None, "fade": True,
    "custom_color": "#ff6bcb",
    "fps": 0,
}


def _bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _validate(cfg: dict) -> dict:
    if cfg.get("lang") not in ("ru", "uk", "en"):
        cfg["lang"] = "ru"
    if cfg.get("n") not in SAMPLE_OPTIONS:
        try:
            cfg["n"] = int(cfg["n"])
        except Exception:
            cfg["n"] = 4096
        if cfg["n"] not in SAMPLE_OPTIONS:
            cfg["n"] = 4096
    if cfg.get("gain") not in GAIN_OPTIONS:
        try:
            cfg["gain"] = int(cfg["gain"])
        except Exception:
            cfg["gain"] = 1
        if cfg["gain"] not in GAIN_OPTIONS:
            cfg["gain"] = 1
    if cfg.get("fps") not in FPS_OPTIONS:
        try:
            cfg["fps"] = int(cfg["fps"])
        except Exception:
            cfg["fps"] = 0
        if cfg["fps"] not in FPS_OPTIONS:
            cfg["fps"] = 0
    cfg["stabilization"] = _bool(cfg.get("stabilization", True))
    cfg["glow"] = _bool(cfg.get("glow", False))
    cfg["fade"] = _bool(cfg.get("fade", True))
    if cfg.get("channels") not in CHANNEL_MODES:
        cfg["channels"] = "mono"
    did = cfg.get("device_id")
    if did is not None and str(did).strip() in ("", "None", "null"):
        cfg["device_id"] = None
    elif did is not None:
        cfg["device_id"] = str(did)
    return cfg


def _load_from_ini(path: str):
    cp = configparser.ConfigParser()
    try:
        if not cp.read(path, encoding="utf-8"):
            return None
    except Exception:
        return None
    cfg = dict(DEFAULT_CONFIG)
    if cp.has_section("ui"):
        s = cp["ui"]
        for k in ("lang", "theme", "channels", "custom_color"):
            if k in s:
                cfg[k] = s.get(k)
        for k in ("n", "gain", "fps"):
            if k in s:
                try:
                    cfg[k] = s.getint(k)
                except Exception:
                    pass
        for k in ("stabilization", "glow", "fade"):
            if k in s:
                cfg[k] = s.getboolean(k, DEFAULT_CONFIG[k])
        if "channels" in s:
            cfg["channels"] = s.get("channels", "mono").strip().lower()
    if cp.has_section("audio") and "device_id" in cp["audio"]:
        v = cp["audio"].get("device_id", "").strip()
        cfg["device_id"] = v if v and v.lower() not in ("none", "null") else None
    return cfg


def _load_config() -> dict:
    if os.path.isfile(CONFIG_PATH):
        loaded = _load_from_ini(CONFIG_PATH)
        if loaded:
            return _validate(loaded)
    return _validate(dict(DEFAULT_CONFIG))


def _save_config(cfg: dict):
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        cp = configparser.ConfigParser()
        cp["ui"] = {
            "lang": str(cfg.get("lang", DEFAULT_CONFIG["lang"])),
            "theme": str(cfg.get("theme", DEFAULT_CONFIG["theme"])),
            "n": str(int(cfg.get("n", DEFAULT_CONFIG["n"]))),
            "gain": str(int(cfg.get("gain", DEFAULT_CONFIG["gain"]))),
            "stabilization": "true" if _bool(cfg.get("stabilization", True)) else "false",
            "glow": "true" if _bool(cfg.get("glow", False)) else "false",
            "channels": str(cfg.get("channels", "mono")),
            "fade": "true" if _bool(cfg.get("fade", True)) else "false",
            "custom_color": str(cfg.get("custom_color", DEFAULT_CONFIG["custom_color"])),
            "fps": str(int(cfg.get("fps", 0))),
        }
        did = cfg.get("device_id")
        cp["audio"] = {"device_id": "" if did is None else str(did)}
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            cp.write(f)
    except Exception:
        pass


CONFIG = _load_config()

# ---------------- Локализация ----------------
LANGUAGES = [("Русский", "ru"), ("Українська", "uk"), ("English", "en")]
LANG = {"current": CONFIG["lang"]}

TRANSLATIONS = {
    "ru": {
        "brand": "PC Waveform",
        "main_title": "PC Waveform — осциллограмма системного звука (github: riverya4life)",
        "solo_title": "PC Waveform — соло (ПКМ или Esc — вернуться)",
        "subtitle": "осциллограмма системного звука",
        "freeze": "Заморозить", "resume": "Продолжить",
        "stabilization": "Стабилизация",
        "stabilization_tip": "Динамическая стабилизация: захват периода и фазы (как в FL Studio)",
        "glow": "Свечение",
        "fade": "Затухание",
        "fade_tip": "Мягкое затухание сигнала к краям окна + AGC",
        "window": "Окно", "gain": "Усиление", "stream": "Поток",
        "theme": "Тема", "language": "Язык", "channels": "Каналы",
        "fps": "Частота кадров",
        "fps_unlimited": "Максимум",
        "fps_monitor": "По монитору",
        "fps_tip": "Ограничение FPS осциллограммы.\n"
                   "• «Максимум» — без ограничения\n"
                   "• «По монитору» — фиксируется под герцовку экрана\n"
                   "• или конкретное значение FPS",
        "mono": "Моно", "stereo": "Стерео",
        "samples": "смп", "ms": "мс", "khz": "кГц",
        "peak": "PEAK", "rms": "RMS",
        "window_label": "ОКНО", "fps_label": "FPS", "backend_label": "БЭКЕНД",
        "hz_label": "ГЦ",
        "hz_tip": "Оценка основной частоты по периоду сигнала (как в FL Studio)",
        "backend_opengl": "OpenGL", "backend_vulkan": "Vulkan", "backend_raster": "Raster",
        "backend_tip": "Графический движок отрисовки с поддержкой OpenGL и Vulkan.\n"
        "Автовыбор: OpenGL (если доступен)\n\n"
        "• OpenGL — OpenGL-рендер pyqtgraph, максимальный FPS\n"
        "• Vulkan — Vulkan-контекст Qt + GL-рендер графика\n"
        "• Raster — без GL: QPainter (медленнее, но совместимо)\n\n"
        "Принудительно (для разработчиков): "
        "PCWAVEFORM_BACKEND=opengl|vulkan|raster",
        "clip": "CLIP", "paused": "пауза", "error_prefix": "Ошибка:",
        "searching": "поиск устройства…", "no_devices": "(нет устройств)",
        "solo_tip": "ПКМ — режим «соло»",
        "solo_pin": "Поверх всех окон",
        "solo_exit": "Выйти из соло (Esc)",
        "solo_freeze": "Заморозить / продолжить (Пробел)",
        "solo_bg": "Прозрачный фон",
        "theme_mint": "Мята", "theme_cyan": "Циан", "theme_azure": "Лазурь",
        "theme_indigo": "Индиго", "theme_violet": "Фиолет", "theme_pink": "Розовый",
        "theme_raspberry": "Малина", "theme_coral": "Коралл", "theme_amber": "Янтарь",
        "theme_lime": "Лайм", "theme_turquoise": "Бирюза", "theme_white": "Белый",
        "theme_custom": "Свой цвет",
        "theme_custom_tip": "Выбрать свой цвет темы",
        "theme_close": "Готово",
        "theme_presets": "ПАЛИТРА",
        "settings": "Настройки",
        "settings_tip": "Язык, каналы, частота кадров, стабилизация, свечение, "
                        "затухание, окно и усиление",
        "settings_general": "ОБЩИЕ",
        "settings_signal": "СИГНАЛ",
        "settings_effects": "ЭФФЕКТЫ",
    },
    "uk": {
        "brand": "PC Waveform",
        "main_title": "PC Waveform — осцилограма системного звуку (github: riverya4life)",
        "solo_title": "PC Waveform — соло (ПКМ або Esc — повернутися)",
        "subtitle": "осцилограма системного звуку",
        "freeze": "Заморозити", "resume": "Продовжити",
        "stabilization": "Стабілізація",
        "stabilization_tip": "Динамічна стабілізація: захоплення періоду і фази (як у FL Studio)",
        "glow": "Світіння",
        "fade": "Загасання",
        "fade_tip": "М'яке загасання сигналу до країв вікна + AGC",
        "window": "Вікно", "gain": "Підсилення", "stream": "Потік",
        "theme": "Тема", "language": "Мова", "channels": "Канали",
        "fps": "Частота кадрів",
        "fps_unlimited": "Максимум",
        "fps_monitor": "За монітором",
        "fps_tip": "Обмеження FPS осцилограми.\n"
                   "• «Максимум» — без обмеження\n"
                   "• «За монітором» — під герцовку екрана\n"
                   "• або конкретне значення FPS",
        "mono": "Моно", "stereo": "Стерео",
        "samples": "семп", "ms": "мс", "khz": "кГц",
        "peak": "PEAK", "rms": "RMS",
        "window_label": "ВІКНО", "fps_label": "FPS", "backend_label": "БЕКЕНД",
        "hz_label": "ГЦ",
        "hz_tip": "Оцінка основної частоти за періодом сигналу (як у FL Studio)",
        "backend_opengl": "OpenGL", "backend_vulkan": "Vulkan", "backend_raster": "Raster",
        "backend_tip": "Графічний рушій відтворення з підтримкою OpenGL і Vulkan.\n"
        "Автовибір: OpenGL (якщо доступний)\n\n"
        "• OpenGL — OpenGL-рендер pyqtgraph, максимальний FPS\n"
        "• Vulkan — Vulkan-контекст Qt + GL-рендер графика\n"
        "• Raster — без GL: QPainter (повільніше, але сумісно)\n\n"
        "Примусово (для розробників): "
        "PCWAVEFORM_BACKEND=opengl|vulkan|raster",
        "clip": "CLIP", "paused": "пауза", "error_prefix": "Помилка:",
        "searching": "пошук пристрою…", "no_devices": "(немає пристроїв)",
        "solo_tip": "ПКМ — режим «соло»",
        "solo_pin": "Поверх усіх вікон",
        "solo_exit": "Вийти з соло (Esc)",
        "solo_freeze": "Заморозити / продовжити (Пробіл)",
        "solo_bg": "Прозорий фон",
        "theme_mint": "М'ята", "theme_cyan": "Ціан", "theme_azure": "Лазур",
        "theme_indigo": "Індиго", "theme_violet": "Фіолет", "theme_pink": "Рожевий",
        "theme_raspberry": "Малина", "theme_coral": "Корал", "theme_amber": "Бурштин",
        "theme_lime": "Лайм", "theme_turquoise": "Бірюза", "theme_white": "Білий",
        "theme_custom": "Свій колір",
        "theme_custom_tip": "Вибрати свій колір теми",
        "theme_close": "Готово",
        "theme_presets": "ПАЛІТРА",
        "settings": "Налаштування",
        "settings_tip": "Мова, канали, частота кадрів, стабілізація, світіння, "
                        "загасання, вікно та підсилення",
        "settings_general": "ЗАГАЛЬНІ",
        "settings_signal": "СИГНАЛ",
        "settings_effects": "ЕФЕКТИ",
    },
    "en": {
        "brand": "PC Waveform",
        "main_title": "PC Waveform — system audio waveform (github: riverya4life)",
        "solo_title": "PC Waveform — solo (RMB or Esc — exit)",
        "subtitle": "system audio waveform",
        "freeze": "Freeze", "resume": "Resume",
        "stabilization": "Stabilization",
        "stabilization_tip": "Dynamic stabilization: period & phase lock (like FL Studio)",
        "glow": "Glow",
        "fade": "Fade",
        "fade_tip": "Soft edge fade of the signal + AGC",
        "window": "Window", "gain": "Gain", "stream": "Stream",
        "theme": "Theme", "language": "Language", "channels": "Channels",
        "fps": "Frame rate",
        "fps_unlimited": "Maximum",
        "fps_monitor": "Monitor sync",
        "fps_tip": "Oscilloscope FPS cap.\n"
                   "• “Maximum” — unlimited\n"
                   "• “Monitor sync” — locked to display refresh rate\n"
                   "• or a fixed FPS value",
        "mono": "Mono", "stereo": "Stereo",
        "samples": "smp", "ms": "ms", "khz": "kHz",
        "peak": "PEAK", "rms": "RMS",
        "window_label": "WINDOW", "fps_label": "FPS", "backend_label": "BACKEND",
        "hz_label": "HZ",
        "hz_tip": "Estimated fundamental frequency from signal period (like FL Studio)",
        "backend_opengl": "OpenGL", "backend_vulkan": "Vulkan", "backend_raster": "Raster",
        "backend_tip": "Graphics rendering backend with OpenGL and Vulkan support.\n"
        "Auto-select: OpenGL (if available)\n\n"
        "• OpenGL — pyqtgraph OpenGL render, max FPS\n"
        "• Vulkan — Qt Vulkan context + GL plot render\n"
        "• Raster — no GL: QPainter (slower, more compatible)\n\n"
        "Force (for developers): "
        "PCWAVEFORM_BACKEND=opengl|vulkan|raster",
        "clip": "CLIP", "paused": "paused", "error_prefix": "Error:",
        "searching": "searching device…", "no_devices": "(no devices)",
        "solo_tip": "RMB — solo mode",
        "solo_pin": "Always on top",
        "solo_exit": "Exit solo (Esc)",
        "solo_freeze": "Freeze / resume (Space)",
        "solo_bg": "Transparent background",
        "theme_mint": "Mint", "theme_cyan": "Cyan", "theme_azure": "Azure",
        "theme_indigo": "Indigo", "theme_violet": "Violet", "theme_pink": "Pink",
        "theme_raspberry": "Raspberry", "theme_coral": "Coral", "theme_amber": "Amber",
        "theme_lime": "Lime", "theme_turquoise": "Turquoise", "theme_white": "White",
        "theme_custom": "Custom",
        "theme_custom_tip": "Pick a custom theme colour",
        "theme_close": "Done",
        "theme_presets": "PALETTE",
        "settings": "Settings",
        "settings_tip": "Language, channels, frame rate, stabilization, glow, "
                        "fade, window and gain",
        "settings_general": "GENERAL",
        "settings_signal": "SIGNAL",
        "settings_effects": "EFFECTS",
    },
}


def tr(key: str) -> str:
    d = TRANSLATIONS.get(LANG["current"], TRANSLATIONS["ru"])
    return d.get(key, key)


def cap_text(key: str) -> str:
    return tr(key).upper()


THEMES = [
    ("mint", "#00ff8c"), ("cyan", "#19c8ff"), ("azure", "#4d8dff"),
    ("indigo", "#7c6cff"), ("violet", "#b26bff"), ("pink", "#ff50a0"),
    ("raspberry", "#ff2e63"), ("coral", "#ff6b4a"), ("amber", "#ffb000"),
    ("lime", "#b6ff2e"), ("turquoise", "#2ee6d6"), ("white", "#ebebeb"),
]
THEME_HEX = dict(THEMES)
DEFAULT_THEME = "mint"
# custom theme is allowed; its colour lives in CONFIG["custom_color"]
if CONFIG.get("theme") not in THEME_HEX and CONFIG.get("theme") != "custom":
    CONFIG["theme"] = DEFAULT_THEME
_cc = str(CONFIG.get("custom_color", DEFAULT_CONFIG["custom_color"]))
if not (_cc.startswith("#") and len(_cc) in (4, 7)):
    CONFIG["custom_color"] = DEFAULT_CONFIG["custom_color"]

try:
    import OpenGL  # noqa: F401
    _HAS_GL = True
except Exception:
    _HAS_GL = False

if _FORCED == "raster":
    _USE_GL = False
elif _FORCED == "vulkan":
    _USE_GL = _HAS_GL
else:
    _USE_GL = _HAS_GL

_BACKEND_ID = ("vulkan" if (_USE_GL and _FORCED == "vulkan")
               else ("opengl" if _USE_GL else "raster"))
_BACKEND_NAME_KEYS = {
    "opengl": "backend_opengl", "vulkan": "backend_vulkan", "raster": "backend_raster",
}

pg.setConfigOptions(useOpenGL=_USE_GL, antialias=False, foreground="#8b949e")

# Зеркальный кольцевой буфер: каждый сэмпл пишется в [i] и [i + RING_N], поэтому
# последние N сэмплов всегда лежат одним непрерывным срезом (без сдвигов всего
# массива на каждый чанк — раньше это было ~140 МБ/с memmove).
RING_N = MAX_N * 2
ring_l = np.zeros(RING_N * 2, dtype=np.float32)
ring_r = np.zeros(RING_N * 2, dtype=np.float32)
_ring_w = [0]
lock = threading.Lock()
state = {"device": None, "error": None, "total_samples": 0}


def _ring_push(ring, w, data, n):
    end = w + n
    if end <= RING_N:
        ring[w:end] = data
        ring[w + RING_N:end + RING_N] = data
    else:
        k = RING_N - w
        m = n - k
        ring[w:RING_N] = data[:k]
        ring[w + RING_N:RING_N * 2] = data[:k]
        ring[:m] = data[k:]
        ring[RING_N:RING_N + m] = data[k:]


def audio_snapshot(need, out_l, out_r):
    """Копия последних `need` сэмплов (L и R) в scratch-буферы. Возвращает total."""
    with lock:
        e = _ring_w[0] + RING_N
        s = e - need
        out_l[:need] = ring_l[s:e]
        out_r[:need] = ring_r[s:e]
        return state["total_samples"]


class _AudioBridge(QtCore.QObject):
    chunk = QtCore.Signal()


_bridge = _AudioBridge()
_chunk_pending = [False]


def _notify_chunk():
    """Из аудио-потока: «есть новые данные» (с коалесцированием событий)."""
    if _chunk_pending[0]:
        return
    _chunk_pending[0] = True
    try:
        _bridge.chunk.emit()
    except Exception:
        _chunk_pending[0] = False


# ---------------- Демо-огибающая: гладкая синусоида ----------------
def _demo_edge_window(t: np.ndarray, edge: float = DEMO_EDGE) -> np.ndarray:
    """Raised-cosine окно — плавное затухание к краям (без резкого обрыва)."""
    w = np.ones_like(t)
    left = t < edge
    right = t > 1.0 - edge
    if np.any(left):
        w[left] = 0.5 * (1.0 - np.cos(np.pi * t[left] / edge))
    if np.any(right):
        w[right] = 0.5 * (1.0 - np.cos(np.pi * (1.0 - t[right]) / edge))
    return w


def _demo_envelope(t_norm: np.ndarray) -> np.ndarray:
    """Огибающая демо-режима: ровная 1.0 в центре + мягкое затухание по краям.
    Никаких горбов и глубоких ям — только raised-cosine fade на границах окна."""
    return _demo_edge_window(t_norm, edge=DEMO_EDGE).astype(np.float32)


class GlowRenderer:
    SPECS = [(-3, 18), (-2, 34), (-1, 62), (1, 62), (2, 34), (3, 18)]
    DS_FACTOR = 5.0      # точек на пиксель (как autoDownsampleFactor у основной кривой)

    def __init__(self, plot_item):
        self._plot = plot_item
        self._color = QtGui.QColor(THEME_HEX[DEFAULT_THEME])
        self._enabled = False
        self._layers = []
        self._xbuf = None
        self._ybuf = None
        for dy_px, alpha in self.SPECS:
            item = pg.PlotCurveItem(pen=None)
            item.setZValue(-1)
            item.setVisible(False)
            plot_item.addItem(item)
            self._layers.append([item, dy_px, alpha, None])
        self._apply_pens()

    def _apply_pens(self):
        for item, _, alpha, _b in self._layers:
            col = QtGui.QColor(self._color)
            col.setAlpha(alpha)
            item.setPen(pg.mkPen(col, width=1))

    def set_color(self, color):
        self._color = QtGui.QColor(color)
        self._apply_pens()

    def set_enabled(self, on: bool):
        on = bool(on)
        was = self._enabled
        self._enabled = on
        for layer in self._layers:
            layer[0].setVisible(on)
        if was and not on:
            # освобождаем данные и буферы, пока свечение выключено
            for layer in self._layers:
                layer[0].clear()
                layer[3] = None
            self._xbuf = None
            self._ybuf = None

    def is_enabled(self) -> bool:
        return self._enabled

    def update(self, x, y, viewbox):
        if not self._enabled:
            return
        n = len(y)
        try:
            width_px = float(viewbox.width())
            dy_data = float(viewbox.viewPixelSize()[1])
        except Exception:
            width_px = 0.0
            dy_data = 0.0

        # Даунсэмплинг (mean) делаем один раз на канал, а не отдельно для каждого слоя
        ds = 1
        if width_px > 1.0:
            ds = max(1, int(n / (width_px * self.DS_FACTOR)))
        if ds > 1:
            m = n // ds
            if self._xbuf is None or self._xbuf.size < m:
                self._xbuf = np.empty(max(m, 1024), dtype=np.float32)
                self._ybuf = np.empty(max(m, 1024), dtype=np.float32)
            xs = np.add(x[:m * ds:ds], 0.5 * (ds - 1), out=self._xbuf[:m])
            ys = np.mean(y[:m * ds].reshape(m, ds), axis=1, out=self._ybuf[:m])
        else:
            m = n
            xs = x
            ys = y

        for layer in self._layers:
            item, dy_px, _, buf = layer
            if buf is None or buf.size < m:
                buf = layer[3] = np.empty(max(m, 1024), dtype=np.float32)
            b = buf[:m]
            np.add(ys, dy_px * dy_data, out=b)
            item.setData(xs, b, skipFiniteCheck=True)


class Capturer:
    def __init__(self):
        self._thread = None
        self._stop = threading.Event()

    def start(self, device_id):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.15)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(device_id, self._stop), daemon=True
        )
        self._thread.start()

    @staticmethod
    def _run(device_id, stop):
        try:
            mic = sc.get_microphone(id=str(device_id), include_loopback=True)
            state["device"] = str(mic.name)
            state["error"] = None
            with mic.recorder(samplerate=SAMPLERATE, channels=2, blocksize=CHUNK) as rec:
                while not stop.is_set():
                    data = rec.record(numframes=CHUNK)
                    if stop.is_set():
                        break
                    if data.ndim == 1:
                        left = right = data.astype(np.float32, copy=False)
                    else:
                        left = data[:, 0].astype(np.float32, copy=False)
                        right = (data[:, 1].astype(np.float32, copy=False)
                                 if data.shape[1] > 1 else left)
                    n = len(left)
                    if n == 0:
                        continue
                    if n > RING_N:
                        left = left[-RING_N:]
                        right = right[-RING_N:]
                        n = RING_N
                    with lock:
                        w = _ring_w[0]
                        _ring_push(ring_l, w, left, n)
                        _ring_push(ring_r, w, right, n)
                        _ring_w[0] = (w + n) % RING_N
                        state["total_samples"] += n
                    _notify_chunk()
        except Exception as e:
            if not stop.is_set():
                state["error"] = str(e)


def list_loopback_devices():
    out = []
    try:
        for m in sc.all_microphones(include_loopback=True):
            if getattr(m, "isloopback", False):
                out.append((str(m.name), str(m.id)))
    except Exception:
        pass
    if not out:
        try:
            sp = sc.default_speaker()
            out.append((str(sp.name), str(sp.name)))
        except Exception:
            pass
    return out


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
        return QtGui.QColor(max(0, c.red() - 40), max(0, c.green() - 40),
                            max(0, c.blue() - 40), 220)
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
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                           QtWidgets.QSizePolicy.Policy.Fixed)
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
        p.drawText(QtCore.QRectF(self.TW + 10, 0, self.width() - self.TW - 10, h),
                   QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
                   self.text())


class Segmented(QtWidgets.QFrame):
    currentIndexChanged = QtCore.Signal(int)

    def __init__(self, items, index=0, parent=None):
        super().__init__(parent)
        self.setObjectName("seg")
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum,
                           QtWidgets.QSizePolicy.Policy.Fixed)
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
        b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum,
                        QtWidgets.QSizePolicy.Policy.Fixed)
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
        if ev.type() in (QtCore.QEvent.Type.StyleChange,
                         QtCore.QEvent.Type.FontChange):
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
        col = (self._accent if active
               else QtGui.QColor("#e6edf3" if self._hover else "#9aa4af"))
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
                p.drawPolygon(QtGui.QPolygonF([
                    QtCore.QPointF(12.0, 9.0),
                    QtCore.QPointF(22.0, 16.0),
                    QtCore.QPointF(12.0, 23.0),
                ]))
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
    """Статичная тень карточки (nine-patch, рендерится один раз).

    Заменяет QGraphicsDropShadowEffect: в растровом режиме он заново размывал
    всю карточку вместе с графиком на КАЖДЫЙ кадр. Внешний вид тот же
    (blur 24, смещение 0×6, чёрный 80/255)."""
    PAD = 40       # на сколько тень выходит за границы карточки
    CORE = 120     # размер «ядра» карточки в исходном изображении
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
            # вырезаем силуэт самой карточки — она рисуется сверху своим виджетом
            ptr.setCompositionMode(
                QtGui.QPainter.CompositionMode.CompositionMode_DestinationOut)
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
            self.setGeometry(c.geometry().adjusted(
                -self.PAD, -self.PAD, self.PAD, self.PAD))
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
        # внутри карточки (она непрозрачная) тень не нужна — не рисуем зря
        inset = self.PAD + self.RADIUS
        region = ev.region()
        if w > 2 * inset and h > 2 * inset:
            region = region.subtracted(
                QtGui.QRegion(QtCore.QRect(inset, inset, w - 2 * inset, h - 2 * inset)))
        if region.isEmpty():
            return
        img = self._image()
        sw = img.width()
        a = sw // 2 - 1            # левая/верхняя часть; средняя полоска — 2 px
        b = sw - a - 2             # правая/нижняя часть
        if w < a + b or h < a + b:
            return
        p = QtGui.QPainter(self)
        p.setClipRegion(region)
        xs = [(0, a, 0, a), (a, 2, a, w - a - b), (a + 2, b, w - b, b)]
        ys = [(0, a, 0, a), (a, 2, a, h - a - b), (a + 2, b, h - b, b)]
        for sy, sh, dy, dh in ys:
            for sx, sw_, dx, dw in xs:
                p.drawImage(QtCore.QRect(dx, dy, dw, dh), img,
                            QtCore.QRect(sx, sy, sw_, sh))
        p.end()


class MainWindow(QtWidgets.QWidget):
    CTRL_H = 36
    ROW_H = 54
    _RESIZE_MARGIN = 8
    _MIN_W = 160
    _MIN_H = 40
    SOLO_W = 450
    SOLO_H = 150

    def __init__(self):
        super().__init__()
        self.setObjectName("root")
        self.setWindowTitle(tr("main_title"))
        self.resize(1040, 700)
        self.setMinimumSize(640, 560)

        self.n = CONFIG.get("n", 4096)
        self.frozen = False
        self.channels = CONFIG.get("channels", "mono")
        self.last = np.zeros(self.n, dtype=np.float32)
        self.last_r = np.zeros(self.n, dtype=np.float32)
        self.lim = 1.0
        self.prev_tpl = None
        self._crossing_smooth = None
        self._last_total = 0
        self._trig_alpha = 0.35
        self._period_smooth = 0.0   # период в сэмплах (динамическая стабилизация)
        self._lock_quality = 0.0    # 0..1 — насколько уверен захват

        # --- Затухание краёв (вкл. по умолчанию) + AGC ---
        self._fade_on = bool(CONFIG.get("fade", True))
        self._fade_blend = 1.0 if self._fade_on else 0.0  # 0=raw, 1=full fade
        self._fade_target = self._fade_blend
        self._demo_env_cache = None
        self._demo_agc = 0.0
        self._demo_agc_l = 0.0
        self._demo_agc_r = 0.0

        # --- Solo ---
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

        # --- FPS ---
        self._fps_value = int(CONFIG.get("fps", 0))

        # --- Hot-path scratch buffers (без аллокаций на каждом кадре) ---
        self._scr_t = np.empty(MAX_N * 2, dtype=np.float32)   # (L+R)/2 текущего окна
        self._scr_l = np.empty(MAX_N * 2, dtype=np.float32)   # копия окна L
        self._scr_r = np.empty(MAX_N * 2, dtype=np.float32)   # копия окна R
        self._fade_buf = np.empty(MAX_N, dtype=np.float32)
        self._fade_buf_r = np.empty(MAX_N, dtype=np.float32)
        self._disp_buf = np.empty(MAX_N, dtype=np.float32)
        self._disp_buf_r = np.empty(MAX_N, dtype=np.float32)
        self._xs = np.empty(MAX_N, dtype=np.float32)          # x со сдвигом триггера
        self._frame_sig = None
        self._was_hidden = False
        self._last_keepalive = 0.0
        self._chunk_driven = False
        self._chunk_connected = False

        self._kernel = np.ones(8, dtype=np.float32) / 8
        self._x = np.arange(self.n, dtype=np.float32)
        self.frames = 0
        self._hz_smooth = 0.0  # сглаженная оценка частоты (Гц)
        self.t0 = time.perf_counter()

        # --- Тема ---
        self.theme_id = CONFIG.get("theme", DEFAULT_THEME)
        self.theme_btns = {}
        self._theme_dialog = None
        # accent may be custom
        if self.theme_id == "custom":
            self.accent = str(CONFIG.get("custom_color", DEFAULT_CONFIG["custom_color"]))
        else:
            self.accent = THEME_HEX.get(self.theme_id, THEME_HEX[DEFAULT_THEME])
        self._current_color = QtGui.QColor(self.accent)
        self._theme_anim = None

        self._capturer = Capturer()
        self._loading = True
        self._set_chunk_driven(int(CONFIG.get("fps", 0)) == 0)

        self._build_ui()
        self._apply_theme(self.theme_id, animate=False)
        self.set_n(self.n)
        self.set_gain(CONFIG.get("gain", 1))
        if CONFIG.get("glow", False):
            self.glow_renderer.set_enabled(True)
            if self.channels == "stereo":
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
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(150)

        shortcut_cls = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut
        shortcut_cls(QtGui.QKeySequence("Space"), self, activated=self.freeze_btn.click)
        self._solo_esc = shortcut_cls(
            QtGui.QKeySequence("Escape"), self, activated=self._exit_solo
        )
        self._solo_esc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._solo_esc.setEnabled(False)

    def _fit_pill(self):
        if not hasattr(self, "pill"):
            return
        used = (52 + self.logo.width() + 14
                + self.title_holder.sizeHint().width() + 14 + 12)
        self.pill.setFixedWidth(max(220, min(480, self.width() - used)))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
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
        CONFIG["channels"] = self.channels
        CONFIG["fade"] = self._fade_on
        CONFIG["fps"] = int(self._fps_value)
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

    # ---------- FPS ----------
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
            idx = FPS_OPTIONS.index(self._fps_value)
        except ValueError:
            idx = 0
        self.fps_combo.setCurrentIndex(idx)
        self.fps_combo.blockSignals(False)
        self.fps_combo.setToolTip(tr("fps_tip"))

    def _on_fps_change(self, idx):
        if idx < 0:
            return
        v = self.fps_combo.itemData(idx)
        if v is None or v == self._fps_value:
            return
        self._fps_value = int(v)
        self._apply_fps()
        if not getattr(self, "_loading", False):
            CONFIG["fps"] = self._fps_value
            _save_config(CONFIG)

    def _apply_fps(self):
        """Пересчёт интервала таймера отрисовки по выбранному режиму."""
        interval = UPDATE_MS
        v = int(getattr(self, "_fps_value", 0))
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
        if getattr(self, "timer", None) is not None:
            self.timer.setInterval(interval)

    def _set_chunk_driven(self, on: bool):
        """«Без ограничения»: рисуем по сигналу от аудио-потока, а не поллингом."""
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
        _chunk_pending[0] = False      # страховка: сигнал не может «застрять»
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
            p.drawPolygon(QtGui.QPolygonF([
                QtCore.QPointF(5.5, 3.0),
                QtCore.QPointF(15.0, 9.0),
                QtCore.QPointF(5.5, 15.0),
            ]))
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
            for sw in (self.trigger, self.glow, getattr(self, "fade_sw", None)):
                if sw is not None:
                    sw.setText("")
            if getattr(self, "_cap_stab", None):
                self._cap_stab.setText(tr("stabilization"))
            if getattr(self, "_cap_glow", None):
                self._cap_glow.setText(tr("glow"))
            if getattr(self, "_cap_fade", None):
                self._cap_fade.setText(tr("fade"))
        if self.cap_window:
            self.cap_window.setText(tr("window"))
        if self.cap_gain:
            self.cap_gain.setText(tr("gain"))
        self.cap_stream.setText(tr("stream"))
        self.cap_theme.setText(tr("theme"))
        if self.cap_lang:
            self.cap_lang.setText(tr("language"))
        if self.cap_channels:
            self.cap_channels.setText(tr("channels"))
        if hasattr(self, "theme_btn"):
            self.theme_btn.setText(tr("theme"))
        if hasattr(self, "settings_btn"):
            self.settings_btn.setText(tr("settings"))
            self.settings_btn.setToolTip(tr("settings_tip"))
        if hasattr(self, "cap_settings") and self.cap_settings:
            self.cap_settings.setText(cap_text("settings"))
        for i, m in enumerate(CHANNEL_MODES):
            self.ch_combo.setItemText(i, tr(m))
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
        self.card.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, False)
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
        self._root_layout.insertWidget(1, self.card)
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
            QFrame#card {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 0;
            }}
            QToolTip {{
                background: #161b22; color: #c9d1d9;
                border: 1px solid #30363d; border-radius: 6px;
                padding: 5px 10px; font-family: 'Segoe UI'; font-size: 12px;
            }}
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
            vp.setUpdateBehavior(ub.NoPartialUpdate if transparent else ub.PartialUpdate)
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
        self.solo_title = QtWidgets.QLabel(tr("brand"))
        self.solo_title.setObjectName("stitle")
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
        lay.addWidget(self.solo_title)
        lay.addSpacing(10)
        lay.addWidget(self.solo_read)
        lay.addStretch(1)
        lay.addWidget(self.solo_freeze)
        lay.addWidget(self.solo_bg)
        lay.addWidget(self.solo_pin)
        lay.addWidget(self.solo_close)
        self._hud = hud
        self._hud_lay = lay
        self._solo_vals = ("−∞", "−∞")
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
            QFrame#hud {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0,0,0,175), stop:1 rgba(0,0,0,0));
                border: none;
            }}
            QLabel {{ color: #c9d1d9; font-family: 'Segoe UI';
                font-size: 12px; background: transparent; }}
            QLabel#stitle {{ color: #f0f6fc; font-size: 13px; font-weight: 700; }}
            QLabel#sread {{ color: #8b949e;
                font-family: 'Cascadia Mono', 'Consolas', monospace;
                font-size: 12px; }}
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
        title_w = self.solo_title.sizeHint().width() + sp
        fm = self.solo_read.fontMetrics()
        p, r = self._solo_vals
        cands = [
            f"PEAK {p}  ·  RMS {r} dBFS",
            f"PEAK {p} · RMS {r}",
            f"{p} · {r} dB",
        ]
        show_title, text = True, ""
        found = False
        for st in (True, False):
            left = avail - dot_w - (title_w if st else 0) - 10
            for t in cands:
                if fm.horizontalAdvance(t) <= left:
                    show_title, text, found = st, t, True
                    break
            if found:
                break
        if not found:
            show_title = avail - dot_w - title_w >= 0
            text = ""
        self.solo_title.setVisible(show_title)
        self.solo_read.setVisible(bool(text))
        self.solo_read.setText(text)

    def _hud_show(self, on: bool):
        if not self._solo:
            return
        self._hud_on = on
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
        return (local.x() <= m,
                local.x() >= geo.width() - m,
                local.y() <= m,
                local.y() >= geo.height() - m)

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
            hasattr(self.plot, "viewport") and obj is self.plot.viewport())
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
        if (self._theme_anim is not None
                and self._theme_anim.state() == QtCore.QAbstractAnimation.State.Running):
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

    def _apply_theme_color(self, color: QtGui.QColor):
        c = QtGui.QColor(color)
        c.setAlpha(255)
        self.curve.setPen(pg.mkPen(c, width=1.7))
        c2 = _stereo_r_color(color.name() if hasattr(color, "name") else str(self.accent))
        self.curve_r.setPen(pg.mkPen(c2, width=1.5))
        self.glow_renderer.set_color(color)
        self.glow_renderer_r.set_color(c2)
        self.logo.set_accent(c)
        self._update_freeze_icon()
        self.trigger.set_accent(c)
        self.glow.set_accent(c)
        if hasattr(self, "fade_sw"):
            self.fade_sw.set_accent(c)
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
            prev.setStyleSheet(
                f"background: {self.accent}; border-radius: 6px;"
            )
        dlg = getattr(self, "_theme_dialog", None)
        if dlg is not None and dlg.isVisible():
            dlg.setStyleSheet(self._theme_dialog_qss())

    def _theme_color(self, theme_id: str) -> str:
        if theme_id == "custom":
            return str(CONFIG.get("custom_color", DEFAULT_CONFIG["custom_color"]))
        return THEME_HEX.get(theme_id, THEME_HEX[DEFAULT_THEME])

    def _apply_theme(self, theme_id: str, animate: bool = True):
        btns = getattr(self, "theme_btns", None)
        if btns and theme_id in btns:
            btns[theme_id].setChecked(True)
        self.theme_id = theme_id
        target = self._theme_color(theme_id)
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
        self.device_lbl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                      QtWidgets.QSizePolicy.Policy.Preferred)
        self.device_err = False
        pill_l.addWidget(self.dot, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        pill_l.addWidget(self.device_lbl, 1)
        header.addWidget(self.pill, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        root.addLayout(header)

        self.card = QtWidgets.QFrame()
        self.card.setObjectName("card")
        if not _USE_GL:
            self._card_shadow = StaticShadow(self.card, self)
        card_l = QtWidgets.QVBoxLayout(self.card)
        card_l.setContentsMargins(10, 14, 16, 8)
        self._card_layout = card_l
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
        self.plot.addItem(self.zero_line)
        limit_pen = pg.mkPen((255, 255, 255, 45), width=1,
                             style=QtCore.Qt.PenStyle.DashLine)
        self.lim_hi = pg.InfiniteLine(pos=1, angle=0, pen=limit_pen)
        self.lim_lo = pg.InfiniteLine(pos=-1, angle=0, pen=limit_pen)
        self.plot.addItem(self.lim_hi)
        self.plot.addItem(self.lim_lo)
        self.curve = pg.PlotDataItem(pen=None)
        self.curve.setZValue(0)
        self.curve.setDownsampling(auto=True, method="mean")
        if not _USE_GL:
            self.curve.setClipToView(True)
        self.plot.addItem(self.curve)
        self.curve_r = pg.PlotDataItem(pen=None)
        self.curve_r.setZValue(-0.5)
        self.curve_r.setDownsampling(auto=True, method="mean")
        if not _USE_GL:
            self.curve_r.setClipToView(True)
        self.curve_r.setVisible(self.channels == "stereo")
        self.plot.addItem(self.curve_r)
        self.glow_renderer = GlowRenderer(pi)
        self.glow_renderer_r = GlowRenderer(pi)
        if CONFIG.get("glow", False) and self.channels == "stereo":
            self.glow_renderer_r.set_enabled(True)
        card_l.addWidget(self.plot)
        root.addWidget(self.card, 1)
        self.plot.installEventFilter(self)
        try:
            self.plot.viewport().installEventFilter(self)
        except Exception:
            pass
        self.card.installEventFilter(self)
        self._build_solo_hud()

        self.panel = QtWidgets.QFrame()
        self.panel.setObjectName("panel")
        panel_l = QtWidgets.QVBoxLayout(self.panel)
        panel_l.setContentsMargins(20, 16, 20, 16)
        ctl = FlowLayout(hspacing=22, vspacing=14)
        panel_l.addLayout(ctl)

        self.freeze_btn = QtWidgets.QPushButton(self._freeze_text(False))
        self.freeze_btn.setObjectName("primary")
        self.freeze_btn.setCheckable(True)
        self.freeze_btn.setFocusPolicy(no_focus)
        self.freeze_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.freeze_btn.setFixedHeight(self.ROW_H)
        self.freeze_btn.setIconSize(QtCore.QSize(18, 18))
        self.freeze_btn.toggled.connect(self.on_freeze)
        ctl.addWidget(self.freeze_btn)

        # Поток
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.setMinimumWidth(210)
        self.device_combo.setFixedHeight(self.CTRL_H)
        self.device_combo.setFocusPolicy(no_focus)
        self.device_combo.currentIndexChanged.connect(self._on_device_change)
        self._style_combo_popup(self.device_combo)
        holder, self.cap_stream = self._caption_holder("stream", self.device_combo)
        ctl.addWidget(holder)

        # Тема
        self.theme_btn = QtWidgets.QPushButton(tr("theme"))
        self.theme_btn.setFocusPolicy(no_focus)
        self.theme_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.theme_btn.setFixedHeight(self.CTRL_H)
        self.theme_btn.clicked.connect(self._open_theme_dialog)
        holder, self.cap_theme = self._caption_holder("theme", self.theme_btn)
        ctl.addWidget(holder)

        # Настройки (язык, каналы, fps, стабилизация, свечение, затухание, окно, усиление)
        self.settings_btn = QtWidgets.QPushButton(tr("settings"))
        self.settings_btn.setFocusPolicy(no_focus)
        self.settings_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.settings_btn.setFixedHeight(self.CTRL_H)
        self.settings_btn.setToolTip(tr("settings_tip"))
        self.settings_btn.clicked.connect(self._open_settings_dialog)
        holder, self.cap_settings = self._caption_holder("settings", self.settings_btn)
        ctl.addWidget(holder)

        # --- Виджеты для диалога «Настройки» (parent=self, скрыты до открытия) ---
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

        ch_cfg = CONFIG.get("channels", "mono")
        ch_idx = CHANNEL_MODES.index(ch_cfg) if ch_cfg in CHANNEL_MODES else 0
        self.ch_combo = Segmented([(tr(m), m) for m in CHANNEL_MODES], ch_idx, self)
        self.ch_combo.setFixedHeight(self.CTRL_H)
        self.ch_combo.currentIndexChanged.connect(self._on_channels_change)
        self.ch_combo.hide()
        self.cap_channels = None

        n_idx = (SAMPLE_OPTIONS.index(CONFIG.get("n", 4096))
                 if CONFIG.get("n") in SAMPLE_OPTIONS
                 else SAMPLE_OPTIONS.index(4096))
        self.n_combo = QtWidgets.QComboBox(self)
        self.n_combo.addItems([f"{v} {tr('samples')}" for v in SAMPLE_OPTIONS])
        self.n_combo.setCurrentIndex(n_idx)
        self.n_combo.setFixedHeight(self.CTRL_H)
        self.n_combo.setFocusPolicy(no_focus)
        self._style_combo_popup(self.n_combo)
        self.n_combo.currentIndexChanged.connect(
            lambda i: self.set_n(SAMPLE_OPTIONS[i]))
        self.n_combo.hide()

        g_cfg = CONFIG.get("gain", 1)
        g_idx = GAIN_OPTIONS.index(g_cfg) if g_cfg in GAIN_OPTIONS else 0
        self.gain_combo = Segmented([(f"×{g}", g) for g in GAIN_OPTIONS], g_idx, self)
        self.gain_combo.setFixedHeight(self.CTRL_H)
        self.gain_combo.currentIndexChanged.connect(
            lambda i: self.set_gain(GAIN_OPTIONS[i]))
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

        # FPS
        fps_idx = (FPS_OPTIONS.index(self._fps_value)
                   if self._fps_value in FPS_OPTIONS else 0)
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

        root.addWidget(self.panel)

        stats = FlowLayout(hspacing=12, vspacing=10)
        self.peak_tile = StatTile(290)
        self.rms_tile = StatTile(160)
        self.hz_tile = StatTile(130)
        self.info_tile = StatTile(240)
        self.fps_tile = StatTile(110)
        self.bknd_tile = StatTile(140)
        self.hz_tile.setToolTip(tr("hz_tip"))
        for t in (self.peak_tile, self.rms_tile, self.hz_tile, self.info_tile,
                  self.fps_tile, self.bknd_tile):
            stats.addWidget(t)
        root.addLayout(stats)

    @staticmethod
    def _style_combo_popup(combo):
        view = QtWidgets.QListView()
        view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        combo.setView(view)
        popup = combo.view().window()
        popup.setWindowFlags(
            popup.windowFlags()
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.NoDropShadowWindowHint
        )
        popup.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

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

    # ---------- Диалог «Настройки» ----------
    def _settings_dialog_qss(self) -> str:
        a = self.accent
        top = _mix_dark(a, 0.040)
        bot = _mix_dark(a, 0.014)
        return f"""
            QDialog {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {top}, stop:1 {bot});
            }}
            QLabel {{ background: transparent; color: #c9d1d9; }}
            QLabel#stTitle {{
                color: #f0f6fc; font-size: 16px; font-weight: 700;
            }}
            QLabel#stCap {{
                color: #6e7681; font-size: 10px; font-weight: 600;
                letter-spacing: 1px;
                padding: 2px 4px 6px 4px;
            }}
            QLabel#stRowLabel {{
                color: #e6edf3; font-size: 13px; font-weight: 500;
            }}
            QFrame#stCard {{
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 12px;
            }}
            QFrame#stSep {{
                background: rgba(255,255,255,0.06);
                max-height: 1px; min-height: 1px;
                border: none;
            }}
            QPushButton#stDone {{
                background: {a}; color: #0d1117; border: none;
                border-radius: 10px; padding: 10px 28px;
                font-weight: 700; font-size: 13px;
            }}
            QPushButton#stDone:hover {{ background: {_rgba(a, 0.85)}; }}
            QComboBox {{
                background: rgba(255,255,255,0.08);
                color: #c9d1d9;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 10px;
                padding: 6px 12px;
                min-width: 120px;
                min-height: 28px;
            }}
            QComboBox:hover {{ border-color: {a}; }}
            QComboBox::drop-down {{ border: none; width: 22px; }}
            QComboBox QAbstractItemView {{
                background: #161b22; color: #c9d1d9;
                border: 1px solid #30363d; border-radius: 10px;
                selection-background-color: {a};
                selection-color: #0d1117;
            }}
            QFrame#seg {{
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 10px;
            }}
            QPushButton#segbtn {{
                background: transparent; color: #8b949e;
                border: none; border-radius: 8px;
                padding: 6px 12px; font-weight: 500;
                font-family: 'Segoe UI'; font-size: 12px;
                min-width: 0px;
            }}
            QPushButton#segbtn:hover {{
                background: rgba(255,255,255,0.10); color: #ffffff;
            }}
            QPushButton#segbtn:checked {{
                background: {a}; color: #0d1117; font-weight: 700;
            }}
        """

    def _settings_row(self, label_key, widget, store_cap=None):
        """Строка настроек: подпись слева, контроль справа."""
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
        dlg.setFixedWidth(440)
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

        # --- Общие ---
        w1, c1, self._st_cap_general = _section("settings_general")
        c1.addWidget(self._settings_row("language", self.lang_combo, "cap_lang"))
        c1.addWidget(self._settings_sep())
        c1.addWidget(self._settings_row("channels", self.ch_combo, "cap_channels"))
        c1.addWidget(self._settings_sep())
        c1.addWidget(self._settings_row("fps", self.fps_combo, "cap_fps"))
        lay.addLayout(w1)

        # --- Сигнал ---
        w2, c2, self._st_cap_signal = _section("settings_signal")
        c2.addWidget(self._settings_row("window", self.n_combo, "cap_window"))
        c2.addWidget(self._settings_sep())
        c2.addWidget(self._settings_row("gain", self.gain_combo, "cap_gain"))
        lay.addLayout(w2)

        # --- Эффекты ---
        w3, c3, self._st_cap_effects = _section("settings_effects")
        c3.addWidget(self._settings_row("stabilization", self.trigger, "_cap_stab"))
        c3.addWidget(self._settings_sep())
        c3.addWidget(self._settings_row("glow", self.glow, "_cap_glow"))
        c3.addWidget(self._settings_sep())
        c3.addWidget(self._settings_row("fade", self.fade_sw, "_cap_fade"))
        lay.addLayout(w3)

        for w in (self.lang_combo, self.n_combo, self.gain_combo, self.ch_combo,
                  self.fps_combo, self.trigger, self.glow, self.fade_sw):
            w.setVisible(True)
            w.show()

        for sw in (self.trigger, self.glow, self.fade_sw):
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
        for w in (self.lang_combo, self.n_combo, self.gain_combo, self.ch_combo,
                  self.fps_combo, self.trigger, self.glow, self.fade_sw):
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
            self.cap_channels.setText(tr("channels"))
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
        for sw in (self.trigger, self.glow, self.fade_sw):
            sw.setText("")
        self.trigger.setToolTip(tr("stabilization_tip"))
        self.fade_sw.setToolTip(tr("fade_tip"))
        for i, m in enumerate(CHANNEL_MODES):
            self.ch_combo.setItemText(i, tr(m))
        self.trigger.set_accent(self.accent)
        self.glow.set_accent(self.accent)
        self.fade_sw.set_accent(self.accent)
        # refresh sample labels
        idx = self.n_combo.currentIndex()
        self.n_combo.blockSignals(True)
        self.n_combo.clear()
        self.n_combo.addItems([f"{v} {tr('samples')}" for v in SAMPLE_OPTIONS])
        self.n_combo.setCurrentIndex(idx)
        self.n_combo.blockSignals(False)
        # refresh fps combo (labels depend on language)
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
            QDialog {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {top}, stop:1 {bot});
            }}
            QLabel {{ background: transparent; color: #c9d1d9; }}
            QLabel#thTitle {{
                color: #f0f6fc; font-size: 15px; font-weight: 700;
            }}
            QLabel#thCap {{
                color: #6e7681; font-size: 10px; font-weight: 600;
                letter-spacing: 1px;
            }}
            QLabel#thSub {{ color: #8b949e; font-size: 12px; }}
            QFrame#thCard {{
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 12px;
            }}
            QPushButton#thCustom {{
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 12px; color: #c9d1d9;
                text-align: left; padding: 10px 14px; font-size: 12px;
            }}
            QPushButton#thCustom:hover {{
                background: rgba(255,255,255,0.08);
                border-color: {a};
            }}
            QPushButton#thDone {{
                background: {a}; color: #0d1117; border: none;
                border-radius: 10px; padding: 10px 28px;
                font-weight: 700; font-size: 13px;
            }}
            QPushButton#thDone:hover {{ background: {_rgba(a, 0.85)}; }}
            QFrame#thPreview {{
                background: {a}; border-radius: 6px;
            }}
        """

    def _build_theme_dialog(self):
        """Создаёт (один раз) окно выбора темы."""
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
        self.custom_swatch.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        custom_inner.addWidget(self.custom_swatch)
        custom_txt_box = QtWidgets.QVBoxLayout()
        custom_txt_box.setSpacing(1)
        self._theme_custom_title = QtWidgets.QLabel(tr("theme_custom"))
        self._theme_custom_title.setStyleSheet(
            "font-size: 12px; font-weight: 600; color: #e6edf3; background: transparent;"
        )
        self._theme_custom_hex = QtWidgets.QLabel(cc)
        self._theme_custom_hex.setStyleSheet(
            "font-size: 11px; color: #8b949e; background: transparent;"
            "font-family: 'Cascadia Mono', 'Consolas', monospace;"
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
        for tid, btn in self.theme_btns.items():
            if tid == "custom":
                btn.setToolTip(tr("theme_custom_tip"))
            else:
                btn.setToolTip(tr(f"theme_{tid}"))
                if hasattr(btn, "_name_lbl"):
                    btn._name_lbl.setText(tr(f"theme_{tid}"))
        if self.theme_id in self.theme_btns:
            self.theme_btns[self.theme_id].setChecked(True)
        if self.theme_id == "custom":
            name = tr("theme_custom")
            hex_c = CONFIG.get("custom_color", "")
        else:
            name = tr(f"theme_{self.theme_id}")
            hex_c = THEME_HEX.get(self.theme_id, "")
        self._theme_dlg_label.setText(f"{name}  ·  {hex_c}")
        self._theme_custom_hex.setText(
            str(CONFIG.get("custom_color", DEFAULT_CONFIG["custom_color"]))
        )
        chip = self._theme_color(self.theme_id)
        self._theme_preview.setStyleSheet(
            f"background: {chip}; border-radius: 6px;"
        )
        self._theme_dialog.setStyleSheet(self._theme_dialog_qss())

    def _open_theme_dialog(self):
        if not getattr(self, "_theme_dialog", None):
            self._build_theme_dialog()
        self._refresh_theme_dialog_texts()
        self._theme_dialog.adjustSize()
        self._theme_dialog.exec()

    def _on_custom_theme(self):
        """Open colour picker; apply chosen colour as custom theme."""
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

    def _combo(self, caption_key, items, index, layout):
        combo = QtWidgets.QComboBox()
        combo.addItems(items)
        combo.setFixedHeight(self.CTRL_H)
        self._style_combo_popup(combo)
        combo.setCurrentIndex(index)
        combo.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        holder, cap = self._caption_holder(caption_key, combo)
        layout.addWidget(holder)
        return combo, cap

    def set_n(self, n):
        self.n = n
        self._x = np.arange(n, dtype=np.float32)
        self.last = np.zeros(n, dtype=np.float32)
        self.last_r = np.zeros(n, dtype=np.float32)
        self.prev_tpl = None
        self._crossing_smooth = None
        self._period_smooth = 0.0
        self._lock_quality = 0.0
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
        self.glow_renderer_r.set_enabled(on and self.channels == "stereo")
        CONFIG["glow"] = bool(on)
        _save_config(CONFIG)

    def _on_channels_change(self, idx):
        if idx < 0:
            return
        mode = self.ch_combo.itemData(idx)
        if not mode or mode == self.channels:
            return
        self.channels = mode
        stereo = mode == "stereo"
        self.curve_r.setVisible(stereo)
        self.lim_hi.setVisible(True)
        self.lim_lo.setVisible(True)
        self.glow_renderer_r.set_enabled(stereo and self.glow_renderer.is_enabled())
        self.prev_tpl = None
        self._crossing_smooth = None
        self._period_smooth = 0.0
        self._lock_quality = 0.0
        if not getattr(self, "_loading", False):
            CONFIG["channels"] = mode
            _save_config(CONFIG)

    # ---------- Затухание краёв ----------
    def _rebuild_demo_caches(self):
        n = self.n
        if n <= 0:
            self._demo_env_cache = None
            return
        t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
        self._demo_env_cache = _demo_envelope(t)

    def _apply_demo_state(self):
        if self._fade_on or self._fade_blend > 0.01:
            self._rebuild_demo_caches()
        else:
            self._demo_env_cache = None
        self.curve.setDownsampling(auto=True, method="mean")
        self.curve_r.setDownsampling(auto=True, method="mean")

    def _on_fade_toggle(self, on: bool):
        self._fade_on = bool(on)
        self._fade_target = 1.0 if self._fade_on else 0.0
        if self._fade_on:
            self._rebuild_demo_caches()
        CONFIG["fade"] = self._fade_on
        _save_config(CONFIG)

    def _demo_shape(self, raw: np.ndarray, agc_attr: str = "_demo_agc") -> np.ndarray:
        """Мягкое затухание по краям (совместимость). В hot-path не используется —
        логика инлайнится в update_frame с preallocated буферами."""
        b = float(self._fade_blend)
        if b < 0.001 or self._demo_env_cache is None:
            return raw.astype(np.float32, copy=False)
        env = 1.0 + (self._demo_env_cache - 1.0) * b
        return (raw * env).astype(np.float32)

    # ---------- Отрисовка (hot-path, без аллокаций) ----------
    def _ui_hidden(self) -> bool:
        """Нечего рисовать: окно (или соло-карточка) свёрнуто/скрыто."""
        w = self.card if self._solo else self
        return w.isMinimized() or not w.isVisible()

    def _keepalive(self):
        # Соло-режим (прозрачный GL) — изредка перерисовываем и без новых данных
        if self._solo:
            t = time.perf_counter()
            if t - self._last_keepalive >= 0.1:
                self._last_keepalive = t
                self.plot.viewport().update()

    def update_frame(self):
        # Окно свёрнуто/скрыто — рисовать некому (захват звука продолжается)
        if self._ui_hidden():
            self._was_hidden = True
            return

        # Плавная анимация переключения затухания (~350 мс)
        animating = False
        if abs(self._fade_blend - self._fade_target) > 1e-4:
            animating = True
            self._fade_blend += (self._fade_target - self._fade_blend) * 0.08
            if abs(self._fade_blend - self._fade_target) < 0.002:
                self._fade_blend = self._fade_target
                if self._fade_blend < 0.01 and not self._fade_on:
                    self._demo_env_cache = None

        if self.frozen:
            self._keepalive()
            return

        n = self.n
        n2 = n + n
        stereo = (self.channels == "stereo")
        trig_on = self.trigger.isChecked()
        vb = self.plot.getViewBox()

        # Нет новых сэмплов и ничего не менялось — кадр был бы точной копией прошлого
        sig = (n, self.lim, stereo, trig_on, self._fade_blend,
               self.glow_renderer.is_enabled(), self.glow_renderer_r.is_enabled(),
               vb.width(), vb.height())
        if (not animating and not self._was_hidden and sig == self._frame_sig
                and state["total_samples"] == self._last_total):
            self._keepalive()
            return
        self._frame_sig = sig
        if self._was_hidden:
            self._was_hidden = False
            self.prev_tpl = None
            self._crossing_smooth = None
        self.frames += 1

        use_fade = (self._fade_blend > 0.01
                    and self._demo_env_cache is not None
                    and self._demo_env_cache.size == n)

        # --- Один заход под лок: копируем только то, что реально нужно ---
        need = n2 if trig_on else n
        total = audio_snapshot(need, self._scr_l, self._scr_r)
        shift = total - self._last_total
        self._last_total = total
        if trig_on or not stereo:
            # (L+R)/2 — считаем уже вне лока
            np.add(self._scr_l[:need], self._scr_r[:need], out=self._scr_t[:need])
            np.multiply(self._scr_t[:need], 0.5, out=self._scr_t[:need])
        data_t = self._scr_t[:need]

        # --- Триггер ---
        if trig_on:
            start = n
            x = self._x
            found = self.find_trigger(data_t, shift)
            if found is not None:
                start, x_off = found
                start = max(0, min(start, need - n))
                x = np.add(self._x, x_off, out=self._xs[:n])
        else:
            start = 0
            x = self._x

        lim = self.lim

        # --- Огибающая: считаем один раз на кадр ---
        if use_fade:
            b = float(self._fade_blend)
            if b >= 0.999:
                env = self._demo_env_cache
            else:
                env = 1.0 + (self._demo_env_cache - 1.0) * b
        else:
            env = None

        # --- Отрисовка (всё in-place, без свежих аллокаций) ---
        if stereo:
            raw_l = self._scr_l[start:start + n]
            raw_r = self._scr_r[start:start + n]
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
            np.clip(view_l, -lim, lim, out=self._disp_buf[:n])
            np.clip(view_r, -lim, lim, out=self._disp_buf_r[:n])
            shown_l = self._disp_buf[:n]
            shown_r = self._disp_buf_r[:n]
            self.curve.setData(x, shown_l, skipFiniteCheck=True)
            self.curve_r.setData(x, shown_r, skipFiniteCheck=True)
            self.glow_renderer.update(x, shown_l, vb)
            self.glow_renderer_r.update(x, shown_r, vb)
        else:
            raw = data_t[start:start + n]
            if env is not None:
                np.multiply(raw, env, out=self._fade_buf[:n])
                view = self._fade_buf[:n]
            else:
                view = raw
            self.last = view
            self.last_r = view
            np.clip(view, -lim, lim, out=self._disp_buf[:n])
            shown = self._disp_buf[:n]
            self.curve.setData(x, shown, skipFiniteCheck=True)
            if self.curve_r.isVisible():
                self.curve_r.clear()
            self.glow_renderer.update(x, shown, vb)

    def find_trigger(self, data, shift):
        """Динамическая стабилизация (в духе FL Studio):
        1) ищем нулевые пересечения с гистерезисом
        2) оцениваем период и держим фазовый захват
        3) при смене высоты тона период плавно подстраивается"""
        n = self.n
        L = len(data)
        half = n // 2
        s = np.convolve(data, self._kernel, mode="same")
        s -= s.mean()
        amp = float(np.max(np.abs(s[half:L - half])))
        if amp < 1e-4:
            self._lock_quality *= 0.9
            return None

        h = (0.12 + 0.06 * (1.0 - min(1.0, amp))) * amp
        # Состояние гистерезиса меняется только в «ненулевых» точках, поэтому фронты
        # (−1 → +1) находим по соседним ненулевым отсчётам — без O(L) заливки.
        hi = s > h
        nz_idx = np.flatnonzero(hi | (s < -h))
        if nz_idx.size == 0:
            return None
        up = hi[nz_idx]
        edges = nz_idx[1:][(~up[:-1]) & up[1:]]
        if edges.size == 0:
            return None
        zc = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0)) + 1
        if zc.size == 0:
            return None
        pos = np.searchsorted(zc, edges, side="right") - 1
        valid = pos >= 0
        if not valid.any():
            return None
        cand = zc[pos[valid]]
        cand = cand[(cand >= half) & (cand <= L - half)]
        if cand.size == 0:
            return None

        periods = np.diff(cand.astype(np.float64))
        min_p = SAMPLERATE / 8000.0
        max_p = SAMPLERATE / 20.0
        periods = periods[(periods >= min_p) & (periods <= max_p)]
        period_est = 0.0
        if periods.size > 0:
            period_est = float(np.median(periods))
            if self._period_smooth < 1.0:
                self._period_smooth = period_est
            else:
                ratio = period_est / max(self._period_smooth, 1.0)
                pa = 0.35 if (ratio > 1.12 or ratio < 0.88) else 0.12
                self._period_smooth = (
                    self._period_smooth * (1.0 - pa) + period_est * pa
                )

        predicted = None
        if self._crossing_smooth is not None and shift >= 0:
            predicted = self._crossing_smooth - shift
            if self._period_smooth > 8.0 and predicted is not None:
                pass

        recent = cand[cand >= L - half - TRIGGER_MAX_AGE]
        if recent.size:
            cand = recent
        cand = cand[-32:]

        if self._period_smooth > 16.0:
            T = int(min(max(self._period_smooth * 0.5, 48), min(256, n // 2)))
        else:
            T = min(192, n // 2)
        T = max(32, T - (T % 2))
        hT = T // 2

        best = len(cand) - 1
        best_score = -1e9
        cand_f = cand.astype(np.float64)

        for i, c in enumerate(cand):
            score = 0.0
            if (self.prev_tpl is not None
                    and len(self.prev_tpl) == T
                    and c - hT >= 0 and c + hT <= L):
                ref = data[c - hT:c + hT].astype(np.float32, copy=False)
                ref = ref - ref.mean()
                nrm = float(np.linalg.norm(ref)) + 1e-9
                score += float(np.dot(ref / nrm, self.prev_tpl)) * 1.0
            if predicted is not None:
                if self._period_smooth > 8.0:
                    d = (float(c) - predicted) % self._period_smooth
                    d = min(d, self._period_smooth - d)
                    score -= (d / self._period_smooth) * 0.9
                else:
                    score -= abs(float(c) - predicted) * 0.004
            score += 0.015 * (i / max(len(cand) - 1, 1))
            if score > best_score:
                best_score = score
                best = i

        if best_score < -0.5 and self._lock_quality < 0.2:
            best = len(cand) - 1

        c = int(cand[best])
        if c - hT >= 0 and c + hT <= L:
            ref = data[c - hT:c + hT].astype(np.float32, copy=False)
            ref = ref - ref.mean()
            self.prev_tpl = ref / (np.linalg.norm(ref) + 1e-9)

        a, b = float(s[c - 1]), float(s[c])
        frac = a / (a - b) if a != b else 0.0
        frac = min(max(frac, 0.0), 1.0)
        crossing = (c - 1) + frac

        q = 0.5
        if best_score > 0.5:
            q = 0.9
        elif best_score > 0.2:
            q = 0.7
        elif best_score > 0.0:
            q = 0.45
        else:
            q = 0.2
        self._lock_quality = self._lock_quality * 0.7 + q * 0.3

        if self._crossing_smooth is None or shift <= 0:
            self._crossing_smooth = crossing
        else:
            expected = self._crossing_smooth - shift
            delta = crossing - expected
            if self._period_smooth > 8.0:
                half_p = self._period_smooth * 0.5
                while delta > half_p:
                    delta -= self._period_smooth
                while delta < -half_p:
                    delta += self._period_smooth
            alpha = 0.18 + 0.35 * self._lock_quality
            if abs(delta) > n * 0.35:
                self._crossing_smooth = crossing
            else:
                self._crossing_smooth = crossing - (1.0 - alpha) * delta

        cs = self._crossing_smooth
        start = int(cs) - half
        x_off = -(cs - int(cs))
        return start, x_off

    @staticmethod
    def _db(x):
        return "−∞" if x < 1e-6 else f"{20 * np.log10(x):+.1f}".replace("-", "−")

    def _estimate_hz(self, seg: np.ndarray) -> float:
        if seg is None or len(seg) < 32:
            return 0.0
        x = seg.astype(np.float64, copy=False)
        if len(x) >= 8:
            k = np.ones(5, dtype=np.float64) / 5.0
            x = np.convolve(x, k, mode="same")
        amp = float(np.max(np.abs(x)))
        if amp < 1e-4:
            return 0.0
        zc = np.flatnonzero((x[:-1] < 0.0) & (x[1:] >= 0.0))
        if zc.size < 2:
            return 0.0
        periods = np.diff(zc.astype(np.float64))
        min_p = SAMPLERATE / 8000.0
        max_p = SAMPLERATE / 20.0
        periods = periods[(periods >= min_p) & (periods <= max_p)]
        if periods.size == 0:
            return 0.0
        p = float(np.median(periods))
        if p < 1.0:
            return 0.0
        return float(SAMPLERATE / p)

    def _format_hz(self, hz: float) -> str:
        if hz < 1.0:
            return "—"
        if hz < 100.0:
            return f"{hz:.1f}"
        if hz < 1000.0:
            return f"{hz:.0f}"
        return f"{hz / 1000.0:.2f}k"

    def update_stats(self):
        if self._ui_hidden():
            self.frames = 0
            self.t0 = time.perf_counter()
            return
        dim = '<span style="color:#6e7681">'
        seg = self.last
        if self.channels == "stereo" and len(self.last_r):
            if len(seg):
                peak_l = float(np.max(np.abs(seg)))
                peak_r = float(np.max(np.abs(self.last_r)))
                peak = max(peak_l, peak_r)
                rms = float(np.sqrt(0.5 * (np.mean(seg * seg)
                                           + np.mean(self.last_r * self.last_r))))
            else:
                peak = peak_l = peak_r = rms = 0.0
            clip_l = peak_l >= 0.999
            clip_r = peak_r >= 0.999
            col_l = "#f85149" if clip_l else self.accent
            col_r = "#f85149" if clip_r else _stereo_r_color(self.accent).name()
            mark_l = f" {tr('clip')}" if clip_l else ""
            mark_r = f" {tr('clip')}" if clip_r else ""
            self.peak_tile.set(
                tr("peak") + " · dBFS",
                f'<span style="color:{col_l}">L {self._db(peak_l)}{mark_l}</span>'
                f"{dim} · </span>"
                f'<span style="color:{col_r}">R {self._db(peak_r)}{mark_r}</span>')
            self.rms_tile.set(tr("rms"), f"{self._db(rms)} dBFS")
        else:
            if len(seg):
                peak = float(np.max(np.abs(seg)))
                rms = float(np.sqrt(np.mean(seg * seg)))
            else:
                peak = rms = 0.0
            peak_color = "#f85149" if peak >= 0.999 else self.accent
            clip = f"  {tr('clip')}" if peak >= 0.999 else ""
            self.peak_tile.set(
                tr("peak"),
                f'<span style="color:{peak_color}">{self._db(peak)} dBFS{clip}</span>')
            self.rms_tile.set(tr("rms"), f"{self._db(rms)} dBFS")
        if self._solo:
            self._solo_vals = (self._db(peak), self._db(rms))
            self._fit_hud()

        hz_raw = self._estimate_hz(seg)
        if hz_raw > 1.0:
            if self._hz_smooth < 1.0:
                self._hz_smooth = hz_raw
            else:
                a = 0.25 if abs(hz_raw - self._hz_smooth) / max(self._hz_smooth, 1.0) > 0.15 else 0.12
                self._hz_smooth = self._hz_smooth * (1.0 - a) + hz_raw * a
        else:
            self._hz_smooth *= 0.85
            if self._hz_smooth < 5.0:
                self._hz_smooth = 0.0
        hz_txt = self._format_hz(self._hz_smooth)
        self.hz_tile.set(tr("hz_label"), hz_txt)
        if self.hz_tile.toolTip() != tr("hz_tip"):
            self.hz_tile.setToolTip(tr("hz_tip"))

        self.info_tile.set(
            tr("window_label"),
            f'{self.n} {tr("samples")} {dim}·</span> '
            f'{self.n / SAMPLERATE * 1000:.1f} {tr("ms")}')
        now = time.perf_counter()
        fps = self.frames / max(now - self.t0, 1e-3)
        self.frames = 0
        self.t0 = now
        self.fps_tile.set(tr("fps_label"),
                          tr("paused") if self.frozen else f"{fps:.0f}")
        be_name = tr(_BACKEND_NAME_KEYS.get(_BACKEND_ID, "backend_raster"))
        self.bknd_tile.set(tr("backend_label"), be_name)
        if self.bknd_tile.toolTip() != tr("backend_tip"):
            self.bknd_tile.setToolTip(tr("backend_tip"))
        has_err = bool(state["error"])
        if has_err != self.device_err:
            self.device_err = has_err
            self.device_lbl.setStyleSheet("color: #f85149;" if has_err else "")
            self.dot.setStyleSheet(
                "background: #f85149; border-radius: 4px;" if has_err else "")
        if has_err:
            full = f'{tr("error_prefix")} {state["error"]}'
        else:
            name = state["device"] or tr("searching")
            full = f'{name}  ·  {SAMPLERATE / 1000:g} {tr("khz")}'
        if self.pill.toolTip() != full:
            self.pill.setToolTip(full)
        elided = self.device_lbl.fontMetrics().elidedText(
            full, QtCore.Qt.TextElideMode.ElideRight,
            max(self.device_lbl.width() - 4, 20))
        if self.device_lbl.text() != elided:
            self.device_lbl.setText(elided)

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


def _app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


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


def _set_app_user_model_id():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "riverya4life.PCWaveform")
    except Exception:
        pass


def _apply_taskbar_icon(win, icon: QtGui.QIcon):
    if sys.platform != "win32":
        return
    try:
        from PyQt5.QtWinExtras import QWinTaskbarButton  # type: ignore
    except Exception:
        try:
            from PyQt6.QtWinExtras import QWinTaskbarButton  # type: ignore
        except Exception:
            return
    try:
        btn = QWinTaskbarButton(win)
        btn.setWindow(win.windowHandle())
        btn.setOverlayIcon(icon)
        win._taskbar_button = btn
    except Exception:
        pass


def _setup_gl_format():
    if not _USE_GL:
        return
    try:
        fmt = QtGui.QSurfaceFormat(QtGui.QSurfaceFormat.defaultFormat())
        fmt.setAlphaBufferSize(8)
        QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    except Exception:
        pass


def main():
    _set_app_user_model_id()
    _setup_gl_format()
    app = pg.mkQApp("PC Waveform")
    win = MainWindow()
    win.show()
    dark_titlebar(win)
    QtCore.QTimer.singleShot(0, lambda: _apply_taskbar_icon(win, win.windowIcon()))
    pg.exec()


if __name__ == "__main__":
    main()