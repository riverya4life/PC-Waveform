"""
PC Waveform — осциллограмма звука с ПК (WASAPI loopback) by riverya4life
Optimized: device selection, fast trigger, layered glow, smooth stabilization,
solo mode (RMB on plot, solid dark background, axes hidden),
i18n: ru / uk / en (default ru).
Settings: INI in %APPDATA%/PC Waveform/settings.ini
(lang, theme, window, gain, stabilization, glow, device).
Backend auto: OpenGL, если доступен, иначе растр.
Соло-режим: прозрачный фон прямо на GL (альфа-буфер + постоянная перерисовка).

UI v2: новый визуальный стиль — логотип, «пилюля» устройства, панель
управления с группами, переключатели-свитчи, сегментированные контролы,
круглые свотчи тем, карточки статистики.
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
MAX_N = 32768
TRIGGER_MAX_AGE = 1536
SAMPLE_OPTIONS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
GAIN_OPTIONS = [1, 2, 4, 8]
CHANNEL_MODES = ["mono", "stereo"]
TICKS = {
    256: 64,
    512: 128,
    1024: 250,
    2048: 500,
    4096: 500,
    8192: 1000,
    16384: 2000,
    32768: 4000,
}


# AppData\PC Waveform\settings.ini  (Windows);  ~/.config/PC Waveform/  (прочее)
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
    "lang": "ru",
    "theme": "mint",
    "n": 4096,
    "gain": 1,
    "stabilization": True,
    "glow": False,
    "channels": "mono",
    "device_id": None,
}


def _bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "on")


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
    cfg["stabilization"] = _bool(cfg.get("stabilization", True))
    cfg["glow"] = _bool(cfg.get("glow", False))
    if cfg.get("channels") not in CHANNEL_MODES:
        cfg["channels"] = "mono"
    did = cfg.get("device_id")
    if did is not None and str(did).strip() in ("", "None", "null"):
        cfg["device_id"] = None
    elif did is not None:
        cfg["device_id"] = str(did)
    return cfg


def _load_from_ini(path: str) -> dict | None:
    cp = configparser.ConfigParser()
    try:
        read = cp.read(path, encoding="utf-8")
        if not read:
            return None
    except Exception:
        return None
    cfg = dict(DEFAULT_CONFIG)
    if cp.has_section("ui"):
        s = cp["ui"]
        if "lang" in s:
            cfg["lang"] = s.get("lang")
        if "theme" in s:
            cfg["theme"] = s.get("theme")
        if "n" in s:
            try:
                cfg["n"] = s.getint("n")
            except Exception:
                pass
        if "gain" in s:
            try:
                cfg["gain"] = s.getint("gain")
            except Exception:
                pass
        if "stabilization" in s:
            cfg["stabilization"] = s.getboolean("stabilization", True)
        if "glow" in s:
            cfg["glow"] = s.getboolean("glow", False)
        if "channels" in s:
            cfg["channels"] = s.get("channels", "mono").strip().lower()
    if cp.has_section("audio"):
        s = cp["audio"]
        if "device_id" in s:
            v = s.get("device_id", "").strip()
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
            "stabilization": (
                "true" if _bool(cfg.get("stabilization", True)) else "false"
            ),
            "glow": "true" if _bool(cfg.get("glow", False)) else "false",
            "channels": str(cfg.get("channels", "mono")),
        }
        did = cfg.get("device_id")
        cp["audio"] = {
            "device_id": "" if did is None else str(did),
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            cp.write(f)
    except Exception:
        pass


CONFIG = _load_config()

# ---------------- Локализация ----------------
LANGUAGES = [
    ("Русский", "ru"),
    ("Українська", "uk"),
    ("English", "en"),
]


LANG = {"current": CONFIG["lang"]}

TRANSLATIONS = {
    "ru": {
        "brand": "PC Waveform",
        "main_title": "PC Waveform — осциллограмма системного звука (github: riverya4life)",
        "solo_title": "PC Waveform — соло (ПКМ или Esc — вернуться)",
        "subtitle": "осциллограмма системного звука",
        "freeze": "Заморозить",
        "resume": "Продолжить",
        "stabilization": "Стабилизация",
        "stabilization_tip": "Триггер по фронту с гистерезисом + захват фазы",
        "glow": "Свечение",
        "window": "Окно",
        "gain": "Усиление",
        "stream": "Поток",
        "theme": "Тема",
        "language": "Язык",
        "channels": "Каналы",
        "mono": "Моно",
        "stereo": "Стерео",
        "samples": "смп",
        "ms": "мс",
        "khz": "кГц",
        "peak": "PEAK",
        "rms": "RMS",
        "window_label": "ОКНО",
        "fps_label": "FPS",
        "backend_label": "БЭКЕНД",
        "backend_opengl": "OpenGL",
        "backend_vulkan": "Vulkan",
        "backend_raster": "Raster",
        "backend_tip": "Графический движок отрисовки с поддержкой OpenGL и Vulkan.\n"
        "Автовыбор: OpenGL (если доступен)\n"
        "\n"
        "• OpenGL — OpenGL-рендер pyqtgraph, максимальный FPS\n"
        "• Vulkan — Vulkan-контекст Qt + GL-рендер графика\n"
        "• Raster — без GL: QPainter (медленнее, но совместимо)\n"
        "\n"
        "Принудительно (для разработчиков): "
        "PCWAVEFORM_BACKEND=opengl|vulkan|raster",
        "clip": "CLIP",
        "paused": "пауза",
        "error_prefix": "Ошибка:",
        "searching": "поиск устройства…",
        "no_devices": "(нет устройств)",
        "solo_tip": "ПКМ — режим «соло»",
        "solo_pin": "Поверх всех окон",
        "solo_exit": "Выйти из соло (Esc)",
        "solo_freeze": "Заморозить / продолжить (Пробел)",
        "solo_bg": "Прозрачный фон",
        "theme_mint": "Мята",
        "theme_cyan": "Циан",
        "theme_azure": "Лазурь",
        "theme_indigo": "Индиго",
        "theme_violet": "Фиолет",
        "theme_pink": "Розовый",
        "theme_raspberry": "Малина",
        "theme_coral": "Коралл",
        "theme_amber": "Янтарь",
        "theme_lime": "Лайм",
        "theme_turquoise": "Бирюза",
        "theme_white": "Белый",
    },
    "uk": {
        "brand": "PC Waveform",
        "main_title": "PC Waveform — осцилограма системного звуку (github: riverya4life)",
        "solo_title": "PC Waveform — соло (ПКМ або Esc — повернутися)",
        "subtitle": "осцилограма системного звуку",
        "freeze": "Заморозити",
        "resume": "Продовжити",
        "stabilization": "Стабілізація",
        "stabilization_tip": "Тригер по фронту з гістерезисом + захоплення фази",
        "glow": "Світіння",
        "window": "Вікно",
        "gain": "Підсилення",
        "stream": "Потік",
        "theme": "Тема",
        "language": "Мова",
        "channels": "Канали",
        "mono": "Моно",
        "stereo": "Стерео",
        "samples": "семп",
        "ms": "мс",
        "khz": "кГц",
        "peak": "PEAK",
        "rms": "RMS",
        "window_label": "ВІКНО",
        "fps_label": "FPS",
        "backend_label": "БЕКЕНД",
        "backend_opengl": "OpenGL",
        "backend_vulkan": "Vulkan",
        "backend_raster": "Raster",
        "backend_tip": "Графічний рушій відтворення з підтримкою OpenGL і Vulkan.\n"
        "Автовибір: OpenGL (якщо доступний)\n"
        "\n"
        "• OpenGL — OpenGL-рендер pyqtgraph, максимальний FPS\n"
        "• Vulkan — Vulkan-контекст Qt + GL-рендер графіка\n"
        "• Raster — без GL: QPainter (повільніше, але сумісно)\n"
        "\n"
        "Примусово (для розробників): "
        "PCWAVEFORM_BACKEND=opengl|vulkan|raster",
        "clip": "CLIP",
        "paused": "пауза",
        "error_prefix": "Помилка:",
        "searching": "пошук пристрою…",
        "no_devices": "(немає пристроїв)",
        "solo_tip": "ПКМ — режим «соло»",
        "solo_pin": "Поверх усіх вікон",
        "solo_exit": "Вийти з соло (Esc)",
        "solo_freeze": "Заморозити / продовжити (Пробіл)",
        "solo_bg": "Прозорий фон",
        "theme_mint": "М'ята",
        "theme_cyan": "Ціан",
        "theme_azure": "Лазур",
        "theme_indigo": "Індиго",
        "theme_violet": "Фіолет",
        "theme_pink": "Рожевий",
        "theme_raspberry": "Малина",
        "theme_coral": "Корал",
        "theme_amber": "Бурштин",
        "theme_lime": "Лайм",
        "theme_turquoise": "Бірюза",
        "theme_white": "Білий",
    },
    "en": {
        "brand": "PC Waveform",
        "main_title": "PC Waveform — system audio waveform (github: riverya4life)",
        "solo_title": "PC Waveform — solo (RMB or Esc — exit)",
        "subtitle": "system audio waveform",
        "freeze": "Freeze",
        "resume": "Resume",
        "stabilization": "Stabilization",
        "stabilization_tip": "Edge trigger with hysteresis + phase lock",
        "glow": "Glow",
        "window": "Window",
        "gain": "Gain",
        "stream": "Stream",
        "theme": "Theme",
        "language": "Language",
        "channels": "Channels",
        "mono": "Mono",
        "stereo": "Stereo",
        "samples": "smp",
        "ms": "ms",
        "khz": "kHz",
        "peak": "PEAK",
        "rms": "RMS",
        "window_label": "WINDOW",
        "fps_label": "FPS",
        "backend_label": "BACKEND",
        "backend_opengl": "OpenGL",
        "backend_vulkan": "Vulkan",
        "backend_raster": "Raster",
        "backend_tip": "Graphics rendering backend with OpenGL and Vulkan support.\n"
        "Auto-select: OpenGL (if available)\n"
        "\n"
        "• OpenGL — pyqtgraph OpenGL render, max FPS\n"
        "• Vulkan — Qt Vulkan context + GL plot render\n"
        "• Raster — no GL: QPainter (slower, more compatible)\n"
        "\n"
        "Force (for developers): "
        "PCWAVEFORM_BACKEND=opengl|vulkan|raster",
        "clip": "CLIP",
        "paused": "paused",
        "error_prefix": "Error:",
        "searching": "searching device…",
        "no_devices": "(no devices)",
        "solo_tip": "RMB — solo mode",
        "solo_pin": "Always on top",
        "solo_exit": "Exit solo (Esc)",
        "solo_freeze": "Freeze / resume (Space)",
        "solo_bg": "Transparent background",
        "theme_mint": "Mint",
        "theme_cyan": "Cyan",
        "theme_azure": "Azure",
        "theme_indigo": "Indigo",
        "theme_violet": "Violet",
        "theme_pink": "Pink",
        "theme_raspberry": "Raspberry",
        "theme_coral": "Coral",
        "theme_amber": "Amber",
        "theme_lime": "Lime",
        "theme_turquoise": "Turquoise",
        "theme_white": "White",
    },
}


def tr(key: str) -> str:
    d = TRANSLATIONS.get(LANG["current"], TRANSLATIONS["ru"])
    return d.get(key, key)


def cap_text(key: str) -> str:
    """Подпись поля — капсом (мелкий заголовок над контролом)."""
    return tr(key).upper()


# ---------------- Темы ----------------
# Ключ — стабильный id, hex — цвет. Отображаемое имя берётся из tr().
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
if CONFIG.get("theme") not in THEME_HEX:
    CONFIG["theme"] = DEFAULT_THEME
# -------------------------------------------

# --- Автовыбор графического бэкенда ---
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

# Внутренний id: opengl | vulkan | raster
_BACKEND_ID = (
    "vulkan"
    if (_USE_GL and _FORCED == "vulkan")
    else ("opengl" if _USE_GL else "raster")
)
# Отображаемое имя берётся из tr() в UI
_BACKEND_NAME_KEYS = {
    "opengl": "backend_opengl",
    "vulkan": "backend_vulkan",
    "raster": "backend_raster",
}

pg.setConfigOptions(
    useOpenGL=_USE_GL,
    antialias=False,
    foreground="#8b949e",
)

# --- Общие буферы (L / R) ---
buf_l = np.zeros(MAX_N * 2, dtype=np.float32)
buf_r = np.zeros(MAX_N * 2, dtype=np.float32)
lock = threading.Lock()
state = {"device": None, "error": None, "total_samples": 0}


# ---------------- Многослойное свечение ----------------
class GlowRenderer:
    SPECS = [
        (-3, 18),
        (-2, 34),
        (-1, 62),
        (1, 62),
        (2, 34),
        (3, 18),
    ]

    def __init__(self, plot_item):
        self._plot = plot_item
        self._color = QtGui.QColor(THEME_HEX[DEFAULT_THEME])
        self._enabled = False
        self._layers = []

        for dy_px, alpha in self.SPECS:
            item = pg.PlotDataItem(pen=None)
            item.setZValue(-1)
            item.setDownsampling(auto=True, method="mean")
            if not _USE_GL:
                item.setClipToView(True)
            item.setVisible(False)
            plot_item.addItem(item)
            self._layers.append([item, dy_px, alpha])

        self._apply_pens()

    def _apply_pens(self):
        for item, _, alpha in self._layers:
            col = QtGui.QColor(self._color)
            col.setAlpha(alpha)
            item.setPen(pg.mkPen(col, width=1))

    def set_color(self, color):
        self._color = QtGui.QColor(color)
        self._apply_pens()

    def set_enabled(self, on: bool):
        self._enabled = bool(on)
        for item, _, _ in self._layers:
            item.setVisible(self._enabled)

    def is_enabled(self) -> bool:
        return self._enabled

    def update(self, x, y, viewbox):
        if not self._enabled:
            return
        try:
            _, dy_data = viewbox.viewPixelSize()
        except Exception:
            dy_data = 0.0
        for item, dy_px, _ in self._layers:
            item.setData(x, y + dy_px * dy_data, skipFiniteCheck=True)


# --- Управление захватом ---
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
            with mic.recorder(
                samplerate=SAMPLERATE, channels=2, blocksize=CHUNK
            ) as rec:
                while not stop.is_set():
                    data = rec.record(numframes=CHUNK)
                    if stop.is_set():
                        break
                    if data.ndim == 1:
                        left = right = data.astype(np.float32, copy=False)
                    else:
                        left = data[:, 0].astype(np.float32, copy=False)
                        right = (
                            data[:, 1].astype(np.float32, copy=False)
                            if data.shape[1] > 1
                            else left
                        )
                    n = len(left)
                    if n == 0:
                        continue
                    with lock:
                        buf_l[:-n] = buf_l[n:]
                        buf_l[-n:] = left
                        buf_r[:-n] = buf_r[n:]
                        buf_r[-n:] = right
                        state["total_samples"] += n
        except Exception as e:
            if not stop.is_set():
                state["error"] = str(e)


def list_loopback_devices():
    out = []
    try:
        mics = sc.all_microphones(include_loopback=True)
        for m in mics:
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


# ---------------- Цветовые хелперы ----------------
def _mix_dark(accent_hex: str, strength: float = 0.07) -> str:
    c = QtGui.QColor(accent_hex)
    r = min(255, int(c.red() * strength) + 6)
    g = min(255, int(c.green() * strength) + 7)
    b = min(255, int(c.blue() * strength) + 9)
    return f"#{r:02x}{g:02x}{b:02x}"


def _surface(accent_hex: str) -> str:
    """Цвет поверхности карточки / фона графика (приподнят над фоном окна)."""
    return _mix_dark(accent_hex, 0.060)


def _rgba(color, a: float) -> str:
    c = QtGui.QColor(color)
    return f"rgba({c.red()},{c.green()},{c.blue()},{int(max(0.0, min(1.0, a)) * 255)})"


def _lerp_color(a: QtGui.QColor, b: QtGui.QColor, t: float) -> QtGui.QColor:
    t = max(0.0, min(1.0, t))
    return QtGui.QColor(*(int(x + (y - x) * t) for x, y in zip(a.getRgb(), b.getRgb())))


def _stereo_r_color(accent_hex: str) -> QtGui.QColor:
    """Тот же цвет, что у L, но чуть темнее."""
    c = QtGui.QColor(accent_hex)
    h, s, v, a = c.getHsv()
    if h < 0:
        return QtGui.QColor(
            max(0, c.red() - 40),
            max(0, c.green() - 40),
            max(0, c.blue() - 40),
            220,
        )
    v2 = max(40, int(v * 0.62))
    s2 = min(255, int(s * 1.05)) if s > 0 else s
    return QtGui.QColor.fromHsv(h, s2, v2, 220)


# ---------------- Стили ----------------
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
        color: #c9d1d9;
        font-family: 'Segoe UI';
        font-size: 12px;
        background: transparent;
    }}
    QLabel#title {{
        color: #f0f6fc;
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.2px;
    }}
    QLabel#sub {{
        color: #7d8590;
        font-size: 12px;
    }}
    QLabel#cap {{
        color: #6e7681;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1px;
    }}

    /* --- Пилюля устройства --- */
    QFrame#pill {{
        background: rgba(255,255,255,10);
        border: 1px solid rgba(255,255,255,20);
        border-radius: 16px;
    }}
    QLabel#device {{
        color: #9aa4af;
        font-size: 12px;
    }}
    QLabel#dot {{
        background: {accent};
        border-radius: 4px;
    }}

    /* --- Карточка графика и панель управления --- */
    QFrame#card {{
        background: {surf};
        border: 1px solid rgba(255,255,255,18);
        border-radius: 16px;
    }}
    QFrame#panel {{
        background: rgba(255,255,255,7);
        border: 1px solid rgba(255,255,255,16);
        border-radius: 16px;
    }}

    /* --- Карточки статистики --- */
    QFrame#tile {{
        background: rgba(255,255,255,7);
        border: 1px solid rgba(255,255,255,16);
        border-radius: 12px;
    }}
    QLabel#tilecap {{
        color: #6e7681;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1px;
    }}
    QLabel#tileval {{
        color: #e6edf3;
        font-family: 'Cascadia Mono', 'Consolas', monospace;
        font-size: 14px;
        font-weight: 600;
    }}

    /* --- Кнопки --- */
    QPushButton {{
        background: rgba(255,255,255,10);
        color: #c9d1d9;
        border: 1px solid rgba(255,255,255,26);
        border-radius: 10px;
        padding: 8px 18px;
        font-family: 'Segoe UI';
        font-size: 12px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background: rgba(255,255,255,20);
        border-color: {accent};
        color: #ffffff;
    }}
    QPushButton:checked {{
        background: {accent};
        color: #0d1117;
        border-color: {accent};
        font-weight: 600;
    }}
    QPushButton#primary {{
        background: {_rgba(accent, 0.14)};
        color: {accent};
        border: 1px solid {_rgba(accent, 0.45)};
        border-radius: 10px;
        padding: 9px 20px;
        min-width: 118px;
        font-weight: 600;
    }}
    QPushButton#primary:hover {{
        background: {_rgba(accent, 0.26)};
        border-color: {accent};
        color: {accent};
    }}
    QPushButton#primary:checked {{
        background: {accent};
        color: #0d1117;
        border-color: {accent};
    }}

    /* --- Сегментированный контрол --- */
    QFrame#seg {{
        background: rgba(255,255,255,10);
        border: 1px solid rgba(255,255,255,22);
        border-radius: 11px;
    }}
    QPushButton#segbtn {{
        background: transparent;
        color: #8b949e;
        border: none;
        border-radius: 8px;
        padding: 6px 13px;
        font-weight: 500;
    }}
    QPushButton#segbtn:hover {{
        background: rgba(255,255,255,14);
        color: #ffffff;
    }}
    QPushButton#segbtn:checked {{
        background: {accent};
        color: #0d1117;
        font-weight: 700;
    }}

    /* --- Выпадающие списки --- */
    QComboBox {{
        background: rgba(255,255,255,10);
        color: #c9d1d9;
        border: 1px solid rgba(255,255,255,22);
        border-radius: 11px;
        padding: 8px 14px;
        min-width: 90px;
        font-family: 'Segoe UI';
        font-size: 12px;
        combobox-popup: 0;
    }}
    QComboBox:hover {{
        border-color: {accent};
        background: rgba(255,255,255,18);
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
        border-radius: 12px;
        padding: 4px;
        outline: 0;
        selection-background-color: transparent;
        selection-color: #c9d1d9;
    }}
    QComboBox QAbstractItemView::item {{
        min-height: 28px;
        padding: 4px 12px;
        border-radius: 8px;
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


# ---------------- Кастомные виджеты ----------------
class LogoMark(QtWidgets.QWidget):
    """Логотип: скруглённый квадрат с мини-осциллограммой."""

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
    """Переключатель-свитч с анимацией и подписью справа."""

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
        w = self.TW + 10 + self.fontMetrics().horizontalAdvance(self.text()) + 2
        return QtCore.QSize(w, 24)

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
    """Сегментированный контрол (замена QComboBox для коротких списков)."""

    currentIndexChanged = QtCore.Signal(int)

    def __init__(self, items, index=0, parent=None):
        super().__init__(parent)
        self.setObjectName("seg")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
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
        if self._btns:
            self._btns[max(0, min(index, len(self._btns) - 1))].setChecked(True)

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


class Swatch(QtWidgets.QPushButton):
    """Круглый свотч темы с кольцом выбора."""

    def __init__(self, color, parent=None):
        super().__init__(parent)
        self._c = QtGui.QColor(color)
        self._hover = False
        self.setCheckable(True)
        self.setFixedSize(26, 26)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

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
    """Иконка-кнопка для HUD соло-режима (иконки рисуются вручную)."""

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
    """Карточка статистики: мелкая подпись + крупное значение."""

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
        self.cap.setText(cap)
        self.val.setText(val_html)


class WavePlotWidget(pg.PlotWidget):
    """PlotWidget, который в прозрачном режиме перезаписывает (а не накладывает)
    фон каждый кадр. Иначе на прозрачном GL-буфере все кадры копятся друг на
    друге: фон темнеет, а осциллограмма превращается в сплошной «шлейф»."""

    transparent_bg = False

    def drawBackground(self, painter, rect):
        if self.transparent_bg:
            painter.save()
            painter.setCompositionMode(
                QtGui.QPainter.CompositionMode.CompositionMode_Source
            )
            # alpha=3/255: невидим, но окно остаётся «хватабельным» на Windows
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
        # Проход 1: раскладываем элементы по строкам.
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

        # Проход 2: ставим элементы, выравнивая по нижнему краю строки.
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


class MainWindow(QtWidgets.QWidget):
    CTRL_H = 36  # высота контролов в панели
    ROW_H = 54  # высота строки панели (подпись + контрол)
    _RESIZE_MARGIN = 8
    _MIN_W = 160
    _MIN_H = 40

    # --- размер окна соло-режима (ПКМ) ---
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

        # --- Solo ---
        self._solo = False
        self._card_shadow = None
        self._card_layout = None
        self._card_saved_margins = None
        self._plot_bg_main = None
        self._solo_pinned = False
        self._solo_transparent = True
        self._hud_on = False

        # --- Frameless drag/resize ---
        self._solo_drag_mode = None
        self._solo_resize_edges = (False, False, False, False)
        self._solo_origin_global = None
        self._solo_origin_geom = None

        self._kernel = np.ones(8, dtype=np.float32) / 8
        self._x = np.arange(self.n, dtype=np.float32)
        self.frames = 0
        self.t0 = time.perf_counter()

        # --- Тема ---
        self.theme_id = CONFIG.get("theme", DEFAULT_THEME)
        self.accent = THEME_HEX[self.theme_id]
        self._current_color = QtGui.QColor(self.accent)
        self._theme_anim = None

        self._capturer = Capturer()
        self._loading = True

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
        self._fit_pill()

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
        self.stats_timer.start(150)

        shortcut_cls = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut
        shortcut_cls(QtGui.QKeySequence("Space"), self, activated=self.freeze_btn.click)

        self._solo_esc = shortcut_cls(
            QtGui.QKeySequence("Escape"), self, activated=self._exit_solo
        )
        self._solo_esc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._solo_esc.setEnabled(False)

    # ---------- Layout helpers ----------
    def _fit_pill(self):
        """Подгоняет ширину «пилюли» устройства под свободное место в шапке."""
        if not hasattr(self, "pill"):
            return
        used = (
            52  # поля окна
            + self.logo.width()
            + 14
            + self.title_holder.sizeHint().width()
            + 14
            + 12
        )
        self.pill.setFixedWidth(max(220, min(480, self.width() - used)))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._fit_pill()

    # ---------- Устройства ----------
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
        idx = self.device_combo.currentIndex()
        if idx >= 0:
            did = self.device_combo.itemData(idx)
            CONFIG["device_id"] = did
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

    # ---------- Язык ----------
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
    def _freeze_text(on: bool) -> str:
        return "  " + (tr("resume") if on else tr("freeze"))

    @staticmethod
    def _freeze_icon(play: bool, color) -> QtGui.QIcon:
        """Иконка паузы / воспроизведения, нарисованная вручную (не зависит от шрифта)."""
        size, dpr = 18, 2
        pm = QtGui.QPixmap(size * dpr, size * dpr)
        pm.setDevicePixelRatio(dpr)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(color))
        if play:
            tri = QtGui.QPolygonF(
                [
                    QtCore.QPointF(5.5, 3.0),
                    QtCore.QPointF(15.0, 9.0),
                    QtCore.QPointF(5.5, 15.0),
                ]
            )
            p.drawPolygon(tri)
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
        self.trigger.setText(tr("stabilization"))
        self.trigger.setToolTip(tr("stabilization_tip"))
        self.glow.setText(tr("glow"))
        self.cap_window.setText(cap_text("window"))
        self.cap_gain.setText(cap_text("gain"))
        self.cap_stream.setText(cap_text("stream"))
        self.cap_theme.setText(cap_text("theme"))
        self.cap_lang.setText(cap_text("language"))
        self.cap_channels.setText(cap_text("channels"))
        for i, m in enumerate(CHANNEL_MODES):
            self.ch_combo.setItemText(i, tr(m))
        self.plot.setToolTip(tr("solo_tip"))

        # Размеры окна: пересобрать элементы с новым суффиксом.
        idx = self.n_combo.currentIndex()
        self.n_combo.blockSignals(True)
        self.n_combo.clear()
        for v in SAMPLE_OPTIONS:
            self.n_combo.addItem(f"{v} {tr('samples')}")
        self.n_combo.setCurrentIndex(idx)
        self.n_combo.blockSignals(False)

        # Tooltips тем.
        for tid, btn in self.theme_btns.items():
            btn.setToolTip(tr(f"theme_{tid}"))

        # Плейсхолдер «нет устройств».
        if not self.device_combo.isEnabled() and self.device_combo.count() == 1:
            self.device_combo.blockSignals(True)
            self.device_combo.setItemText(0, tr("no_devices"))
            self.device_combo.blockSignals(False)

        if self._solo:
            self.card.setWindowTitle(tr("solo_title"))

        self._apply_hud_texts()
        self._fit_pill()
        self.update_stats()

    # ---------- Solo ----------
    def _reapply_ranges(self):
        n = self.n
        self.plot.setXRange(-n * 0.012, n, padding=0)
        g = 1.0 / self.lim if self.lim else 1.0
        self.plot.setYRange(-1.06 / g, 1.06 / g, padding=0)

    def _enter_solo(self):
        if self._solo:
            return
        self._solo = True

        # Фиксированный размер окна соло-режима.
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
            self.card.setGraphicsEffect(None)

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
        self.card.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.card.setWindowTitle("")
        self.card.setStyleSheet("")
        self.card.unsetCursor()
        if self._card_shadow is not None:
            self.card.setGraphicsEffect(self._card_shadow)
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

    # ---------- Solo HUD ----------
    def _plot_bg(self):
        """Фон графика. В прозрачном соло — alpha=3/255: визуально невидим,
        но окно остаётся «хватабельным» (на Windows полностью прозрачные
        пиксели пропускают клики сквозь окно)."""
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
                background: #161b22;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 5px 10px;
                font-family: 'Segoe UI';
                font-size: 12px;
            }}
        """

    def _apply_solo_flags(self):
        flags = QtCore.Qt.WindowType.Window | QtCore.Qt.WindowType.FramelessWindowHint
        if self._solo_pinned:
            flags |= QtCore.Qt.WindowType.WindowStaysOnTopHint
        self.card.setWindowFlags(flags)

    def _apply_plot_render_mode(self):
        """Прозрачность на GL: viewport остаётся QOpenGLWidget (с альфа-каналом).
        В прозрачном соло буфер не сохраняется между кадрами (NoPartialUpdate)
        и каждый кадр перерисовывается целиком, чтобы не оставалось «шлейфов»
        и чёрного фона от старого содержимого FBO."""
        transparent = self._solo and self._solo_transparent
        self.plot.transparent_bg = transparent
        try:
            # кэш фона мешает перезаписи буфера — отключаем в прозрачном режиме
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
            QLabel {{
                color: #c9d1d9;
                font-family: 'Segoe UI';
                font-size: 12px;
                background: transparent;
            }}
            QLabel#stitle {{
                color: #f0f6fc;
                font-size: 13px;
                font-weight: 700;
            }}
            QLabel#sread {{
                color: #8b949e;
                font-family: 'Cascadia Mono', 'Consolas', monospace;
                font-size: 12px;
            }}
            QLabel#sdot {{
                background: {self.accent};
                border-radius: 4px;
            }}
        """)
        for b in (self.solo_freeze, self.solo_bg, self.solo_pin, self.solo_close):
            b.set_accent(self.accent)

    def _layout_hud(self):
        self._hud.setGeometry(0, 0, self.card.width(), self._hud.height())
        self._fit_hud()

    def _fit_hud(self):
        """Подгоняет содержимое HUD под ширину окна: сначала сокращается
        текст измерений, затем скрывается название, затем сами измерения.
        Кнопки видны всегда."""
        w = self.card.width()
        sp = self._hud_lay.spacing()
        m = self._hud_lay.contentsMargins()
        btns = 4 * 32 + 3 * sp
        avail = w - m.left() - m.right() - btns - sp  # место слева от кнопок
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
            # места нет даже на короткий текст: только точка и кнопки
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

    # ---------- Frameless move / resize ----------
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

    # ---------- Event filter ----------
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

    # ---------- Анимация темы ----------
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
        anim.finished.connect(lambda: setattr(self, "_current_color", target))
        anim.start()
        self._theme_anim = anim

    def _apply_theme_color(self, color: QtGui.QColor):
        c = QtGui.QColor(color)
        c.setAlpha(255)
        self.curve.setPen(pg.mkPen(c, width=1.7))
        c2 = _stereo_r_color(
            color.name() if hasattr(color, "name") else str(self.accent)
        )
        self.curve_r.setPen(pg.mkPen(c2, width=1.5))
        self.glow_renderer.set_color(color)
        self.glow_renderer_r.set_color(c2)
        self.logo.set_accent(c)
        self._update_freeze_icon()
        self.trigger.set_accent(c)
        self.glow.set_accent(c)
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

    def _apply_theme(self, theme_id: str, animate: bool = True):
        if theme_id in self.theme_btns:
            self.theme_btns[theme_id].setChecked(True)
        self.theme_id = theme_id
        target = THEME_HEX[theme_id]
        if animate:
            self._animate_theme(target)
        else:
            self.accent = target
            self._current_color = QtGui.QColor(target)
            self.setStyleSheet(make_qss(self.accent))
            self._apply_theme_color(self._current_color)
            self.plot.setBackground(self._plot_bg())
        if not getattr(self, "_loading", False):
            CONFIG["theme"] = theme_id
            _save_config(CONFIG)

    # ---------- UI ----------
    def _build_ui(self):
        no_focus = QtCore.Qt.FocusPolicy.NoFocus
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 22)
        root.setSpacing(16)
        self._root_layout = root

        # ----- Шапка: логотип · название · пилюля устройства -----
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
        self.device_lbl.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred
        )
        self.device_err = False
        pill_l.addWidget(self.dot, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        pill_l.addWidget(self.device_lbl, 1)
        header.addWidget(self.pill, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        root.addLayout(header)

        # ----- Карточка с графиком -----
        self.card = QtWidgets.QFrame()
        self.card.setObjectName("card")

        if not _USE_GL:
            shadow = QtWidgets.QGraphicsDropShadowEffect(self.card)
            shadow.setBlurRadius(24)
            shadow.setOffset(0, 6)
            shadow.setColor(QtGui.QColor(0, 0, 0, 80))
            self.card.setGraphicsEffect(shadow)
            self._card_shadow = shadow

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

        limit_pen = pg.mkPen(
            (255, 255, 255, 45), width=1, style=QtCore.Qt.PenStyle.DashLine
        )
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
        # R-glow чуть ниже по z, включается только в stereo + glow
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

        # ----- Панель управления -----
        self.panel = QtWidgets.QFrame()
        self.panel.setObjectName("panel")
        panel_l = QtWidgets.QVBoxLayout(self.panel)
        panel_l.setContentsMargins(18, 14, 18, 14)
        ctl = FlowLayout(hspacing=24, vspacing=14)
        panel_l.addLayout(ctl)

        # Заморозка (главное действие)
        self.freeze_btn = QtWidgets.QPushButton(self._freeze_text(False))
        self.freeze_btn.setObjectName("primary")
        self.freeze_btn.setCheckable(True)
        self.freeze_btn.setFocusPolicy(no_focus)
        self.freeze_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.freeze_btn.setFixedHeight(self.ROW_H)
        self.freeze_btn.setIconSize(QtCore.QSize(18, 18))
        self.freeze_btn.toggled.connect(self.on_freeze)
        ctl.addWidget(self.freeze_btn)  # FlowLayout выравнивает по низу строки

        # Свитчи: стабилизация + свечение
        self.trigger = ToggleSwitch(tr("stabilization"))
        self.trigger.setChecked(CONFIG.get("stabilization", True))
        self.trigger.setToolTip(tr("stabilization_tip"))
        self.trigger.toggled.connect(self._on_trigger_toggle)

        self.glow = ToggleSwitch(tr("glow"))
        self.glow.setChecked(CONFIG.get("glow", False))
        self.glow.toggled.connect(self._on_glow_toggle)

        holder = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(holder)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(6)
        sv.addWidget(self.trigger)
        sv.addWidget(self.glow)
        holder.setFixedHeight(self.ROW_H)
        ctl.addWidget(holder)

        # Окно
        n_idx = (
            SAMPLE_OPTIONS.index(CONFIG.get("n", 4096))
            if CONFIG.get("n") in SAMPLE_OPTIONS
            else SAMPLE_OPTIONS.index(4096)
        )
        self.n_combo, self.cap_window = self._combo(
            "window", [f"{v} {tr('samples')}" for v in SAMPLE_OPTIONS], n_idx, ctl
        )
        self.n_combo.currentIndexChanged.connect(
            lambda i: self.set_n(SAMPLE_OPTIONS[i])
        )

        # Усиление (сегменты)
        g_cfg = CONFIG.get("gain", 1)
        g_idx = GAIN_OPTIONS.index(g_cfg) if g_cfg in GAIN_OPTIONS else 0
        self.gain_combo = Segmented([(f"×{g}", g) for g in GAIN_OPTIONS], g_idx)
        self.gain_combo.setFixedHeight(self.CTRL_H)
        holder, self.cap_gain = self._caption_holder("gain", self.gain_combo)
        ctl.addWidget(holder)
        self.gain_combo.currentIndexChanged.connect(
            lambda i: self.set_gain(GAIN_OPTIONS[i])
        )

        # Каналы (моно / стерео — сегменты)
        ch_cfg = CONFIG.get("channels", "mono")
        ch_idx = CHANNEL_MODES.index(ch_cfg) if ch_cfg in CHANNEL_MODES else 0
        self.ch_combo = Segmented([(tr(m), m) for m in CHANNEL_MODES], ch_idx)
        self.ch_combo.setFixedHeight(self.CTRL_H)
        self.ch_combo.currentIndexChanged.connect(self._on_channels_change)
        holder, self.cap_channels = self._caption_holder("channels", self.ch_combo)
        ctl.addWidget(holder)

        # Поток
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.setMinimumWidth(210)
        self.device_combo.setFixedHeight(self.CTRL_H)
        self.device_combo.setFocusPolicy(no_focus)
        self.device_combo.currentIndexChanged.connect(self._on_device_change)
        self._style_combo_popup(self.device_combo)

        holder, self.cap_stream = self._caption_holder("stream", self.device_combo)
        ctl.addWidget(holder)

        # Язык
        self.lang_combo = QtWidgets.QComboBox()
        self.lang_combo.setMinimumWidth(120)
        self.lang_combo.setFixedHeight(self.CTRL_H)
        self.lang_combo.setFocusPolicy(no_focus)
        for label, code in LANGUAGES:
            self.lang_combo.addItem(label, code)
        # Выставить текущий.
        for i in range(self.lang_combo.count()):
            if self.lang_combo.itemData(i) == LANG["current"]:
                self.lang_combo.setCurrentIndex(i)
                break
        self.lang_combo.currentIndexChanged.connect(self._on_language_change)
        self._style_combo_popup(self.lang_combo)

        holder, self.cap_lang = self._caption_holder("language", self.lang_combo)
        ctl.addWidget(holder)

        # Тема
        self._theme_picker(ctl)
        root.addWidget(self.panel)

        # ----- Карточки статистики -----
        stats = FlowLayout(hspacing=12, vspacing=10)
        self.peak_tile = StatTile(290)
        self.rms_tile = StatTile(160)
        self.info_tile = StatTile(240)
        self.fps_tile = StatTile(110)
        self.bknd_tile = StatTile(140)
        for t in (
            self.peak_tile,
            self.rms_tile,
            self.info_tile,
            self.fps_tile,
            self.bknd_tile,
        ):
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
        """Поле: мелкая подпись сверху, контрол снизу."""
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

    def _theme_picker(self, layout):
        row_w = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(row_w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        row_w.setFixedHeight(self.CTRL_H)

        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        self.theme_btns = {}
        for tid, color in THEMES:
            b = Swatch(color)
            b.setToolTip(tr(f"theme_{tid}"))
            b.clicked.connect(lambda _=False, t=tid: self._apply_theme(t))
            group.addButton(b)
            row.addWidget(b)
            self.theme_btns[tid] = b

        holder, self.cap_theme = self._caption_holder("theme", row_w)
        layout.addWidget(holder)

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
        if not getattr(self, "_loading", False):
            CONFIG["channels"] = mode
            _save_config(CONFIG)

    def update_frame(self):
        self.frames += 1
        if self.frozen:
            if self._solo:
                # прозрачное GL-окно: держим буфер свежим даже на паузе
                self.plot.viewport().update()
            return

        n = self.n
        with lock:
            data_l = buf_l[-(n + n) :].copy()
            data_r = buf_r[-(n + n) :].copy()
            total = state.get("total_samples", 0)

        shift = total - self._last_total
        self._last_total = total

        # Триггер по сумме L+R (стабильнее для стерео)
        data_trig = (data_l + data_r) * 0.5

        x = self._x
        start = n

        if self.trigger.isChecked():
            found = self.find_trigger(data_trig, shift)
            if found is not None:
                start, x_off = found
                start = max(0, min(start, len(data_trig) - n))
                x = self._x + x_off

        if self.channels == "stereo":
            view_l = data_l[start : start + n]
            view_r = data_r[start : start + n]
            self.last = view_l
            self.last_r = view_r
            # Overlay: оба канала на одной шкале, разные цвета
            shown_l = np.clip(view_l, -self.lim, self.lim)
            shown_r = np.clip(view_r, -self.lim, self.lim)
            self.curve.setData(x, shown_l, skipFiniteCheck=True)
            self.curve_r.setData(x, shown_r, skipFiniteCheck=True)
            vb = self.plot.getViewBox()
            self.glow_renderer.update(x, shown_l, vb)
            self.glow_renderer_r.update(x, shown_r, vb)
        else:
            view = data_trig[start : start + n]
            self.last = view
            shown = np.clip(view, -self.lim, self.lim)
            self.curve.setData(x, shown, skipFiniteCheck=True)
            if self.curve_r.isVisible():
                self.curve_r.clear()
            self.glow_renderer.update(x, shown, self.plot.getViewBox())

    def find_trigger(self, data, shift):
        n = self.n
        L = len(data)
        half = n // 2

        s = np.convolve(data, self._kernel, mode="same")
        s -= s.mean()

        amp = float(np.max(np.abs(s[half : L - half])))
        if amp < 1e-4:
            return None
        h = 0.15 * amp

        st = np.zeros(L, dtype=np.int8)
        st[s > h] = 1
        st[s < -h] = -1

        nz_idx = np.flatnonzero(st)
        if nz_idx.size == 0:
            return None

        pos_in_nz = (
            np.searchsorted(nz_idx, np.arange(L, dtype=np.int32), side="right") - 1
        )
        filled = np.zeros(L, dtype=np.int8)
        valid = pos_in_nz >= 0
        filled[valid] = st[nz_idx[pos_in_nz[valid]]]
        st = filled

        edges = np.flatnonzero((st[:-1] == -1) & (st[1:] == 1)) + 1
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

        recent = cand[cand >= L - half - TRIGGER_MAX_AGE]
        if recent.size:
            cand = recent
        cand = cand[-24:]

        predicted = None
        if self._crossing_smooth is not None and shift >= 0:
            predicted = self._crossing_smooth - shift

        T = min(192, n // 2)
        hT = T // 2
        best = len(cand) - 1

        if self.prev_tpl is not None and len(self.prev_tpl) == T and len(cand) > 1:
            idxs = cand[:, None] + np.arange(-hT, hT, dtype=np.int32)[None, :]
            tpl = data[idxs].astype(np.float32, copy=False)
            tpl = tpl - tpl.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(tpl, axis=1) + 1e-9
            score = (tpl @ self.prev_tpl) / norms
            score = score.astype(np.float32, copy=False)

            if predicted is not None:
                dist = np.abs(cand.astype(np.float32) - predicted)
                score -= 0.005 * dist

            score += 0.012 * np.linspace(0.0, 1.0, len(cand), dtype=np.float32)

            j = int(np.argmax(score))
            if score[j] >= 0.28:
                best = j

        c = int(cand[best])
        ref = data[c - hT : c + hT].astype(np.float32, copy=False)
        ref = ref - ref.mean()
        self.prev_tpl = ref / (np.linalg.norm(ref) + 1e-9)

        a, b = float(s[c - 1]), float(s[c])
        frac = a / (a - b) if a != b else 0.0
        frac = min(max(frac, 0.0), 1.0)

        crossing = (c - 1) + frac

        if self._crossing_smooth is None or shift <= 0:
            self._crossing_smooth = crossing
        else:
            expected = self._crossing_smooth - shift
            delta = crossing - expected
            if abs(delta) > n * 0.25:
                self._crossing_smooth = crossing
            else:
                self._crossing_smooth = crossing - (1.0 - self._trig_alpha) * delta

        cs = self._crossing_smooth
        start = int(cs) - half
        x_off = -(cs - int(cs))
        return start, x_off

    @staticmethod
    def _db(x):
        return "−∞" if x < 1e-6 else f"{20 * np.log10(x):+.1f}".replace("-", "−")

    def update_stats(self):
        dim = '<span style="color:#6e7681">'
        seg = self.last
        if self.channels == "stereo" and len(self.last_r):
            if len(seg):
                peak_l = float(np.max(np.abs(seg)))
                peak_r = float(np.max(np.abs(self.last_r)))
                peak = max(peak_l, peak_r)
                rms = float(
                    np.sqrt(
                        0.5 * (np.mean(seg * seg) + np.mean(self.last_r * self.last_r))
                    )
                )
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
                f'<span style="color:{col_r}">R {self._db(peak_r)}{mark_r}</span>',
            )
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
                f'<span style="color:{peak_color}">'
                f"{self._db(peak)} dBFS{clip}</span>",
            )
            self.rms_tile.set(tr("rms"), f"{self._db(rms)} dBFS")
        if self._solo:
            self._solo_vals = (self._db(peak), self._db(rms))
            self._fit_hud()
        self.info_tile.set(
            tr("window_label"),
            f'{self.n} {tr("samples")} {dim}·</span> '
            f'{self.n / SAMPLERATE * 1000:.1f} {tr("ms")}',
        )

        now = time.perf_counter()
        fps = self.frames / max(now - self.t0, 1e-3)
        self.frames = 0
        self.t0 = now
        self.fps_tile.set(
            tr("fps_label"), tr("paused") if self.frozen else f"{fps:.0f}"
        )

        be_name = tr(_BACKEND_NAME_KEYS.get(_BACKEND_ID, "backend_raster"))
        self.bknd_tile.set(tr("backend_label"), be_name)
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

        self.pill.setToolTip(full)
        self.device_lbl.setText(
            self.device_lbl.fontMetrics().elidedText(
                full,
                QtCore.Qt.TextElideMode.ElideRight,
                max(self.device_lbl.width() - 4, 20),
            )
        )

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


def _app_dir() -> str:
    """Папка со скриптом или с .exe (PyInstaller)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _icon_path():
    for name in ("icon.ico", "icon.png"):
        p = os.path.join(_app_dir(), name)
        if os.path.isfile(p):
            return p
    return None


def _apply_taskbar_icon(win, icon: QtGui.QIcon):
    """Оверлей иконки в панели задач Windows (если доступен QtWinExtras)."""
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
        win._taskbar_button = btn  # чтобы не собрал GC
    except Exception:
        pass


def _setup_gl_format():
    """Альфа-буфер 8 бит — нужен для прозрачного GL-окна (OpenGL / Vulkan-режим)."""
    if not _USE_GL:
        return
    try:
        fmt = QtGui.QSurfaceFormat(QtGui.QSurfaceFormat.defaultFormat())
        fmt.setAlphaBufferSize(8)
        QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    except Exception:
        pass


def main():
    _setup_gl_format()
    app = pg.mkQApp("PC Waveform")
    icon_file = _icon_path()
    icon = QtGui.QIcon(icon_file) if icon_file else QtGui.QIcon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    win = MainWindow()
    if not icon.isNull():
        win.setWindowIcon(icon)
    win.show()
    dark_titlebar(win)
    if not icon.isNull():
        # windowHandle появляется после show()
        QtCore.QTimer.singleShot(0, lambda: _apply_taskbar_icon(win, icon))
    pg.exec()


if __name__ == "__main__":
    main()