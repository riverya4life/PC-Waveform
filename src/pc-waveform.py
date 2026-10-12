"""
PC Waveform — осциллограмма звука с ПК (WASAPI loopback) by riverya4life
Optimized: device selection, fast trigger, layered glow, smooth stabilization,
solo mode (RMB on plot, solid dark background, axes hidden),
i18n: ru / uk / en (default ru).
Settings: INI in %APPDATA%/PC Waveform/settings.ini
(lang, theme, window, gain, stabilization, glow, device).
Backend auto: OpenGL, если доступен, иначе растр.
"""

import configparser
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
MAX_N = 8192
TRIGGER_MAX_AGE = 1536
SAMPLE_OPTIONS = [1024, 2048, 4096, 8192]
GAIN_OPTIONS = [1, 2, 4, 8]
TICKS = {1024: 250, 2048: 500, 4096: 500, 8192: 1000}

# AppData\PC Waveform\settings.ini  (Windows);  ~/.config/PC Waveform/  (прочее)
def _config_dir() -> str:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
            os.path.expanduser("~"), ".config")
    return os.path.join(base, "PC Waveform")


CONFIG_PATH = os.path.join(_config_dir(), "settings.ini")

DEFAULT_CONFIG = {
    "lang": "ru",
    "theme": "mint",
    "n": 4096,
    "gain": 1,
    "stabilization": True,
    "glow": False,
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
            "stabilization": "true" if _bool(cfg.get("stabilization", True)) else "false",
            "glow": "true" if _bool(cfg.get("glow", False)) else "false",
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
    ("Русский",     "ru"),
    ("Українська",  "uk"),
    ("English",     "en"),
]


LANG = {"current": CONFIG["lang"]}

TRANSLATIONS = {
    "ru": {
        "brand":              "PC Waveform",
        "main_title":         "PC Waveform — осциллограмма системного звука (github: riverya4life)",
        "solo_title":         "PC Waveform — соло (ПКМ или Esc — вернуться)",
        "subtitle":           "осциллограмма системного звука",
        "freeze":             "Заморозить",
        "resume":             "Продолжить",
        "stabilization":      "Стабилизация",
        "stabilization_tip":  "Триггер по фронту с гистерезисом + захват фазы",
        "glow":               "Свечение",
        "window":             "Окно",
        "gain":               "Усиление",
        "stream":             "Поток",
        "theme":              "Тема",
        "language":           "Язык",
        "samples":            "смп",
        "ms":                 "мс",
        "khz":                "кГц",
        "peak":               "PEAK",
        "rms":                "RMS",
        "window_label":       "ОКНО",
        "fps_label":          "FPS",
        "backend_label":      "БЭКЕНД",
        "backend_opengl":     "OpenGL",
        "backend_vulkan":     "Vulkan",
        "backend_raster":     "Raster",
        "backend_tip":        "Графический движок отрисовки с поддержкой OpenGL и Vulkan.\n"
                              "Автовыбор: OpenGL (если доступен)\n"
                              "\n"
                              "• OpenGL — OpenGL-рендер pyqtgraph, максимальный FPS\n"
                              "• Vulkan — Vulkan-контекст Qt + GL-рендер графика\n"
                              "• Raster — без GL: QPainter (медленнее, но совместимо)\n"
                              "\n"
                              "Принудительно (для разработчиков): "
                              "PCWAVEFORM_BACKEND=opengl|vulkan|raster",
        "clip":               "CLIP",
        "paused":             "пауза",
        "error_prefix":       "Ошибка:",
        "searching":          "поиск устройства…",
        "no_devices":         "(нет устройств)",
        "solo_tip":           "ПКМ — режим «соло»",
        "theme_mint":         "Мята",
        "theme_cyan":         "Циан",
        "theme_azure":        "Лазурь",
        "theme_indigo":       "Индиго",
        "theme_violet":       "Фиолет",
        "theme_pink":         "Розовый",
        "theme_raspberry":    "Малина",
        "theme_coral":        "Коралл",
        "theme_amber":        "Янтарь",
        "theme_lime":         "Лайм",
        "theme_turquoise":    "Бирюза",
        "theme_white":        "Белый",
    },
    "uk": {
        "brand":              "PC Waveform",
        "main_title":         "PC Waveform — осцилограма системного звуку (github: riverya4life)",
        "solo_title":         "PC Waveform — соло (ПКМ або Esc — повернутися)",
        "subtitle":           "осцилограма системного звуку",
        "freeze":             "Заморозити",
        "resume":             "Продовжити",
        "stabilization":      "Стабілізація",
        "stabilization_tip":  "Тригер по фронту з гістерезисом + захоплення фази",
        "glow":               "Світіння",
        "window":             "Вікно",
        "gain":               "Підсилення",
        "stream":             "Потік",
        "theme":              "Тема",
        "language":           "Мова",
        "samples":            "семп",
        "ms":                 "мс",
        "khz":                "кГц",
        "peak":               "PEAK",
        "rms":                "RMS",
        "window_label":       "ВІКНО",
        "fps_label":          "FPS",
        "backend_label":      "БЕКЕНД",
        "backend_opengl":     "OpenGL",
        "backend_vulkan":     "Vulkan",
        "backend_raster":     "Raster",
        "backend_tip":        "Графічний рушій відтворення з підтримкою OpenGL і Vulkan.\n"
                              "Автовибір: OpenGL (якщо доступний)\n"
                              "\n"
                              "• OpenGL — OpenGL-рендер pyqtgraph, максимальний FPS\n"
                              "• Vulkan — Vulkan-контекст Qt + GL-рендер графіка\n"
                              "• Raster — без GL: QPainter (повільніше, але сумісно)\n"
                              "\n"
                              "Примусово (для розробників): "
                              "PCWAVEFORM_BACKEND=opengl|vulkan|raster",
        "clip":               "CLIP",
        "paused":             "пауза",
        "error_prefix":       "Помилка:",
        "searching":          "пошук пристрою…",
        "no_devices":         "(немає пристроїв)",
        "solo_tip":           "ПКМ — режим «соло»",
        "theme_mint":         "М'ята",
        "theme_cyan":         "Ціан",
        "theme_azure":        "Лазур",
        "theme_indigo":       "Індиго",
        "theme_violet":       "Фіолет",
        "theme_pink":         "Рожевий",
        "theme_raspberry":    "Малина",
        "theme_coral":        "Корал",
        "theme_amber":        "Бурштин",
        "theme_lime":         "Лайм",
        "theme_turquoise":    "Бірюза",
        "theme_white":        "Білий",
    },
    "en": {
        "brand":              "PC Waveform",
        "main_title":         "PC Waveform — system audio waveform (github: riverya4life)",
        "solo_title":         "PC Waveform — solo (RMB or Esc — exit)",
        "subtitle":           "system audio waveform",
        "freeze":             "Freeze",
        "resume":             "Resume",
        "stabilization":      "Stabilization",
        "stabilization_tip":  "Edge trigger with hysteresis + phase lock",
        "glow":               "Glow",
        "window":             "Window",
        "gain":               "Gain",
        "stream":             "Stream",
        "theme":              "Theme",
        "language":           "Language",
        "samples":            "smp",
        "ms":                 "ms",
        "khz":                "kHz",
        "peak":               "PEAK",
        "rms":                "RMS",
        "window_label":       "WINDOW",
        "fps_label":          "FPS",
        "backend_label":      "BACKEND",
        "backend_opengl":     "OpenGL",
        "backend_vulkan":     "Vulkan",
        "backend_raster":     "Raster",
        "backend_tip":        "Graphics rendering backend with OpenGL and Vulkan support.\n"
                              "Auto-select: OpenGL (if available)\n"
                              "\n"
                              "• OpenGL — pyqtgraph OpenGL render, max FPS\n"
                              "• Vulkan — Qt Vulkan context + GL plot render\n"
                              "• Raster — no GL: QPainter (slower, more compatible)\n"
                              "\n"
                              "Force (for developers): "
                              "PCWAVEFORM_BACKEND=opengl|vulkan|raster",
        "clip":               "CLIP",
        "paused":             "paused",
        "error_prefix":       "Error:",
        "searching":          "searching device…",
        "no_devices":         "(no devices)",
        "solo_tip":           "RMB — solo mode",
        "theme_mint":         "Mint",
        "theme_cyan":         "Cyan",
        "theme_azure":        "Azure",
        "theme_indigo":       "Indigo",
        "theme_violet":       "Violet",
        "theme_pink":         "Pink",
        "theme_raspberry":    "Raspberry",
        "theme_coral":        "Coral",
        "theme_amber":        "Amber",
        "theme_lime":         "Lime",
        "theme_turquoise":    "Turquoise",
        "theme_white":        "White",
    },
}


def tr(key: str) -> str:
    d = TRANSLATIONS.get(LANG["current"], TRANSLATIONS["ru"])
    return d.get(key, key)


# ---------------- Темы ----------------
# Ключ — стабильный id, hex — цвет. Отображаемое имя берётся из tr().
THEMES = [
    ("mint",       "#00ff8c"),
    ("cyan",       "#19c8ff"),
    ("azure",      "#4d8dff"),
    ("indigo",     "#7c6cff"),
    ("violet",     "#b26bff"),
    ("pink",       "#ff50a0"),
    ("raspberry",  "#ff2e63"),
    ("coral",      "#ff6b4a"),
    ("amber",      "#ffb000"),
    ("lime",       "#b6ff2e"),
    ("turquoise",  "#2ee6d6"),
    ("white",      "#ebebeb"),
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
    "vulkan" if (_USE_GL and _FORCED == "vulkan")
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

# --- Общие буферы ---
buf = np.zeros(MAX_N * 2, dtype=np.float32)
lock = threading.Lock()
state = {"device": None, "error": None, "total_samples": 0}


# ---------------- Многослойное свечение ----------------
class GlowRenderer:
    SPECS = [
        (-3, 18),
        (-2, 34),
        (-1, 62),
        ( 1, 62),
        ( 2, 34),
        ( 3, 18),
    ]

    def __init__(self, plot_item):
        self._plot = plot_item
        self._color = QtGui.QColor(THEME_HEX[DEFAULT_THEME])
        self._enabled = False
        self._layers = []

        for dy_px, alpha in self.SPECS:
            item = pg.PlotDataItem(pen=None)
            item.setZValue(-1)
            item.setDownsampling(auto=True, method='mean')
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
            target=self._run, args=(device_id, self._stop), daemon=True)
        self._thread.start()

    @staticmethod
    def _run(device_id, stop):
        try:
            mic = sc.get_microphone(id=str(device_id), include_loopback=True)
            state["device"] = str(mic.name)
            state["error"] = None
            with mic.recorder(samplerate=SAMPLERATE, channels=2,
                              blocksize=CHUNK) as rec:
                while not stop.is_set():
                    data = rec.record(numframes=CHUNK)
                    if stop.is_set():
                        break
                    mono = data.mean(axis=1).astype(np.float32, copy=False)
                    n = len(mono)
                    if n == 0:
                        continue
                    with lock:
                        buf[:-n] = buf[n:]
                        buf[-n:] = mono
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
    _RESIZE_MARGIN = 8
    _MIN_W = 160
    _MIN_H = 40

    def __init__(self):
        super().__init__()
        self.setObjectName("root")
        self.setWindowTitle(tr("main_title"))
        self.resize(980, 580)
        self.setMinimumSize(500, 500)

        self.n = CONFIG.get("n", 4096)
        self.frozen = False
        self.last = np.zeros(self.n, dtype=np.float32)
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

        self._populate_devices()
        self._loading = False

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
        shortcut_cls(QtGui.QKeySequence("Space"), self,
                     activated=self.freeze_btn.click)

        self._solo_esc = shortcut_cls(
            QtGui.QKeySequence("Escape"), self,
            activated=self._exit_solo)
        self._solo_esc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._solo_esc.setEnabled(False)

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

    def _retranslate_ui(self):
        self.setWindowTitle(tr("main_title"))
        self.sub_lbl.setText(tr("subtitle"))
        self.freeze_btn.setText(tr("resume") if self.frozen else tr("freeze"))
        self.trigger.setText(tr("stabilization"))
        self.trigger.setToolTip(tr("stabilization_tip"))
        self.glow.setText(tr("glow"))
        self.cap_window.setText(tr("window"))
        self.cap_gain.setText(tr("gain"))
        self.cap_stream.setText(tr("stream"))
        self.cap_theme.setText(tr("theme"))
        self.cap_lang.setText(tr("language"))
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
        if (not self.device_combo.isEnabled()
                and self.device_combo.count() == 1):
            self.device_combo.blockSignals(True)
            self.device_combo.setItemText(0, tr("no_devices"))
            self.device_combo.blockSignals(False)

        if self._solo:
            self.card.setWindowTitle(tr("solo_title"))

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

        size = self.card.size()
        if size.width() < 500 or size.height() < 300:
            size = QtCore.QSize(1000, 560)

        if self._card_layout is not None:
            self._card_saved_margins = self._card_layout.contentsMargins()
            self._card_layout.setContentsMargins(0, 0, 0, 0)

        pi = self.plot.getPlotItem()
        pi.hideAxis('bottom')
        pi.hideAxis('left')

        try:
            pi.getViewBox().disableAutoRange()
        except Exception:
            pass

        self._plot_bg_main = pg.mkColor(_mix_dark(self.accent, 0.045))
        self.plot.setBackground(self._plot_bg_main)
        self.plot.setStyleSheet("background: transparent; border: none;")

        self.card.hide()
        self._root_layout.removeWidget(self.card)
        self.card.setParent(None)
        self.card.setWindowFlags(
            QtCore.Qt.WindowType.Window
            | QtCore.Qt.WindowType.FramelessWindowHint
        )
        self.card.setWindowTitle(tr("solo_title"))
        self.card.setStyleSheet(f"""
            QFrame#card {{
                background: {_mix_dark(self.accent, 0.045)};
                border: 1px solid {self.accent};
                border-radius: 0;
            }}
        """)
        if self._card_shadow is not None:
            self.card.setGraphicsEffect(None)

        self.card.setMouseTracking(True)
        self.plot.setMouseTracking(True)
        try:
            self.plot.viewport().setMouseTracking(True)
        except Exception:
            pass

        self.card.resize(size)
        self.card.show()
        self.card.raise_()
        self.card.activateWindow()
        self.hide()

        QtCore.QTimer.singleShot(0, self._reapply_ranges)
        self._solo_esc.setEnabled(True)

    def _exit_solo(self):
        if not self._solo:
            return
        self._solo = False
        self._solo_drag_mode = None

        self.card.hide()
        self.card.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.card.setWindowTitle("")
        self.card.setStyleSheet("")
        self.card.unsetCursor()
        if self._card_shadow is not None:
            self.card.setGraphicsEffect(self._card_shadow)
        if (self._card_layout is not None
                and self._card_saved_margins is not None):
            self._card_layout.setContentsMargins(self._card_saved_margins)

        pi = self.plot.getPlotItem()
        pi.showAxis('bottom')
        pi.showAxis('left')
        self.left.setWidth(48)

        if self._plot_bg_main is not None:
            self.plot.setBackground(self._plot_bg_main)

        self._root_layout.insertWidget(1, self.card)

        self.show()
        self.card.show()
        self.raise_()
        self.activateWindow()

        QtCore.QTimer.singleShot(0, self._reapply_ranges)
        self._solo_esc.setEnabled(False)

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

        is_plot = (obj is self.plot
                   or (hasattr(self.plot, "viewport")
                       and obj is self.plot.viewport()))
        is_card = obj is self.card

        if not (is_plot or is_card):
            return super().eventFilter(obj, ev)

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
        if (self._theme_anim is not None
                and self._theme_anim.state()
                == QtCore.QAbstractAnimation.State.Running):
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
        self.curve.setPen(pg.mkPen(c, width=1.6))
        self.glow_renderer.set_color(color)
        if self._solo:
            self.card.setStyleSheet(f"""
                QFrame#card {{
                    background: {_mix_dark(self.accent, 0.045)};
                    border: 1px solid {self.accent};
                    border-radius: 0;
                }}
            """)

    def _on_theme_color(self, color: QtGui.QColor):
        if not isinstance(color, QtGui.QColor):
            return
        self.accent = color.name()
        self.setStyleSheet(make_qss(self.accent))
        self._apply_theme_color(color)
        self.plot.setBackground(pg.mkColor(_mix_dark(self.accent, 0.045)))

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
            self.plot.setBackground(pg.mkColor(_mix_dark(self.accent, 0.045)))
        if not getattr(self, "_loading", False):
            CONFIG["theme"] = theme_id
            _save_config(CONFIG)

    # ---------- UI ----------
    def _build_ui(self):
        no_focus = QtCore.Qt.FocusPolicy.NoFocus
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)
        self._root_layout = root

        header = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        title_box.setSpacing(2)
        title = QtWidgets.QLabel(tr("brand"))
        title.setObjectName("title")
        self.sub_lbl = QtWidgets.QLabel(tr("subtitle"))
        self.sub_lbl.setObjectName("sub")
        title_box.addWidget(title)
        title_box.addWidget(self.sub_lbl)
        header.addLayout(title_box)

        self.device_lbl = QtWidgets.QLabel("")
        self.device_lbl.setObjectName("device")
        self.device_lbl.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.device_lbl.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred)
        self.device_err = False
        header.addWidget(self.device_lbl, 1)
        root.addLayout(header)

        self.card = QtWidgets.QFrame()
        self.card.setObjectName("card")

        if not _USE_GL:
            shadow = QtWidgets.QGraphicsDropShadowEffect(self.card)
            shadow.setBlurRadius(18)
            shadow.setOffset(0, 5)
            shadow.setColor(QtGui.QColor(0, 0, 0, 70))
            self.card.setGraphicsEffect(shadow)
            self._card_shadow = shadow

        card_l = QtWidgets.QVBoxLayout(self.card)
        card_l.setContentsMargins(10, 12, 14, 8)
        self._card_layout = card_l

        self.bottom = SpacedAxis("bottom")
        self.left = SpacedAxis("left")
        self.plot = pg.PlotWidget(
            axisItems={"bottom": self.bottom, "left": self.left})

        self.plot.setBackground(pg.mkColor(_mix_dark(self.accent, 0.045)))
        self.plot.setStyleSheet("background: transparent;")
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setToolTip(tr("solo_tip"))

        pi = self.plot.getPlotItem()
        pi.setMenuEnabled(False)
        pi.hideButtons()
        pi.showGrid(x=True, y=True, alpha=0.09)
        try:
            pi.getViewBox().disableAutoRange()
        except Exception:
            pass

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

        self.curve = pg.PlotDataItem(pen=None)
        self.curve.setZValue(0)
        self.curve.setDownsampling(auto=True, method='mean')
        if not _USE_GL:
            self.curve.setClipToView(True)
        self.plot.addItem(self.curve)

        self.glow_renderer = GlowRenderer(pi)

        card_l.addWidget(self.plot)
        root.addWidget(self.card, 1)

        self.plot.installEventFilter(self)
        try:
            self.plot.viewport().installEventFilter(self)
        except Exception:
            pass
        self.card.installEventFilter(self)

        ctl = FlowLayout(hspacing=18, vspacing=12)

        self.freeze_btn = QtWidgets.QPushButton(tr("freeze"))
        self.freeze_btn.setCheckable(True)
        self.freeze_btn.setFocusPolicy(no_focus)
        self.freeze_btn.toggled.connect(self.on_freeze)
        ctl.addWidget(self.freeze_btn)

        self.trigger = QtWidgets.QCheckBox(tr("stabilization"))
        self.trigger.setChecked(CONFIG.get("stabilization", True))
        self.trigger.setFocusPolicy(no_focus)
        self.trigger.setToolTip(tr("stabilization_tip"))
        self.trigger.toggled.connect(self._on_trigger_toggle)
        ctl.addWidget(self.trigger)

        self.glow = QtWidgets.QCheckBox(tr("glow"))
        self.glow.setChecked(CONFIG.get("glow", False))
        self.glow.setFocusPolicy(no_focus)
        self.glow.toggled.connect(self._on_glow_toggle)
        ctl.addWidget(self.glow)

        # Окно
        n_idx = SAMPLE_OPTIONS.index(CONFIG.get("n", 4096)) if CONFIG.get("n") in SAMPLE_OPTIONS else SAMPLE_OPTIONS.index(4096)
        self.n_combo, self.cap_window = self._combo(
            "window",
            [f"{v} {tr('samples')}" for v in SAMPLE_OPTIONS],
            n_idx, ctl)
        self.n_combo.currentIndexChanged.connect(
            lambda i: self.set_n(SAMPLE_OPTIONS[i]))

        # Усиление
        g_cfg = CONFIG.get("gain", 1)
        g_idx = GAIN_OPTIONS.index(g_cfg) if g_cfg in GAIN_OPTIONS else 0
        self.gain_combo, self.cap_gain = self._combo(
            "gain",
            [f"×{g}" for g in GAIN_OPTIONS], g_idx, ctl)
        self.gain_combo.currentIndexChanged.connect(
            lambda i: self.set_gain(GAIN_OPTIONS[i]))

        # Поток
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.setMinimumWidth(200)
        self.device_combo.setFocusPolicy(no_focus)
        self.device_combo.currentIndexChanged.connect(self._on_device_change)
        self._style_combo_popup(self.device_combo)

        holder, self.cap_stream = self._caption_holder("stream",
                                                        self.device_combo)
        ctl.addWidget(holder)

        # Язык
        self.lang_combo = QtWidgets.QComboBox()
        self.lang_combo.setMinimumWidth(120)
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

        holder, self.cap_lang = self._caption_holder("language",
                                                     self.lang_combo)
        ctl.addWidget(holder)

        # Тема
        self._theme_picker(ctl)
        root.addLayout(ctl)

        stats = FlowLayout(hspacing=12, vspacing=10)
        self.peak_lbl = QtWidgets.QLabel()
        self.rms_lbl = QtWidgets.QLabel()
        self.info_lbl = QtWidgets.QLabel()
        self.fps_lbl = QtWidgets.QLabel()
        self.bknd_lbl = QtWidgets.QLabel()

        for lbl, w in ((self.peak_lbl, 200), (self.rms_lbl, 155),
                       (self.info_lbl, 235), (self.fps_lbl, 105),
                       (self.bknd_lbl, 130)):
            lbl.setObjectName("stat")
            lbl.setMinimumWidth(w)
            stats.addWidget(lbl)

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
            | QtCore.Qt.WindowType.NoDropShadowWindowHint)
        popup.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

    def _caption_holder(self, caption_key, widget):
        cap = QtWidgets.QLabel(tr(caption_key))
        cap.setObjectName("cap")
        holder = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(9)
        box.addWidget(cap)
        box.addWidget(widget)
        return holder, cap

    def _theme_picker(self, layout):
        holder = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(7)
        self.cap_theme = QtWidgets.QLabel(tr("theme"))
        self.cap_theme.setObjectName("cap")
        row.addWidget(self.cap_theme)
        row.addSpacing(4)

        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        self.theme_btns = {}
        for tid, color in THEMES:
            b = QtWidgets.QPushButton()
            b.setCheckable(True)
            b.setFixedSize(24, 24)
            b.setToolTip(tr(f"theme_{tid}"))
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            b.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            b.setStyleSheet(swatch_qss(color))
            b.clicked.connect(lambda _=False, t=tid: self._apply_theme(t))
            group.addButton(b)
            row.addWidget(b)
            self.theme_btns[tid] = b
        layout.addWidget(holder)

    def _combo(self, caption_key, items, index, layout):
        cap = QtWidgets.QLabel(tr(caption_key))
        cap.setObjectName("cap")
        combo = QtWidgets.QComboBox()
        combo.addItems(items)
        self._style_combo_popup(combo)
        combo.setCurrentIndex(index)
        combo.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        holder = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(9)
        box.addWidget(cap)
        box.addWidget(combo)
        layout.addWidget(holder)
        return combo, cap

    def set_n(self, n):
        self.n = n
        self._x = np.arange(n, dtype=np.float32)
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
        self.freeze_btn.setText(tr("resume") if on else tr("freeze"))

    def _on_trigger_toggle(self, on):
        self.prev_tpl = None
        self._crossing_smooth = None
        CONFIG["stabilization"] = bool(on)
        _save_config(CONFIG)

    def _on_glow_toggle(self, on: bool):
        self.glow_renderer.set_enabled(on)
        CONFIG["glow"] = bool(on)
        _save_config(CONFIG)

    def update_frame(self):
        self.frames += 1
        if self.frozen:
            return

        n = self.n
        with lock:
            data = buf[-(n + n):].copy()
            total = state.get("total_samples", 0)

        shift = total - self._last_total
        self._last_total = total

        x = self._x
        start = n

        if self.trigger.isChecked():
            found = self.find_trigger(data, shift)
            if found is not None:
                start, x_off = found
                start = max(0, min(start, len(data) - n))
                x = self._x + x_off

        view = data[start:start + n]
        self.last = view
        shown = np.clip(view, -self.lim, self.lim)

        self.curve.setData(x, shown, skipFiniteCheck=True)
        self.glow_renderer.update(x, shown, self.plot.getViewBox())

    def find_trigger(self, data, shift):
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

        nz_idx = np.flatnonzero(st)
        if nz_idx.size == 0:
            return None

        pos_in_nz = np.searchsorted(
            nz_idx, np.arange(L, dtype=np.int32), side="right") - 1
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

        if (self.prev_tpl is not None
                and len(self.prev_tpl) == T
                and len(cand) > 1):
            idxs = cand[:, None] + np.arange(-hT, hT, dtype=np.int32)[None, :]
            tpl = data[idxs].astype(np.float32, copy=False)
            tpl = tpl - tpl.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(tpl, axis=1) + 1e-9
            score = (tpl @ self.prev_tpl) / norms
            score = score.astype(np.float32, copy=False)

            if predicted is not None:
                dist = np.abs(cand.astype(np.float32) - predicted)
                score -= 0.005 * dist

            score += 0.012 * np.linspace(0.0, 1.0, len(cand),
                                          dtype=np.float32)

            j = int(np.argmax(score))
            if score[j] >= 0.28:
                best = j

        c = int(cand[best])
        ref = data[c - hT:c + hT].astype(np.float32, copy=False)
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
        dim = '<span style="color:#8b949e">'
        seg = self.last
        if len(seg):
            peak = float(np.max(np.abs(seg)))
            rms = float(np.sqrt(np.mean(seg * seg)))
        else:
            peak = rms = 0.0
        peak_color = "#f85149" if peak >= 0.999 else self.accent
        clip = f"  {tr('clip')}" if peak >= 0.999 else ""

        self.peak_lbl.setText(
            f'{dim}{tr("peak")}</span>  '
            f'<span style="color:{peak_color}">'
            f'{self._db(peak)} dBFS{clip}</span>')
        self.rms_lbl.setText(f'{dim}{tr("rms")}</span>  {self._db(rms)} dBFS')
        self.info_lbl.setText(
            f'{dim}{tr("window_label")}</span>  {self.n} {tr("samples")} · '
            f'{self.n / SAMPLERATE * 1000:.1f} {tr("ms")}')

        now = time.perf_counter()
        fps = self.frames / max(now - self.t0, 1e-3)
        self.frames = 0
        self.t0 = now
        self.fps_lbl.setText(
            f'{dim}{tr("fps_label")}</span>  '
            + (tr("paused") if self.frozen else f"{fps:.0f}"))

        be_name = tr(_BACKEND_NAME_KEYS.get(_BACKEND_ID, "backend_raster"))
        self.bknd_lbl.setText(
            f'{dim}{tr("backend_label")}</span>  {be_name}')
        self.bknd_lbl.setToolTip(tr("backend_tip"))

        has_err = bool(state["error"])
        if has_err != self.device_err:
            self.device_err = has_err
            self.device_lbl.setStyleSheet(
                "color: #f85149;" if has_err else "")

        if has_err:
            full = f'{tr("error_prefix")} {state["error"]}'
        else:
            name = state["device"] or tr("searching")
            full = f'{name}  ·  {SAMPLERATE / 1000:g} {tr("khz")}'

        self.device_lbl.setToolTip(full)
        self.device_lbl.setText(self.device_lbl.fontMetrics().elidedText(
            full, QtCore.Qt.TextElideMode.ElideRight,
            max(self.device_lbl.width() - 4, 20)))

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


def main():
    pg.mkQApp("PC Waveform")
    win = MainWindow()
    win.show()
    dark_titlebar(win)
    pg.exec()


if __name__ == "__main__":
    main()