"""Константы, настройки (INI в %APPDATA%/PC Waveform/settings.ini) и локализация ru/uk/en."""

from io import TextIOWrapper

import backend  # noqa: F401
import configparser
import os
import sys

SAMPLERATE = 48000
CHUNK = 256
UPDATE_MS = 1
STATS_MS = 50
HZ_EST_SAMPLES = 4096
BPM_HISTORY_SAMPLES = 4096
BASS_HISTORY_SAMPLES = 8192
FPS_AVG_S = 0.25
SLOW_STATS_S = 0.25
WATCHDOG_MS = 16
MAX_N = 32768
TRIGGER_MAX_AGE = 1536
MAX_CH = 8

SAMPLE_OPTIONS: list[int] = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
GAIN_OPTIONS: list[int] = [1, 2, 4, 8]
FPS_OPTIONS: list[int] = [0, -1, 30, 60, 75, 90, 120, 144, 165, 240]

CHANNEL_LAYOUTS: list[tuple[str, int, str]] = [
    ("mono", 1, "mono"),
    ("stereo", 2, "stereo"),
    ("quad", 4, "quad"),
    ("5.1", 6, "5_1"),
    ("7.1", 8, "7_1"),
]
CH_LAYOUT_TO_N: dict[str, int] = {k: n for k, n, _ in CHANNEL_LAYOUTS}

TICKS: dict[int, int] = {
    256: 64,
    512: 128,
    1024: 250,
    2048: 500,
    4096: 500,
    8192: 1000,
    16384: 2000,
    32768: 4000,
}

DEMO_EDGE = 0.12


def _config_dir() -> str:
    if sys.platform == "win32":
        base: str = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
            os.path.expanduser("~"), ".config"
        )
    return os.path.join(base, "PC Waveform")


CONFIG_PATH: str = os.path.join(_config_dir(), "settings.ini")

DEFAULT_CONFIG = {
    "lang": "ru",
    "theme": "mint",
    "n": 4096,
    "gain": 1,
    "stabilization": True,
    "glow": False,
    "ch_layout": "mono",
    "device_id": None,
    "fade": True,
    "custom_color": "#ff6bcb",
    "fps": 0,
    "fps_customized": False,
    "bandpass": "off",
    "gradient": False,
    "peak_hold": True,
    "bpm_on": True,
    "smooth_stab": True,
    "edge_clip": False,
    "solo_geom": None,
    "solo_pinned": False,
}


def _bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _parse_geom(v):
    """Геометрия окна: (x, y, w, h) из кортежа/строки «x,y,w,h» или None."""
    if v is None:
        return None
    try:
        parts = v if isinstance(v, (list, tuple)) else str(v).split(",")
        x, y, w, h = (int(float(p)) for p in parts)
    except Exception:
        return None
    if not (100 <= w <= 10000 and 60 <= h <= 10000):
        return None
    return (x, y, w, h)


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

    if cfg.get("ch_layout") not in CH_LAYOUT_TO_N:
        cfg["ch_layout"] = "mono"

    if cfg.get("bandpass") not in ("off", "low", "mid", "high"):
        cfg["bandpass"] = "off"

    for k in (
        "gradient",
        "peak_hold",
        "bpm_on",
        "smooth_stab",
        "edge_clip",
        "solo_pinned",
    ):
        cfg[k] = _bool(cfg.get(k, DEFAULT_CONFIG[k]))

    cfg["solo_geom"] = _parse_geom(cfg.get("solo_geom"))

    cfg["fps_customized"] = _bool(cfg.get("fps_customized", False))
    if cfg["fps"] == 0 and not cfg["fps_customized"]:
        cfg["fps"] = DEFAULT_CONFIG["fps"]

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
        s: configparser.SectionProxy = cp["ui"]
        for k in ("lang", "theme", "ch_layout", "custom_color", "bandpass", "solo_geom"):
            if k in s:
                cfg[k] = s.get(k)
        for k in ("n", "gain", "fps"):
            if k in s:
                try:
                    cfg[k] = s.getint(k)
                except Exception:
                    pass
        for k in (
            "stabilization",
            "glow",
            "fade",
            "gradient",
            "peak_hold",
            "bpm_on",
            "smooth_stab",
            "edge_clip",
            "solo_pinned",
            "fps_customized",
        ):
            if k in s:
                cfg[k] = s.getboolean(k, DEFAULT_CONFIG[k])
    if cp.has_section("audio") and "device_id" in cp["audio"]:
        v: str = cp["audio"].get("device_id", "").strip()
        cfg["device_id"] = v if v and v.lower() not in ("none", "null") else None
    return cfg


def _load_config() -> dict:
    if os.path.isfile(CONFIG_PATH):
        loaded = _load_from_ini(CONFIG_PATH)
        if loaded:
            return _validate(loaded)
    return _validate(dict(DEFAULT_CONFIG))


def _save_config(cfg: dict) -> None:
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
            "ch_layout": str(cfg.get("ch_layout", "mono")),
            "fade": "true" if _bool(cfg.get("fade", True)) else "false",
            "custom_color": str(
                cfg.get("custom_color", DEFAULT_CONFIG["custom_color"])
            ),
            "fps": str(int(cfg.get("fps", 0))),
            "fps_customized": (
                "true" if _bool(cfg.get("fps_customized", False)) else "false"
            ),
            "bandpass": str(cfg.get("bandpass", "off")),
            "gradient": ("true" if _bool(cfg.get("gradient", False)) else "false"),
            "peak_hold": ("true" if _bool(cfg.get("peak_hold", True)) else "false"),
            "bpm_on": "true" if _bool(cfg.get("bpm_on", True)) else "false",
            "smooth_stab": ("true" if _bool(cfg.get("smooth_stab", True)) else "false"),
            "edge_clip": ("true" if _bool(cfg.get("edge_clip", True)) else "false"),
        }
        cp["ui"]["solo_pinned"] = (
            "true" if _bool(cfg.get("solo_pinned", False)) else "false"
        )
        geom = _parse_geom(cfg.get("solo_geom"))
        cp["ui"]["solo_geom"] = ",".join(str(v) for v in geom) if geom else ""
        did = cfg.get("device_id")
        cp["audio"] = {"device_id": "" if did is None else str(did)}
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            cp.write(f)
    except Exception:
        pass


CONFIG = _load_config()

LANGUAGES: list[tuple[str, str]] = [
    ("Русский", "ru"),
    ("Українська", "uk"),
    ("English", "en"),
]
LANG = {"current": CONFIG["lang"]}

TRANSLATIONS: dict[str, dict[str, str]] = {
    "ru": {
        "brand": "PC Waveform",
        "main_title": "PC Waveform — осциллограмма системного звука (github: riverya4life)",
        "solo_title": "PC Waveform — соло (ПКМ или Esc — вернуться)",
        "subtitle": "осциллограмма системного звука",
        "freeze": "Заморозить",
        "resume": "Продолжить",
        "stabilization": "Стабилизация",
        "stabilization_tip": "Динамическая стабилизация: захват периода и фазы",
        "glow": "Свечение",
        "fade": "Затухание",
        "fade_tip": "Мягкое затухание сигнала к краям окна + AGC",
        "window": "Окно",
        "gain": "Усиление",
        "stream": "Поток",
        "theme": "Тема",
        "language": "Язык",
        "channels": "Каналы",
        "ch_layout": "Раскладка",
        "quad": "Quad",
        "5_1": "5.1",
        "7_1": "7.1",
        "fps": "Частота кадров",
        "fps_unlimited": "Максимум",
        "fps_monitor": "По монитору",
        "fps_tip": "Ограничение FPS осциллограммы.",
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
        "hz_label": "ЧАСТОТА",
        "hz_tip": "Частота баса: пик спектра 20–250 Гц",
        "backend_opengl": "OpenGL",
        "backend_vulkan": "Vulkan",
        "backend_raster": "Raster",
        "backend_tip": "Графический движок отрисовки.",
        "clip": "CLIP",
        "paused": "пауза",
        "error_prefix": "Ошибка:",
        "audio_error_title": "Ошибка захвата аудио",
        "audio_error_click_details": "Нажмите, чтобы посмотреть подробности",
        "audio_error_hint": "Проверьте подключение устройства. Технические подробности можно открыть отдельно.",
        "audio_error_show_details": "Технические детали",
        "audio_error_hide_details": "Скрыть детали",
        "close": "Закрыть",
        "audio_error_device": "Устройство",
        "audio_error_device_id": "ID устройства",
        "audio_error_sample_rate": "Частота дискретизации",
        "audio_error_block_size": "Размер блока",
        "audio_error_channels": "Запрошено каналов",
        "audio_error_type": "Тип ошибки",
        "audio_error_message": "Сообщение",
        "audio_error_raw": "Данные исключения",
        "audio_error_traceback": "Стек вызовов",
        "audio_error_unknown_type": "неизвестно",
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
        "theme_custom": "Свой цвет",
        "theme_custom_tip": "Выбрать свой цвет темы",
        "theme_system": "Акцент системы",
        "theme_system_tip": "Брать цвет из акцента системы и следовать за ним",
        "theme_system_na": "недоступно на этой системе",
        "theme_close": "Готово",
        "theme_presets": "ПАЛИТРА",
        "settings": "Настройки",
        "settings_tip": "Язык, каналы, эффекты и усиление",
        "settings_general": "ОБЩИЕ",
        "settings_signal": "СИГНАЛ",
        "settings_effects": "ЭФФЕКТЫ",
        "band_off": "Выкл",
        "band_low": "Низ",
        "band_mid": "Середина",
        "band_high": "Верх",
        "bandpass": "Фильтр",
        "bandpass_tip": "Полосовой фильтр на отображаемый сигнал",
        "gradient": "Градиент",
        "gradient_tip": "Раскрасить волну по амплитуде",
        "peak_hold": "Пик-холд",
        "peak_hold_tip": "Огибающая пиков с затуханием",
        "bpm": "BPM",
        "bpm_tip": "Определение темпа",
        "smooth_stab": "Плавная стабилизация (БЕТА)",
        "smooth_stab_tip": "Плавная смена кадров; выключено — классическая стабилизация",
        "edge_clip": "Об края",
        "edge_clip_tip": "Жестко обрезать пики по границам окна",
    },
    "uk": {
        "brand": "PC Waveform",
        "main_title": "PC Waveform — осцилограма системного звуку (github: riverya4life)",
        "solo_title": "PC Waveform — соло (ПКМ або Esc — повернутися)",
        "subtitle": "осцилограма системного звуку",
        "freeze": "Заморозити",
        "resume": "Продовжити",
        "stabilization": "Стабілізація",
        "stabilization_tip": "Динамічна стабілізація",
        "glow": "Світіння",
        "fade": "Загасання",
        "fade_tip": "М'яке загасання сигналу до країв вікна",
        "window": "Вікно",
        "gain": "Підсилення",
        "stream": "Потік",
        "theme": "Тема",
        "language": "Мова",
        "channels": "Канали",
        "ch_layout": "Розкладка",
        "quad": "Quad",
        "5_1": "5.1",
        "7_1": "7.1",
        "fps": "Частота кадрів",
        "fps_unlimited": "Максимум",
        "fps_monitor": "За монітором",
        "fps_tip": "Обмеження FPS",
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
        "hz_label": "ЧАСТОТА",
        "hz_tip": "Частота басу: пік спектра 20–250 Гц",
        "backend_opengl": "OpenGL",
        "backend_vulkan": "Vulkan",
        "backend_raster": "Raster",
        "backend_tip": "Графічний рушій",
        "clip": "CLIP",
        "paused": "пауза",
        "error_prefix": "Помилка:",
        "audio_error_title": "Помилка захоплення аудіо",
        "audio_error_click_details": "Натисніть, щоб переглянути подробиці",
        "audio_error_hint": "Перевірте підключення пристрою. Технічні подробиці можна відкрити окремо.",
        "audio_error_show_details": "Технічні деталі",
        "audio_error_hide_details": "Сховати деталі",
        "close": "Закрити",
        "audio_error_device": "Пристрій",
        "audio_error_device_id": "ID пристрою",
        "audio_error_sample_rate": "Частота дискретизації",
        "audio_error_block_size": "Розмір блока",
        "audio_error_channels": "Запитано каналів",
        "audio_error_type": "Тип помилки",
        "audio_error_message": "Повідомлення",
        "audio_error_raw": "Дані винятку",
        "audio_error_traceback": "Стек викликів",
        "audio_error_unknown_type": "невідомо",
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
        "theme_turquoise": "Бирюза",
        "theme_white": "Білий",
        "theme_custom": "Свій колір",
        "theme_custom_tip": "Вибрати свій колір теми",
        "theme_system": "Акцент системи",
        "theme_system_tip": "Брати колір з акценту системи та слідувати за ним",
        "theme_system_na": "недоступно в цій системі",
        "theme_close": "Готово",
        "theme_presets": "ПАЛІТРА",
        "settings": "Налаштування",
        "settings_tip": "Мова, канали, ефекти",
        "settings_general": "ЗАГАЛЬНІ",
        "settings_signal": "СИГНАЛ",
        "settings_effects": "ЕФЕКТИ",
        "band_off": "Вимк",
        "band_low": "Низ",
        "band_mid": "Середина",
        "band_high": "Верх",
        "bandpass": "Фільтр",
        "bandpass_tip": "Смуговий фільтр",
        "gradient": "Градієнт",
        "gradient_tip": "Розфарбувати хвилю",
        "peak_hold": "Пік-холд",
        "peak_hold_tip": "Огинаюча піків",
        "bpm": "BPM",
        "bpm_tip": "Визначення темпу",
        "smooth_stab": "Плавна стабілізація",
        "smooth_stab_tip": "Плавна зміна кадрів; вимкнено — класична стабілізація",
        "edge_clip": "Об краї",
        "edge_clip_tip": "Жорстко обрізати піки по межах вікна",
    },
    "en": {
        "brand": "PC Waveform",
        "main_title": "PC Waveform — system audio waveform (github: riverya4life)",
        "solo_title": "PC Waveform — solo (RMB or Esc — exit)",
        "subtitle": "system audio waveform",
        "freeze": "Freeze",
        "resume": "Resume",
        "stabilization": "Stabilization",
        "stabilization_tip": "Dynamic stabilization",
        "glow": "Glow",
        "fade": "Fade",
        "fade_tip": "Soft edge fade of the signal + AGC",
        "window": "Window",
        "gain": "Gain",
        "stream": "Stream",
        "theme": "Theme",
        "language": "Language",
        "channels": "Channels",
        "ch_layout": "Layout",
        "quad": "Quad",
        "5_1": "5.1",
        "7_1": "7.1",
        "fps": "Frame rate",
        "fps_unlimited": "Maximum",
        "fps_monitor": "Monitor sync",
        "fps_tip": "Oscilloscope FPS cap",
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
        "hz_label": "FREQ",
        "hz_tip": "Bass frequency: spectrum peak 20–250 Hz",
        "backend_opengl": "OpenGL",
        "backend_vulkan": "Vulkan",
        "backend_raster": "Raster",
        "backend_tip": "Rendering backend",
        "clip": "CLIP",
        "paused": "paused",
        "error_prefix": "Error:",
        "audio_error_title": "Audio capture error",
        "audio_error_click_details": "Click to view details",
        "audio_error_hint": "Check the device connection. Technical details can be opened separately.",
        "audio_error_show_details": "Technical details",
        "audio_error_hide_details": "Hide details",
        "close": "Close",
        "audio_error_device": "Device",
        "audio_error_device_id": "Device ID",
        "audio_error_sample_rate": "Sample rate",
        "audio_error_block_size": "Block size",
        "audio_error_channels": "Channels requested",
        "audio_error_type": "Error type",
        "audio_error_message": "Message",
        "audio_error_raw": "Exception details",
        "audio_error_traceback": "Traceback",
        "audio_error_unknown_type": "unknown",
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
        "theme_custom": "Custom",
        "theme_custom_tip": "Pick a custom theme colour",
        "theme_system": "System accent",
        "theme_system_tip": "Use the system accent colour and follow its changes",
        "theme_system_na": "not available on this system",
        "theme_close": "Done",
        "theme_presets": "PALETTE",
        "settings": "Settings",
        "settings_tip": "Language, channels, effects",
        "settings_general": "GENERAL",
        "settings_signal": "SIGNAL",
        "settings_effects": "EFFECTS",
        "band_off": "Off",
        "band_low": "Low",
        "band_mid": "Mid",
        "band_high": "High",
        "bandpass": "Filter",
        "bandpass_tip": "Bandpass filter on the displayed signal",
        "gradient": "Gradient",
        "gradient_tip": "Color the wave by amplitude",
        "peak_hold": "Peak hold",
        "peak_hold_tip": "Peak envelope with decay",
        "bpm": "BPM",
        "bpm_tip": "Tempo detection",
        "smooth_stab": "Smooth stabilization (BETA)",
        "smooth_stab_tip": "Smooth frame transitions; off = classic stabilization",
        "edge_clip": "Edge clip",
        "edge_clip_tip": "Hard clip peaks at window bounds",
    },
}


def tr(key: str) -> str:
    d: dict[str, str] = TRANSLATIONS.get(LANG["current"], TRANSLATIONS["ru"])
    return d.get(key, key)


def cap_text(key: str) -> str:
    return tr(key).upper()