"""PC Waveform — осциллограмма звука с ПК (WASAPI loopback).
Автор: riverya4life. Точка входа: python main.py."""

from PyQt6.QtWidgets import QApplication

import backend  # noqa: F401  (переменные окружения и GL-конфиг — до numpy/Qt)
import gc
import logging
import os
import sys
from functools import partial
from logging.handlers import RotatingFileHandler

from animations import fade_in, install_button_hover_effects
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from backend import _USE_GL
from settings import CONFIG_PATH

logger: logging.Logger = logging.getLogger(__name__)

GC_THRESHOLDS = (5000, 20, 20)
POST_LAUNCH_MEMORY_TRIM_DELAY_MS = 2500
LOG_MAX_BYTES = 1_000_000
LOG_BACKUP_COUNT = 3


def _app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _configure_logging() -> None:
    log_path: str = os.path.join(os.path.dirname(CONFIG_PATH), "pc-waveform.log")
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handler = RotatingFileHandler(
            log_path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[handler],
        )
    except OSError:
        logging.basicConfig(level=logging.WARNING)
        logger.exception("Не удалось настроить логирование в файл")


def _set_app_user_model_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "riverya4life.PCWaveform"
        )
    except Exception:
        logger.debug("Не удалось установить AppUserModelID", exc_info=True)


def _apply_taskbar_icon(win, icon: QtGui.QIcon) -> None:
    if sys.platform != "win32":
        return
    try:
        from PyQt5.QtWinExtras import QWinTaskbarButton  # type: ignore
    except Exception:
        try:
            from PyQt6.QtWinExtras import QWinTaskbarButton  # type: ignore
        except Exception:
            logger.debug("QWinTaskbarButton недоступен", exc_info=True)
            return
    try:
        btn = QWinTaskbarButton(win)
        btn.setWindow(win.windowHandle())
        btn.setOverlayIcon(icon)
        win._taskbar_button = btn
    except Exception:
        logger.debug("Не удалось настроить иконку панели задач", exc_info=True)


def _setup_gl_format() -> None:
    if not _USE_GL:
        return
    try:
        fmt = QtGui.QSurfaceFormat(QtGui.QSurfaceFormat.defaultFormat())
        fmt.setAlphaBufferSize(8)
        QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    except Exception:
        logger.debug("Не удалось настроить формат OpenGL", exc_info=True)


def _schedule_post_launch_tasks(win, trim_memory) -> None:
    QtCore.QTimer.singleShot(0, partial(_apply_taskbar_icon, win, win.windowIcon()))
    # После запуска: gc.freeze() исключает стартовые объекты из полных сборок,
    # затем освобождается неиспользуемая память процесса.
    QtCore.QTimer.singleShot(
        POST_LAUNCH_MEMORY_TRIM_DELAY_MS,
        lambda: trim_memory(True),
    )


def main() -> int:
    _configure_logging()
    _set_app_user_model_id()
    _setup_gl_format()
    # Реже запускаем сборщик мусора: в основном цикле почти нет циклических объектов.
    gc.set_threshold(*GC_THRESHOLDS)
    app: QApplication = pg.mkQApp("PC Waveform")
    install_button_hover_effects(app)
    # SoundCard инициализирует COM в режиме MTA при импорте. Сначала Qt должен
    # инициализировать UI-поток Windows в режиме STA, необходимом для OLE.
    from window import MainWindow, _trim_memory

    win = MainWindow()
    win.setWindowOpacity(0.0)
    win.show()
    win._open_animation = fade_in(win)
    _schedule_post_launch_tasks(win, _trim_memory)
    return pg.exec()


if __name__ == "__main__":
    raise SystemExit(main())
