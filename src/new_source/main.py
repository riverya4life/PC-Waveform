"""PC Waveform — осциллограмма звука с ПК (WASAPI loopback) by riverya4life.
Точка входа: python main.py"""

import backend  # noqa: F401  (переменные окружения и GL-конфиг — до numpy/Qt)
import gc
import os
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from backend import _USE_GL
from widgets import dark_titlebar
from window import MainWindow, _trim_memory


def _app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _set_app_user_model_id():
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "riverya4life.PCWaveform"
        )
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
    # реже запускаем сборщик мусора: в hot-path почти нет циклических объектов
    gc.set_threshold(5000, 20, 20)
    app = pg.mkQApp("PC Waveform")
    win = MainWindow()
    win.show()
    dark_titlebar(win)
    QtCore.QTimer.singleShot(0, lambda: _apply_taskbar_icon(win, win.windowIcon()))
    # после старта: gc.freeze() (стартовые объекты не сканируются при полных
    # сборках) + сброс working set
    QtCore.QTimer.singleShot(2500, lambda: _trim_memory(True))
    pg.exec()


if __name__ == "__main__":
    main()
