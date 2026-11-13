"""Выбор графического бэкенда (OpenGL/растр) и переменные окружения.
Должен импортироваться ДО numpy/Qt."""

import os
import pyqtgraph as pg

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

_FORCED: str = os.environ.get("PCWAVEFORM_BACKEND", "").lower()

if _FORCED == "vulkan":
    os.environ.setdefault("QSG_RHI_BACKEND", "vulkan")
    os.environ.setdefault("QT_QUICK_BACKEND", "vulkan")

try:
    import OpenGL  # noqa: F401

    _HAS_GL = True
except Exception:
    _HAS_GL = False

if _FORCED == "raster":
    _USE_GL = False
elif _FORCED == "vulkan":
    _USE_GL: bool = _HAS_GL
else:
    _USE_GL: bool = _HAS_GL

_BACKEND_ID: str = (
    "vulkan"
    if (_USE_GL and _FORCED == "vulkan")
    else ("opengl" if _USE_GL else "raster")
)

_BACKEND_NAME_KEYS: dict[str, str] = {
    "opengl": "backend_opengl",
    "vulkan": "backend_vulkan",
    "raster": "backend_raster",
}

pg.setConfigOptions(useOpenGL=_USE_GL, antialias=False, foreground="#8b949e")
