"""Вспомогательные функции главного окна."""

from ctypes import WinDLL
import gc
import sys

import numpy as np

from settings import DEMO_EDGE


def _trim_memory(freeze_gc=False) -> None:
    gc.collect()
    if freeze_gc and hasattr(gc, "freeze"):
        gc.freeze()
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            k32: WinDLL = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetProcessWorkingSetSize.argtypes = [
                wintypes.HANDLE,
                ctypes.c_size_t,
                ctypes.c_size_t,
            ]
            k32.SetProcessWorkingSetSize(
                k32.GetCurrentProcess(),
                ctypes.c_size_t(-1),
                ctypes.c_size_t(-1),
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
