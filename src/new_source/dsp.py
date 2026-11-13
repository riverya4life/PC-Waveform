"""Полосовые фильтры (RBJ cookbook, биквад)."""

from typing import Any

import numpy as np
from settings import SAMPLERATE

try:
    from scipy.signal import lfilter as _lfilter

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class Biquad:
    def __init__(self) -> None:
        self.b0 = 1.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.z1 = 0.0
        self.z2 = 0.0
        self._b: np.ndarray[Any, np.dtype[np.floating]] = np.array(
            [1.0, 0.0, 0.0], dtype=np.float64
        )
        self._a: np.ndarray[Any, np.dtype[np.floating]] = np.array(
            [1.0, 0.0, 0.0], dtype=np.float64
        )
        self._zi: np.ndarray[Any, np.dtype[np.floating]] = np.zeros(2, dtype=np.float64)

    def _design(self, b0, b1, b2, a0, a1, a2) -> None:
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0
        self._b[:] = (self.b0, self.b1, self.b2)
        self._a[:] = (1.0, self.a1, self.a2)

    def lowpass(self, fc: float, q: float = 0.707) -> None:
        w0: float = 2.0 * np.pi * fc / SAMPLERATE
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / (2.0 * q)
        self._design((1 - cw) / 2, 1 - cw, (1 - cw) / 2, 1 + alpha, -2 * cw, 1 - alpha)

    def highpass(self, fc: float, q: float = 0.707) -> None:
        w0: float = 2.0 * np.pi * fc / SAMPLERATE
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / (2.0 * q)
        self._design(
            (1 + cw) / 2,
            -(1 + cw),
            (1 + cw) / 2,
            1 + alpha,
            -2 * cw,
            1 - alpha,
        )

    def bandpass(self, fc: float, q: float = 1.0) -> None:
        w0: float = 2.0 * np.pi * fc / SAMPLERATE
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / (2.0 * q)
        self._design(alpha, 0.0, -alpha, 1 + alpha, -2 * cw, 1 - alpha)

    def reset(self) -> None:
        self.z1 = self.z2 = 0.0
        self._zi[:] = 0.0

    def process(self, x, out):
        if _HAS_SCIPY:
            try:
                y, self._zi = _lfilter(
                    self._b,
                    self._a,
                    x.astype(np.float64, copy=False),
                    zi=self._zi,
                )
                out[: len(y)] = y.astype(np.float32, copy=False)
                return out
            except Exception:
                pass
        b0, b1, b2 = self.b0, self.b1, self.b2
        a1, a2 = self.a1, self.a2
        z1, z2 = self.z1, self.z2
        for i in range(len(x)):
            xi = float(x[i])
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            out[i] = yi
        self.z1, self.z2 = z1, z2
        return out


_PRESETS: dict[str, tuple[str, float, float]] = {
    "low": ("lowpass", 200.0, 0.707),
    "mid": ("bandpass", 1000.0, 1.2),
    "high": ("highpass", 4000.0, 0.707),
}


def make_filter(kind: str) -> None | Biquad:
    if not kind or kind == "off" or kind not in _PRESETS:
        return None
    q = Biquad()
    name, fc, qv = _PRESETS[kind]
    getattr(q, name)(fc, qv)
    return q
