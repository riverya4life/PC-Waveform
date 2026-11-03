"""Анализ сигнала: BPM-детектор и пик-холд."""

import time
from collections import deque
import numpy as np


class BpmDetector:
    HOP = 512
    WIN = 1024
    HISTORY_S = 8.0
    CALC_INTERVAL_S = 1.5

    def __init__(self, samplerate: int):
        self.sr = samplerate
        self.history = deque(maxlen=int(self.HISTORY_S * samplerate / self.HOP))
        self._win = np.hanning(self.WIN).astype(np.float32)
        self._prev_spec = None
        self._buf = np.empty(self.WIN + 4096, dtype=np.float32)
        self._buf_size = 0
        self._windowed = np.empty(self.WIN, dtype=np.float32)
        self.bpm = 0.0
        self.confidence = 0.0
        self._last_calc = 0.0

    def feed(self, mono: np.ndarray):
        if mono is None or mono.size == 0:
            return
        data = np.asarray(mono, dtype=np.float32)
        offset = 0
        while offset < data.size:
            count = min(self._buf.size - self._buf_size, data.size - offset)
            self._buf[self._buf_size : self._buf_size + count] = data[
                offset : offset + count
            ]
            self._buf_size += count
            offset += count
            while self._buf_size >= self.WIN:
                self._process(self._buf[: self.WIN])
                # Keep the 50% overlap without allocating a new array each hop.
                remaining = self._buf_size - self.HOP
                self._buf[:remaining] = self._buf[self.HOP : self._buf_size]
                self._buf_size = remaining

    def _process(self, seg: np.ndarray):
        np.multiply(seg, self._win, out=self._windowed)
        spec = np.abs(np.fft.rfft(self._windowed))
        if self._prev_spec is not None and spec.shape == self._prev_spec.shape:
            flux = float(np.maximum(0.0, spec - self._prev_spec).sum())
            self.history.append(flux)
        self._prev_spec = spec

    def maybe_update(self) -> bool:
        now = time.perf_counter()
        if now - self._last_calc < self.CALC_INTERVAL_S:
            return False
        self._last_calc = now
        self._compute()
        return True

    def _compute(self):
        if len(self.history) < int(2.0 * self.sr / self.HOP):
            return
        x = np.asarray(self.history, dtype=np.float32)
        x = x - x.mean()
        if x.std() < 1e-6:
            self.bpm *= 0.9
            self.confidence = 0.0
            return
        n = len(x)
        f = np.fft.rfft(x, 2 * n)
        ac = np.fft.irfft(f * np.conj(f))[:n]
        if ac[0] <= 1e-9:
            return
        ac /= ac[0]
        frame_rate = self.sr / self.HOP
        min_lag = max(2, int(frame_rate * 60.0 / 200.0))
        max_lag = min(n - 1, int(frame_rate * 60.0 / 60.0))
        if max_lag <= min_lag:
            return
        region = ac[min_lag:max_lag]
        best = int(np.argmax(region))
        conf = float(region[best])
        lag = min_lag + best
        bpm = frame_rate * 60.0 / lag
        self.confidence = conf
        if conf > 0.15:
            if self.bpm < 1.0:
                self.bpm = bpm
            else:
                self.bpm = 0.85 * self.bpm + 0.15 * bpm
        else:
            self.bpm *= 0.95


class PeakHold:
    """Огибающая пиков с затуханием. Затухание/сглаживание считаются от dt
    (эталон — 60 кадров/с), поэтому вид не зависит от FPS."""

    REF_FPS = 60.0

    def __init__(self, decay: float = 0.94, smooth: float = 0.5, window: int = 15):
        self.decay = decay
        self.smooth = smooth
        self.window = window if window % 2 == 1 else window + 1
        self._h = self.window // 2
        self._kernel = np.full(self.window, 1.0 / self.window, dtype=np.float32)
        self.env = None
        self._sm = None
        self._abs = None
        self._tmp = None
        self._pad = None

    def _alloc(self, m: int):
        f = np.float32
        self.env = np.empty(m, f)
        self._sm = np.empty(m, f)
        self._abs = np.empty(m, f)
        self._tmp = np.empty(m, f)
        self._pad = np.empty(m + 2 * self._h, f)

    def update(self, y: np.ndarray, dt: float = None) -> np.ndarray:
        m = y.shape[0]
        if self.env is None or self.env.shape[0] != m:
            self._alloc(m)
            np.abs(y, out=self.env)
            self._sm[:] = self.env
        else:
            if dt is None:
                k_dec, k_sm = self.decay, self.smooth
            else:
                e = max(dt, 1e-4) * self.REF_FPS
                k_dec = self.decay**e
                k_sm = self.smooth**e
            np.abs(y, out=self._abs)
            self.env *= k_dec
            np.maximum(self.env, self._abs, out=self.env)
            if self.smooth > 0.0:
                self._sm *= k_sm
                np.multiply(self.env, 1.0 - k_sm, out=self._tmp)
                self._sm += self._tmp

        src = self._sm if self.smooth > 0.0 else self.env
        if self.window <= 1 or m <= self._h:
            return src
        # скользящее среднее с «приклеенными» краями (раньше нули по краям
        # просаживали линию вниз у левой и правой границы)
        h = self._h
        pad = self._pad
        pad[h : h + m] = src
        pad[:h] = src[0]
        pad[h + m :] = src[-1]
        return np.convolve(pad, self._kernel, mode="valid")

    def reset(self):
        self.env = None
        self._sm = None
