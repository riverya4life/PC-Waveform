"""Триггер стабилизации и оценка частоты (миксин для MainWindow)."""

from typing import Any

import backend  # noqa: F401  (переменные окружения и GL-конфиг — до numpy/Qt)
import math
import numpy as np
from settings import BASS_HISTORY_SAMPLES, SAMPLERATE, TRIGGER_MAX_AGE
import audio


def _absmax(a) -> float:
    """max|a| без аллокации временного массива abs()."""
    if a.size == 0:
        return 0.0
    return max(float(a.max()), -float(a.min()))


class TriggerMixin:
    """Триггер (стабилизация) и оценка частоты — методы окна MainWindow."""

    def find_trigger(self, data, shift):
        """Динамическая стабилизация (в духе FL Studio):
        1) ищем нулевые пересечения с гистерезисом
        2) оцениваем период и держим фазовый захват
        3) при смене высоты тона период плавно подстраивается"""
        n = self.n
        L: int = len(data)
        half = n // 2

        # При тишине выходим до np.convolve — это экономит около 60 % времени
        # обработки триггера в паузах.
        if L > half * 2:
            seg_amp: float = _absmax(data[half : L - half])
            if seg_amp < 1e-5:
                self._lock_quality *= 0.9
                return None

        s: np.ndarray[Any, np.dtype[Any]] = np.convolve(data, self._kernel, mode="same")
        s -= s.mean()
        amp: float = _absmax(s[half : L - half])
        if amp < 1e-4:
            self._lock_quality *= 0.9
            return None

        h: float = (0.12 + 0.06 * (1.0 - min(1.0, amp))) * amp
        # Состояние гистерезиса меняется только в «ненулевых» точках, поэтому фронты
        # (−1 → +1) находим по соседним ненулевым отсчётам — без O(L) заливки.
        hi = s > h
        nz_idx: np.ndarray[Any, np.dtype[np.signedinteger]] = np.flatnonzero(
            hi | (s < -h)
        )
        if nz_idx.size == 0:
            return None
        up = hi[nz_idx]
        edges: np.ndarray[Any, np.dtype[np.signedinteger]] = nz_idx[1:][
            (~up[:-1]) & up[1:]
        ]
        if edges.size == 0:
            return None
        zc = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0)) + 1
        if zc.size == 0:
            return None
        pos = np.searchsorted(zc, edges, side="right") - 1
        valid = pos >= 0
        if not valid.any():
            return None
        cand: np.ndarray[Any, np.dtype[np.signedinteger]] = zc[pos[valid]]
        cand: np.ndarray[Any, np.dtype[np.signedinteger]] = cand[
            (cand >= half) & (cand <= L - half)
        ]
        if cand.size == 0:
            return None

        periods: np.ndarray[Any, np.dtype[Any]] = np.diff(cand.astype(np.float64))
        min_p: float = SAMPLERATE / 8000.0
        max_p: float = SAMPLERATE / 20.0
        periods: np.ndarray[Any, np.dtype[Any]] = periods[
            (periods >= min_p) & (periods <= max_p)
        ]
        period_est = 0.0
        if periods.size > 0:
            period_est = float(np.median(periods))
            if self._period_smooth < 1.0:
                self._period_smooth: float = period_est
            else:
                ratio: float = period_est / max(self._period_smooth, 1.0)
                pa: float = 0.35 if (ratio > 1.12 or ratio < 0.88) else 0.12
                self._period_smooth: float = (
                    self._period_smooth * (1.0 - pa) + period_est * pa
                )

        predicted = None
        if self._crossing_smooth is not None and shift >= 0:
            predicted = self._crossing_smooth - shift

        recent: np.ndarray[Any, np.dtype[np.signedinteger]] = cand[
            cand >= L - half - TRIGGER_MAX_AGE
        ]
        if recent.size:
            cand: np.ndarray[Any, np.dtype[np.signedinteger]] = recent
        cand: np.ndarray[Any, np.dtype[np.signedinteger]] = cand[-32:]

        if self._period_smooth > 16.0:
            T = int(min(max(self._period_smooth * 0.5, 48), min(256, n // 2)))
        else:
            T: int = min(192, n // 2)
        T: int = max(32, T - (T % 2))
        hT: int = T // 2

        # --- Векторизованный скоринг кандидатов (без Python-цикла) ---
        n_cand: int = len(cand)
        scores: np.ndarray[Any, np.dtype[np.floating]] = np.zeros(
            n_cand, dtype=np.float32
        )
        if self.prev_tpl is not None and len(self.prev_tpl) == T:
            valid_mask = (cand - hT >= 0) & (cand + hT <= L)
            if valid_mask.any():
                idx_offsets: np.ndarray[Any, np.dtype[np.signedinteger]] = np.arange(
                    -hT, hT
                )
                c_vals: np.ndarray[Any, np.dtype[np.signedinteger]] = cand[valid_mask]
                slices = data[c_vals[:, None] + idx_offsets[None, :]]
                slices = slices - slices.mean(axis=1, keepdims=True)
                norms = np.linalg.norm(slices, axis=1) + 1e-9
                dots = (slices @ self.prev_tpl) / norms
                scores[valid_mask] += dots.astype(np.float32, copy=False)
        if predicted is not None:
            cand_f: np.ndarray[Any, np.dtype[np.floating]] = cand.astype(np.float32)
            if self._period_smooth > 8.0:
                d = (cand_f - predicted) % self._period_smooth
                d = np.minimum(d, self._period_smooth - d)
                scores -= (d / self._period_smooth) * np.float32(0.9)
            else:
                scores -= np.abs(cand_f - predicted) * np.float32(0.004)
        scores += (
            np.float32(0.015) * np.arange(n_cand, dtype=np.float32) / max(n_cand - 1, 1)
        )
        best = int(np.argmax(scores))
        best_score = float(scores[best])
        if best_score < -0.5 and self._lock_quality < 0.2:
            best: int = n_cand - 1

        c = int(cand[best])
        if c - hT >= 0 and c + hT <= L:
            ref = data[c - hT : c + hT].astype(np.float32, copy=False)
            ref = ref - ref.mean()
            self.prev_tpl = ref / (np.linalg.norm(ref) + 1e-9)

        a, b = float(s[c - 1]), float(s[c])
        frac: float = a / (a - b) if a != b else 0.0
        frac: float = min(max(frac, 0.0), 1.0)
        crossing: float = (c - 1) + frac

        q = 0.5
        if best_score > 0.5:
            q = 0.9
        elif best_score > 0.2:
            q = 0.7
        elif best_score > 0.0:
            q = 0.45
        else:
            q = 0.2
        self._lock_quality: float = self._lock_quality * 0.7 + q * 0.3

        if self._crossing_smooth is None or shift <= 0:
            self._crossing_smooth: float = crossing
        else:
            expected = self._crossing_smooth - shift
            delta = crossing - expected
            if self._period_smooth > 8.0:
                half_p: float = self._period_smooth * 0.5
                while delta > half_p:
                    delta -= self._period_smooth
                while delta < -half_p:
                    delta += self._period_smooth
            alpha: float = 0.18 + 0.35 * self._lock_quality
            if abs(delta) > n * 0.35:
                self._crossing_smooth: float = crossing
            else:
                self._crossing_smooth = crossing - (1.0 - alpha) * delta

        cs = self._crossing_smooth
        start = int(cs) - half
        x_off = -(cs - int(cs))
        return start, x_off

    HOLD_MAX_S = 0.25

    def _hold_trigger(self, shift, need, n):
        """Триггер не нашёл фронт — не прыгаем на несинхронный кадр, а ведём
        предыдущую фазу вперёд (с поправкой на сдвиг и период). Не дольше
        HOLD_MAX_S, после чего используется обычный запасной режим."""
        cs = self._crossing_smooth
        if cs is None:
            return None
        held: Any | int = getattr(self, "_hold_samples", 0) + max(shift, 0)
        if held > SAMPLERATE * self.HOLD_MAX_S:
            return None
        self._hold_samples: Any | int = held
        half = n // 2
        cs -= max(shift, 0)
        p: float = self._period_smooth
        if p > 8.0:
            lo, hi = half, need - n + half
            while cs < lo:
                cs += p
            while cs > hi:
                cs -= p
        if cs < half or cs > need - n + half:
            return None
        self._crossing_smooth = cs
        return int(cs) - half, -(cs - int(cs))

    def _estimate_hz(self, seg: np.ndarray) -> float:
        if seg is None or len(seg) < 32:
            return 0.0
        x: np.ndarray[Any, np.dtype[np.floating]] = seg.astype(np.float64, copy=False)
        if len(x) >= 8:
            k = np.ones(5, dtype=np.float64) / 5.0
            x: np.ndarray[Any, np.dtype[np.floating]] = np.convolve(x, k, mode="same")
        amp = float(np.max(np.abs(x)))
        if amp < 1e-4:
            return 0.0
        zc: np.ndarray[Any, np.dtype[np.signedinteger]] = np.flatnonzero(
            (x[:-1] < 0.0) & (x[1:] >= 0.0)
        )
        if zc.size < 2:
            return 0.0
        # субсэмпловая линейная интерполяция пересечений нуля —
        # даёт осмысленные сотые доли герца
        y0: (
            np.ndarray[Any, np.dtype[np.floating]]
            | np.ndarray[Any, np.dtype[np.floating]]
        ) = x[zc]
        y1: (
            np.ndarray[Any, np.dtype[np.floating]]
            | np.ndarray[Any, np.dtype[np.floating]]
        ) = x[zc + 1]
        zf = zc + (-y0) / np.maximum(y1 - y0, 1e-12)
        periods: np.ndarray[Any, np.dtype[Any]] = np.diff(zf)
        min_p: float = SAMPLERATE / 8000.0
        max_p: float = SAMPLERATE / 20.0
        periods: np.ndarray[Any, np.dtype[Any]] = periods[
            (periods >= min_p) & (periods <= max_p)
        ]
        if periods.size == 0:
            return 0.0
        med = float(np.median(periods))
        good: np.ndarray[Any, np.dtype[Any]] = periods[
            np.abs(periods - med) <= 0.15 * med
        ]
        p: float = float(np.mean(good)) if good.size else med
        if p < 1.0:
            return 0.0
        return float(SAMPLERATE / p)

    BASS_LO = 20.0

    BASS_HI = 250.0

    BASS_WIN: int = BASS_HISTORY_SAMPLES

    BASS_NFFT = 8192  # Уменьшили FFT в 4 раза!

    def _estimate_bass_hz(self) -> float:
        """Частота баса: пик спектра в диапазоне 20–250 Гц, как показания в FL Studio.
        Берём последние BASS_WIN сэмплов из кольца, окно Ханна, rfft с zero-padding,
        параболическая интерполяция по лог-амплитуде — точность в сотые доли Гц.
        """
        N: int = self.BASS_WIN
        if audio.state["total_samples"] < N:
            return 0.0
        buf: Any | None = getattr(self, "_bass_buf", None)
        if buf is None:
            buf = self._bass_buf = np.empty(N, dtype=np.float32)
            self._bass_win: np.ndarray[Any, np.dtype[np.floating]] = np.hanning(
                N
            ).astype(np.float32)
            self._bass_wsum = float(self._bass_win.sum())
        with audio.lock:
            e: int = audio._ring_w[0] + audio.RING_N
            st: int = e - N
            buf[:] = audio.ring_m[st:e]  # (L+R)/2 уже посчитан в аудио-потоке
        buf -= float(buf.mean())
        # Ранний выход на тишине — без FFT
        if _absmax(buf) < 1e-5:
            return 0.0
        buf *= self._bass_win
        spec: np.ndarray[Any, np.dtype[Any]] = np.abs(np.fft.rfft(buf, self.BASS_NFFT))
        df: float = SAMPLERATE / self.BASS_NFFT
        k0 = int(math.ceil(self.BASS_LO / df))
        k1 = int(self.BASS_HI / df)
        band: np.ndarray[Any, np.dtype[Any]] = spec[k0 : k1 + 1]
        if band.size < 3:
            return 0.0
        i = int(np.argmax(band))
        k: int = k0 + i
        # амплитуда синусоиды ~ 2|X| / sum(w)
        if 2.0 * float(spec[k]) / self._bass_wsum < 2e-4:
            return 0.0
        if 1 <= k < spec.size - 1:
            a: float = math.log(max(float(spec[k - 1]), 1e-12))
            b: float = math.log(max(float(spec[k]), 1e-12))
            c: float = math.log(max(float(spec[k + 1]), 1e-12))
            den: float = a - 2.0 * b + c
            d: float = 0.5 * (a - c) / den if den < -1e-12 else 0.0
            d: float = max(-1.0, min(1.0, d))
        else:
            d = 0.0
        return (k + d) * df
