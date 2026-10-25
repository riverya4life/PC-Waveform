"""Захват системного звука (WASAPI loopback): многоканальный кольцевой буфер, Capturer."""

import backend  # noqa: F401
import threading
import numpy as np
import soundcard as sc
from pyqtgraph.Qt import QtCore
from settings import CHUNK, MAX_N, SAMPLERATE, MAX_CH

RING_N = MAX_N * 2

# Legacy (для совместимости с триггером)
ring_l = np.zeros(RING_N * 2, dtype=np.float32)
ring_r = np.zeros(RING_N * 2, dtype=np.float32)
ring_m = np.zeros(RING_N * 2, dtype=np.float32)

# Многоканальный ring
ring_ch = [np.zeros(RING_N * 2, dtype=np.float32) for _ in range(MAX_CH)]
n_channels = [2]

_ring_w = [0]
lock = threading.Lock()
state = {"device": None, "error": None, "total_samples": 0}
_ui_gate = [True]


def _ring_push(ring, w, data, n):
    end = w + n
    if end <= RING_N:
        ring[w:end] = data
        ring[w + RING_N : end + RING_N] = data
    else:
        k = RING_N - w
        m = n - k
        ring[w:RING_N] = data[:k]
        ring[w + RING_N : RING_N * 2] = data[:k]
        ring[:m] = data[k:]
        ring[RING_N : RING_N + m] = data[k:]


def audio_snapshot(need, out_m=None, out_l=None, out_r=None):
    with lock:
        e = _ring_w[0] + RING_N
        s = e - need
        if out_m is not None:
            out_m[:need] = ring_m[s:e]
        if out_l is not None:
            out_l[:need] = ring_l[s:e]
            out_r[:need] = ring_r[s:e]
        return state["total_samples"]


def audio_snapshot_multi(need: int, n_ch: int, outs: list, out_m=None) -> int:
    """Снимок всех каналов (+ опционально mid для триггера) под одним lock —
    каналы и триггерный сигнал гарантированно синхронны."""
    with lock:
        e = _ring_w[0] + RING_N
        s = e - need
        for i in range(n_ch):
            outs[i][:need] = ring_ch[i][s:e]
        if out_m is not None:
            out_m[:need] = ring_m[s:e]
        return state["total_samples"]


def set_channel_count(n: int):
    n = max(1, min(MAX_CH, int(n)))
    with lock:
        n_channels[0] = n
        _ring_w[0] = 0
        state["total_samples"] = 0
        for i in range(MAX_CH):
            ring_ch[i].fill(0.0)
        ring_l.fill(0.0)
        ring_r.fill(0.0)
        ring_m.fill(0.0)


class _AudioBridge(QtCore.QObject):
    chunk = QtCore.Signal()


_bridge = _AudioBridge()
_chunk_pending = [False]


def _notify_chunk():
    if _chunk_pending[0] or not _ui_gate[0]:
        return
    _chunk_pending[0] = True
    try:
        _bridge.chunk.emit()
    except Exception:
        _chunk_pending[0] = False


class Capturer:
    def __init__(self):
        self._thread = None
        self._stop = threading.Event()
        self._start_lock = threading.Lock()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.3)

    def start(self, device_id):
        with self._start_lock:
            self._stop.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=0.15)
            self._start_locked(device_id)

    def _start_locked(self, device_id):
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
                samplerate=SAMPLERATE, channels=MAX_CH, blocksize=CHUNK
            ) as rec:
                while not stop.is_set():
                    data = rec.record(numframes=CHUNK)
                    if stop.is_set() or data is None or data.size == 0:
                        continue
                    if data.ndim == 1:
                        data = data[:, None]
                    n = data.shape[0]
                    avail = data.shape[1]
                    if n == 0:
                        continue
                    if n > RING_N:
                        data = data[-RING_N:]
                        n = RING_N

                    # всё, что не требует общего состояния, считаем ДО lock —
                    # так поток UI не ждёт арифметику
                    left = data[:, 0].astype(np.float32, copy=False)
                    right = (
                        data[:, 1].astype(np.float32, copy=False) if avail > 1 else left
                    )
                    mid = (left + right) * 0.5

                    with lock:
                        w = _ring_w[0]
                        nc = n_channels[0]
                        _ring_push(ring_m, w, mid, n)
                        if nc == 2:
                            # L/R нужны только стерео-режиму
                            _ring_push(ring_l, w, left, n)
                            _ring_push(ring_r, w, right, n)
                        elif nc > 2:
                            # отдельные каналы нужны только многоканальному режиму
                            for i in range(nc):
                                ch_data = data[:, min(i, avail - 1)].astype(
                                    np.float32, copy=False
                                )
                                _ring_push(ring_ch[i], w, ch_data, n)
                        _ring_w[0] = (w + n) % RING_N
                        state["total_samples"] += n
                    _notify_chunk()
        except Exception as e:
            if not stop.is_set():
                state["error"] = str(e)


def list_loopback_devices():
    out = []
    try:
        for m in sc.all_microphones(include_loopback=True):
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
