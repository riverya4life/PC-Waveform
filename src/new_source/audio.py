"""Захват системного звука (WASAPI loopback): многоканальный кольцевой буфер, Capturer."""

from _thread import LockType
from typing import Any

import backend  # noqa: F401
import ctypes
import logging
import sys
import threading
import traceback
import numpy as np
import soundcard as sc
from pyqtgraph.Qt import QtCore
from settings import (
    BASS_HISTORY_SAMPLES,
    BPM_HISTORY_SAMPLES,
    CHUNK,
    CONFIG,
    MAX_N,
    SAMPLERATE,
    MAX_CH,
)

logger: logging.Logger = logging.getLogger(__name__)

MAX_RING_N: int = MAX_N * 2
RING_N: int = min(
    MAX_RING_N,
    max(
        CHUNK,
        BPM_HISTORY_SAMPLES,
        BASS_HISTORY_SAMPLES,
        2 * int(CONFIG.get("n", 4096)),
    ),
)
CAPTURE_STOP_TIMEOUT_S = 0.3

# Устаревшие буферы для совместимости с триггером.
ring_l = np.zeros(RING_N * 2, dtype=np.float32)
ring_r = np.zeros(RING_N * 2, dtype=np.float32)
ring_m = np.zeros(RING_N * 2, dtype=np.float32)

# Многоканальные кольца выделяются только для реально выбранной раскладки.
ring_ch = []
n_channels: list[int] = [2]

_ring_w: list[int] = [0]
lock: LockType = threading.Lock()
state = {
    "device": None,
    "device_id": None,
    "error": None,
    "error_type": None,
    "error_details": None,
    "error_traceback": None,
    "total_samples": 0,
}
_ui_gate: list[bool] = [True]


def _ring_push(ring, w, data, n) -> None:
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


def set_ring_window(window_size: int) -> None:
    """Настроить зеркальные аудиобуферы под окно и историю триггера."""
    global RING_N, ring_l, ring_r, ring_m

    new_size: int = min(
        MAX_RING_N,
        max(
            CHUNK,
            BPM_HISTORY_SAMPLES,
            BASS_HISTORY_SAMPLES,
            2 * int(window_size),
        ),
    )
    with lock:
        if new_size == RING_N:
            return
        RING_N = new_size
        ring_l = np.zeros(RING_N * 2, dtype=np.float32)
        ring_r = np.zeros(RING_N * 2, dtype=np.float32)
        ring_m = np.zeros(RING_N * 2, dtype=np.float32)
        ring_ch[:] = (
            [np.zeros(RING_N * 2, dtype=np.float32) for _ in range(n_channels[0])]
            if n_channels[0] > 2
            else []
        )
        _ring_w[0] = 0
        state["total_samples"] = 0


def audio_snapshot(need, out_m=None, out_l=None, out_r=None):
    with lock:
        e: int = _ring_w[0] + RING_N
        s = e - need
        if out_m is not None:
            out_m[:need] = ring_m[s:e]
        if out_l is not None:
            out_l[:need] = ring_l[s:e]
            out_r[:need] = ring_r[s:e]
        return state["total_samples"]


def audio_snapshot_multi(need: int, n_ch: int, outs: list, out_m=None) -> int:
    """Снимок всех каналов и, при необходимости, сигнала mid для триггера
    под одной блокировкой —
    каналы и триггерный сигнал гарантированно синхронны."""
    with lock:
        e: int = _ring_w[0] + RING_N
        s: int = e - need
        for i in range(n_ch):
            outs[i][:need] = ring_ch[i][s:e]
        if out_m is not None:
            out_m[:need] = ring_m[s:e]
        return state["total_samples"]


def set_channel_count(n: int) -> None:
    n = max(1, min(MAX_CH, int(n)))
    with lock:
        n_channels[0] = n
        _ring_w[0] = 0
        state["total_samples"] = 0
        ring_ch[:] = (
            [np.zeros(RING_N * 2, dtype=np.float32) for _ in range(n)] if n > 2 else []
        )
        ring_l.fill(0.0)
        ring_r.fill(0.0)
        ring_m.fill(0.0)


class _AudioBridge(QtCore.QObject):
    chunk: QtCore.pyqtSignal = QtCore.Signal()


_bridge = _AudioBridge()
_chunk_pending: list[bool] = [False]


def _notify_chunk() -> None:
    if _chunk_pending[0] or not _ui_gate[0]:
        return
    _chunk_pending[0] = True
    try:
        _bridge.chunk.emit()
    except Exception:
        _chunk_pending[0] = False


class Capturer:
    def __init__(self) -> None:
        self._thread = None
        self._stop = threading.Event()
        self._start_lock: LockType = threading.Lock()

    def stop(self) -> bool:
        self._stop.set()
        thread: None | threading.Thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=CAPTURE_STOP_TIMEOUT_S)
        stopped: bool = thread is None or not thread.is_alive()
        if not stopped:
            logger.warning("Поток захвата аудио не остановился за отведённое время")
        return stopped

    def start(self, device_id) -> bool:
        with self._start_lock:
            self._stop.set()
            thread: None | threading.Thread = self._thread
            if thread and thread.is_alive():
                thread.join(timeout=CAPTURE_STOP_TIMEOUT_S)
                if thread.is_alive():
                    message = "Предыдущий поток захвата аудио ещё работает"
                    with lock:
                        state["error"] = message
                        state["error_type"] = "CaptureError"
                        state["error_details"] = message
                        state["error_traceback"] = None
                    logger.error(message)
                    return False
            try:
                mic = sc.get_microphone(id=str(device_id), include_loopback=True)
                device_name = str(mic.name)
            except Exception as e:
                with lock:
                    state["device"] = None
                    state["device_id"] = str(device_id)
                    state["error"] = str(e)
                    state["error_type"] = type(e).__name__
                    state["error_details"] = repr(e)
                    state["error_traceback"] = traceback.format_exc()
                logger.exception("Не удалось открыть выбранное аудиоустройство")
                return False

            with lock:
                state["device"] = None
                state["device_id"] = str(device_id)
                state["error"] = None
                state["error_type"] = None
                state["error_details"] = None
                state["error_traceback"] = None
            self._start_locked(mic, device_name, device_id)
            return True

    def _start_locked(self, mic, device_name, device_id) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            args=(mic, device_name, device_id, self._stop),
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _run(mic, device_name, device_id, stop) -> None:
        com_initialized = False
        try:
            if sys.platform == "win32":
                ole32 = ctypes.WinDLL("ole32")
                ole32.CoInitializeEx.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_ulong,
                ]
                ole32.CoInitializeEx.restype = ctypes.c_long
                ole32.CoUninitialize.argtypes = []
                hr = ole32.CoInitializeEx(None, 0)
                if hr in (0, 1):
                    com_initialized = True
                elif hr != -2147417850:  # RPC_E_CHANGED_MODE
                    raise OSError(f"COM initialization failed: 0x{hr & 0xFFFFFFFF:08X}")

            with lock:
                state["device"] = device_name
                state["error"] = None
                state["error_type"] = None
                state["error_details"] = None
                state["error_traceback"] = None
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
                        n: int = RING_N

                    # Всё, что не требует общего состояния, считаем ДО блокировки —
                    # так поток UI не ждёт арифметику
                    left = data[:, 0].astype(np.float32, copy=False)
                    right = (
                        data[:, 1].astype(np.float32, copy=False) if avail > 1 else left
                    )
                    mid = (left + right) * 0.5

                    with lock:
                        w = _ring_w[0]
                        nc: int = n_channels[0]
                        _ring_push(ring_m, w, mid, n)
                        if nc == 2:
                            # Левый и правый каналы нужны только в стереорежиме.
                            _ring_push(ring_l, w, left, n)
                            _ring_push(ring_r, w, right, n)
                        elif nc > 2:
                            # Отдельные каналы нужны только в многоканальном режиме.
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
                with lock:
                    state["device"] = None
                    state["error"] = str(e)
                    state["error_type"] = type(e).__name__
                    state["error_details"] = repr(e)
                    state["error_traceback"] = traceback.format_exc()
                logger.exception("Ошибка захвата системного звука")
        finally:
            if com_initialized:
                ole32.CoUninitialize()


def list_loopback_devices():
    out = []
    try:
        for m in sc.all_microphones(include_loopback=True):
            if getattr(m, "isloopback", False):
                out.append((str(m.name), str(m.id)))
    except Exception:
        logger.warning("Не удалось перечислить loopback-устройства", exc_info=True)
    if not out:
        try:
            sp = sc.default_speaker()
            out.append((str(sp.name), str(sp.name)))
        except Exception:
            logger.warning(
                "Не удалось определить устройство вывода по умолчанию",
                exc_info=True,
            )
    return out
