"""Обновление кадров, отображение сигнала и расчёт статистики."""

from typing import Any, Literal


import math


import time

import numpy as np


from pyqtgraph.Qt import QtCore

from backend import _BACKEND_ID, _BACKEND_NAME_KEYS

from settings import (
    BPM_HISTORY_SAMPLES,
    CHUNK,
    FPS_AVG_S,
    HZ_EST_SAMPLES,
    SAMPLERATE,
    SLOW_STATS_S,
    STATS_MS,
    tr,
)


from audio import (
    _ui_gate,
    audio_snapshot,
    audio_snapshot_multi,
    state,
)

from trigger import _absmax


from dsp import Biquad


from w_support import _demo_envelope


class WindowRenderMixin:
    def _rebuild_demo_caches(self) -> None:
        n = self.n
        if n <= 0:
            self._demo_env_cache = None
            return
        t: np.ndarray[Any, np.dtype[np.floating]] = np.linspace(
            0.0, 1.0, n, endpoint=False, dtype=np.float32
        )
        self._demo_env_cache = _demo_envelope(t)

    def _apply_demo_state(self) -> None:
        if self._fade_on or self._fade_blend > 0.01:
            self._rebuild_demo_caches()
        else:
            self._demo_env_cache = None

    def _ui_hidden(self) -> bool:
        w = self.card if self._solo else self
        return w.isMinimized() or not w.isVisible()

    def _keepalive(self) -> None:
        if self._solo:
            t: float = time.perf_counter()
            if t - self._last_keepalive >= 0.1:
                self._last_keepalive: float = t
                self.plot.viewport().update()

    def _trigger_position(self, data, shift, need, n):
        if not self._smooth_stab:
            found = self.find_trigger(data, shift)
            self._hold_samples = 0
            return found

        force_search = shift <= 0 or self._crossing_smooth is None
        if force_search:
            self._trigger_pending_samples = 0
        else:
            self._trigger_pending_samples += shift

        trigger_interval: int = max(2 * CHUNK, SAMPLERATE // 60)
        if force_search or self._trigger_pending_samples >= trigger_interval:
            self._trigger_pending_samples = 0
            found = self.find_trigger(data, shift)
            if found is None:
                return self._hold_trigger(shift, need, n)
            self._hold_samples = 0
            return found

        found = self._hold_trigger(shift, need, n)
        if found is None:
            self._trigger_pending_samples = 0
            found = self.find_trigger(data, shift)
            if found is not None:
                self._hold_samples = 0
        return found

    def update_frame(self) -> None:
        if self._ui_hidden():
            if not self._was_hidden:
                QtCore.QTimer.singleShot(2500, self._trim_if_hidden)
            self._was_hidden = True
            _ui_gate[0] = False
            return

        animating = False
        if abs(self._fade_blend - self._fade_target) > 1e-4:
            animating = True
            self._fade_blend += (self._fade_target - self._fade_blend) * 0.08
            if abs(self._fade_blend - self._fade_target) < 0.002:
                self._fade_blend: float = self._fade_target
                if self._fade_blend < 0.01 and not self._fade_on:
                    self._demo_env_cache = None

        if self.frozen:
            _ui_gate[0] = False
            self._keepalive()
            return

        n = self.n
        stereo: bool = self.channels == "stereo"
        multi: bool = self._n_ch > 2
        trig_on: bool = self.trigger.isChecked()
        vb = self.plot.getViewBox()
        width_px = float(vb.width())

        sig = (
            n,
            self.lim,
            stereo,
            trig_on,
            round(self._fade_blend, 3),
            self.glow_renderer.is_enabled(),
            self.glow_renderer_r.is_enabled(),
            round(width_px, 1),
            round(vb.height(), 1),
            self._n_ch,
            self._show_gradient,
            self._show_peak_hold,
            self._band_kind,
            self._edge_clip_on,
        )

        if (
            not animating
            and not self._was_hidden
            and sig == self._frame_sig
            and state["total_samples"] == self._last_total
        ):
            self._keepalive()
            return
        self._frame_sig = sig
        if self._was_hidden:
            self._was_hidden = False
            self.prev_tpl = None
            self._crossing_smooth = None
            self._sm_bufs.clear()
        self.frames += 1

        now_t: float = time.perf_counter()
        dt: float = min(max(now_t - self._last_frame_t, 1e-3), 0.05)
        self._last_frame_t: float = now_t
        blend_a: float = 1.0 - math.exp(-dt / self.TRIG_BLEND_TAU)
        blend_on: bool = trig_on and self._smooth_stab

        if self._edge_clip_on:
            # Включено: размер амплитуды в зависимости от усиления, пики бьются об края "прям жёстко"
            clip_lim = float("inf") if blend_on else self.lim
            blend_lim = self.lim
        else:
            # Выключено: обычный режим (за основу взята бывшая включенная настройка)
            clip_lim = self.lim
            blend_lim = self.lim

        if not blend_on and self._sm_bufs:
            self._sm_bufs.clear()

        use_fade = (
            self._fade_blend > 0.01
            and self._demo_env_cache is not None
            and self._demo_env_cache.size == n
        )

        env = None
        if use_fade:
            b = float(self._fade_blend)
            env = (
                self._demo_env_cache
                if b >= 0.999
                else 1.0 + (self._demo_env_cache - 1.0) * b
            )

        ds = 1
        if width_px > 1.0:
            ds: int = max(1, int(n / (width_px * self.CURVE_DS_FACTOR)))

        if ds > 1:
            m = n // ds
            xd: np.ndarray[Any, np.dtype[Any]] = np.add(
                self._x[: m * ds : ds], 0.5 * (ds - 1), out=self._dec_x[:m]
            )
        else:
            m = n
            xd = self._x

        main_yl = None
        main_yr = None

        if multi:
            need = n * 2 if trig_on else n
            total: int = audio_snapshot_multi(
                need,
                self._n_ch,
                self._multi_bufs,
                self._scr_t if trig_on else None,
            )
            shift = total - self._last_total
            self._last_total: int = total
            start = 0
            if trig_on:
                start = n
                x = self._x
                found = self._trigger_position(self._scr_t[:need], shift, need, n)
                if found is not None:
                    start, x_off = found
                    start: int = max(0, min(start, need - n))
                    x: np.ndarray[Any, np.dtype[Any]] = np.add(
                        self._x, x_off, out=self._xs[:n]
                    )
                if ds > 1:
                    xd: np.ndarray[Any, np.dtype[Any]] = np.add(
                        x[: m * ds : ds], 0.5 * (ds - 1), out=self._dec_x[:m]
                    )
                else:
                    xd: (
                        np.ndarray[Any, np.dtype[Any]]
                        | np.ndarray[Any, np.dtype[np.floating]]
                    ) = x

            for i, curve_item in enumerate(self.multi_curves):
                raw: np.ndarray[Any, np.dtype[np.floating]] = self._multi_bufs[i][
                    start : start + n
                ]
                if env is not None:
                    np.multiply(raw, env, out=self._fade_buf[:n])
                    src: np.ndarray[Any, np.dtype[np.floating]] = self._fade_buf[:n]
                else:
                    src: np.ndarray[Any, np.dtype[np.floating]] = raw
                filt: None | Biquad = (
                    self._filters[i] if i < len(self._filters) else None
                )
                if filt is not None:
                    filt.process(src, self._filt_out[:n])
                    filtered: np.ndarray[Any, np.dtype[np.floating]] = self._filt_out[
                        :n
                    ]
                else:
                    filtered: np.ndarray[Any, np.dtype[np.floating]] = src
                np.clip(filtered, -clip_lim, clip_lim, out=self._disp_buf[:n])
                yl = self._decimate(self._disp_buf[:n], ds, m, self._dec_y)
                if blend_on:
                    yl = self._blend_frame(i, yl, blend_a, blend_lim)

                if _absmax(yl) < 1e-6:
                    curve_item.setVisible(False)
                    if i < len(self.multi_peak_curves):
                        self.multi_peak_curves[i].setVisible(False)
                else:
                    curve_item.setVisible(True)
                    curve_item.setData(xd, yl, skipFiniteCheck=True)

                    if self._show_peak_hold and i < len(self.multi_peak_curves):
                        self.multi_peak_curves[i].setVisible(True)
                        env_ph = self.multi_peak_holds[i].update(yl, dt)
                        self.multi_peak_curves[i].setData(
                            xd, env_ph, skipFiniteCheck=True
                        )
                    elif i < len(self.multi_peak_curves):
                        self.multi_peak_curves[i].setVisible(False)

                if i == 0:
                    main_yl = yl

            self.last: np.ndarray[Any, np.dtype[np.floating]] = self._multi_bufs[0][
                start : start + n
            ]
            self.last_r: np.ndarray[Any, np.dtype[np.floating]] = (
                self._multi_bufs[1][start : start + n] if self._n_ch > 1 else self.last
            )

        else:
            for ph_c in self.multi_peak_curves:
                ph_c.setVisible(False)

            n2 = n + n
            need = n2 if trig_on else n
            want_mid: bool = trig_on or not stereo
            total = audio_snapshot(
                need,
                self._scr_t if want_mid else None,
                self._scr_l if stereo else None,
                self._scr_r if stereo else None,
            )
            shift = total - self._last_total
            self._last_total = total
            data_t: np.ndarray[Any, np.dtype[np.floating]] = self._scr_t[:need]

            if trig_on:
                start = n
                x = self._x
                found = self._trigger_position(data_t, shift, need, n)
                if found is not None:
                    start, x_off = found
                    start: int = max(0, min(start, need - n))
                    x: np.ndarray[Any, np.dtype[Any]] = np.add(
                        self._x, x_off, out=self._xs[:n]
                    )
            else:
                start = 0
                x = self._x

            if ds > 1:
                xd: np.ndarray[Any, np.dtype[Any]] = np.add(
                    x[: m * ds : ds], 0.5 * (ds - 1), out=self._dec_x[:m]
                )
            else:
                xd: (
                    np.ndarray[Any, np.dtype[Any]]
                    | np.ndarray[Any, np.dtype[np.floating]]
                ) = x

            if stereo:
                raw_l = self._scr_l[start : start + n]
                raw_r = self._scr_r[start : start + n]
                if env is not None:
                    np.multiply(raw_l, env, out=self._fade_buf[:n])
                    np.multiply(raw_r, env, out=self._fade_buf_r[:n])
                    view_l: np.ndarray[Any, np.dtype[np.floating]] = self._fade_buf[:n]
                    view_r = self._fade_buf_r[:n]
                else:
                    view_l = raw_l
                    view_r = raw_r
                self.last = view_l
                self.last_r = view_r

                if self._filters[0] is not None:
                    self._filters[0].process(view_l, self._filt_out[:n])
                    np.clip(
                        self._filt_out[:n],
                        -clip_lim,
                        clip_lim,
                        out=self._disp_buf[:n],
                    )
                else:
                    np.clip(view_l, -clip_lim, clip_lim, out=self._disp_buf[:n])

                if self._filters[1] is not None:
                    self._filters[1].process(view_r, self._filt_out[:n])
                    np.clip(
                        self._filt_out[:n],
                        -clip_lim,
                        clip_lim,
                        out=self._disp_buf_r[:n],
                    )
                else:
                    np.clip(view_r, -clip_lim, clip_lim, out=self._disp_buf_r[:n])

                yl = self._decimate(self._disp_buf[:n], ds, m, self._dec_y)
                yr = self._decimate(self._disp_buf_r[:n], ds, m, self._dec_yr)
                if blend_on:
                    yl = self._blend_frame("l", yl, blend_a, blend_lim)
                    yr = self._blend_frame("r", yr, blend_a, blend_lim)
                self.curve.setData(xd, yl, skipFiniteCheck=True)
                self.curve_r.setData(xd, yr, skipFiniteCheck=True)
                self.glow_renderer.update(xd, yl, vb, self.lim)
                self.glow_renderer_r.update(xd, yr, vb, self.lim)
                main_yl = yl
                main_yr = yr
            else:
                raw: np.ndarray[Any, np.dtype[np.floating]] = data_t[start : start + n]
                if env is not None:
                    np.multiply(raw, env, out=self._fade_buf[:n])
                    view: np.ndarray[Any, np.dtype[np.floating]] = self._fade_buf[:n]
                else:
                    view: np.ndarray[Any, np.dtype[np.floating]] = raw
                self.last: np.ndarray[Any, np.dtype[np.floating]] = view
                self.last_r: np.ndarray[Any, np.dtype[np.floating]] = view

                if self._filters[0] is not None:
                    self._filters[0].process(view, self._filt_out[:n])
                    np.clip(
                        self._filt_out[:n],
                        -clip_lim,
                        clip_lim,
                        out=self._disp_buf[:n],
                    )
                else:
                    np.clip(view, -clip_lim, clip_lim, out=self._disp_buf[:n])

                yl = self._decimate(self._disp_buf[:n], ds, m, self._dec_y)
                if blend_on:
                    yl = self._blend_frame("l", yl, blend_a, blend_lim)
                self.curve.setData(xd, yl, skipFiniteCheck=True)
                if self.curve_r.isVisible():
                    self.curve_r.clear()
                self.glow_renderer.update(xd, yl, vb, self.lim)
                main_yl = yl

        if self._show_peak_hold and not multi and main_yl is not None:
            self._draw_peak(self.peak_hold_curve, self._peak_hold, main_yl, xd, dt)
            if stereo and main_yr is not None:
                self._draw_peak(
                    self.peak_hold_curve_r, self._peak_hold_r, main_yr, xd, dt
                )
            elif self.peak_hold_curve_r.isVisible():
                self.peak_hold_curve_r.setVisible(False)
        elif not self._show_peak_hold:
            if self.peak_hold_curve.isVisible():
                self.peak_hold_curve.clear()
            if self.peak_hold_curve_r.isVisible():
                self.peak_hold_curve_r.clear()

        self.update_stats()

    @staticmethod
    def _draw_peak(curve, ph, y, xd, dt) -> None:
        env = ph.update(y, dt)
        if _absmax(y) < 1e-6:
            if curve.isVisible():
                curve.setVisible(False)
            return
        if not curve.isVisible():
            curve.setVisible(True)
        curve.setData(xd, env, skipFiniteCheck=True)

    def _blend_frame(self, key, y, a, lim=None):
        """Смешивание кадров с компенсацией амплитуды.
        Восстанавливает исходный размер волны, а затем обрезает по lim,
        создавая эффект 'удара об края' (плоские вершины)."""
        buf = self._sm_bufs.get(key)
        if buf is None or buf.shape != y.shape:
            buf = self._sm_bufs[key] = np.array(y, dtype=np.float32)
            res = buf
        elif a >= 0.999:
            buf[:] = y
            res = y
        else:
            tmp: None | np.ndarray[Any, np.dtype[np.floating]] = self._sm_tmp
            if tmp is None or tmp.shape[0] < y.shape[0]:
                tmp = self._sm_tmp = np.empty(max(y.shape[0], 1024), dtype=np.float32)
            t: np.ndarray[Any, np.dtype[np.floating]] = tmp[: y.shape[0]]
            np.subtract(y, buf, out=t)
            t *= a
            buf += t
            res = buf

        if lim is None:
            return res

        out = self._sm_out.get(key)
        if out is None or out.shape != res.shape:
            out = self._sm_out[key] = np.empty(res.shape, dtype=np.float32)

        if res is y:
            out[:] = y
        else:
            pk_b: float = _absmax(res)
            pk_y: float = _absmax(y)
            if pk_b > 1e-6 and pk_y > 1e-6:
                s: float = min(max(pk_y / pk_b, 0.5), 2.0)
                ps = self._sm_scale.get(key, s)
                if s > ps:
                    ps: float = s
                else:
                    ps = ps + (s - ps) * min(1.0, a * 2.0)
                self._sm_scale[key] = ps
                np.multiply(res, ps, out=out)
            else:
                out[:] = res

        np.clip(out, -lim, lim, out=out)
        return out

    @staticmethod
    def _db(x) -> str:
        return "−∞" if x < 1e-6 else f"{20 * np.log10(x):+.1f}".replace("-", "−")

    def _hz_parts(
        self, hz: float
    ) -> tuple[Literal["0.00Hz"], None, Literal[0]] | tuple[str, str, int]:
        if hz < 1.0:
            return "0.00Hz", None, 0
        txt: str = f"{hz:.2f}Hz" if hz < 1000.0 else f"{hz / 1000.0:.2f}kHz"
        midi: float = 69.0 + 12.0 * math.log2(hz / 440.0)
        n = int(round(midi))
        cents = int((midi - n) * 100.0)
        return txt, f"{self._NOTE_NAMES[n % 12]}{n // 12 - 1}", cents

    def _format_hz(self, hz: float) -> str:
        return self._hz_parts(hz)[0]

    def _format_hz_plain(self, hz: float) -> str:
        txt, name, cents = self._hz_parts(hz)
        if name is None:
            return txt
        sign: str = "-" if cents < 0 else ""
        return f"{txt} | {name} - {sign}{abs(cents)} Cents"

    def _try_reconnect(self) -> None:
        self._retry_pending = False
        if not state["error"]:
            return
        try:
            self._on_device_change(self.device_combo.currentIndex())
        except Exception:
            pass

    def _stats_fallback(self) -> None:
        if (not _ui_gate[0]) and not self.frozen and not self._ui_hidden():
            _ui_gate[0] = True
        if state["error"] and not self._retry_pending:
            self._retry_pending = True
            QtCore.QTimer.singleShot(2000, self._try_reconnect)
        idle: Any | float = getattr(self, "_stats_idle_s", STATS_MS / 1000.0)
        if time.perf_counter() - getattr(self, "_stats_t", 0.0) > idle:
            self.update_stats()

    def _feed_bpm(self) -> None:
        total_s = state["total_samples"]
        new_n = total_s - self._bpm_last_total
        if new_n <= 0:
            return
        if new_n > BPM_HISTORY_SAMPLES:
            new_n: int = BPM_HISTORY_SAMPLES
        audio_snapshot(new_n, out_m=self._bpm_scr[:new_n])
        self._bpm.feed(self._bpm_scr[:new_n])
        self._bpm_last_total = total_s

    def update_stats(self) -> None:
        now: float = time.perf_counter()
        if now - getattr(self, "_stats_t", 0.0) < 1.0 / 30.0:
            return
        self._stats_t: float = now
        if self._ui_hidden():
            self.frames = 0
            self.t0: float = now
            return

        if self._bpm_on:
            self._feed_bpm()
            if self._bpm.maybe_update():
                bpm_val: str = (
                    f"{self._bpm.bpm:.0f}"
                    if (self._bpm.bpm > 30.0 and self._bpm.confidence > 0.1)
                    else "—"
                )
                self.bpm_tile.set(tr("bpm"), bpm_val)

        total_s = state["total_samples"]
        if (
            not self.frozen
            and total_s - getattr(self, "_metrics_total", -(10**9)) >= HZ_EST_SAMPLES
        ):
            self._metrics_total = total_s
            seg = self.last
            if self.channels == "stereo" and len(self.last_r):
                if len(seg):
                    self._peak_l_val: float = _absmax(seg)
                    self._peak_r_val: float = _absmax(self.last_r)
                    self._peak_val: float = max(self._peak_l_val, self._peak_r_val)
                    self._rms_val: float = math.sqrt(
                        0.5
                        * (
                            float(np.dot(seg, seg)) / len(seg)
                            + float(np.dot(self.last_r, self.last_r)) / len(self.last_r)
                        )
                    )
                else:
                    self._peak_val = self._peak_l_val = 0.0
                    self._peak_r_val = self._rms_val = 0.0
            else:
                if len(seg):
                    self._peak_val: float = _absmax(seg)
                    self._rms_val: float = math.sqrt(float(np.dot(seg, seg)) / len(seg))
                else:
                    self._peak_val = self._rms_val = 0.0

            hz_raw: float = self._estimate_bass_hz()
            if hz_raw > 1.0:
                self._hz_silent_since = None
                if self._hz_smooth < 1.0:
                    self._hz_smooth: float = hz_raw
                else:
                    rel: float = abs(hz_raw - self._hz_smooth) / max(
                        self._hz_smooth, 1.0
                    )
                    k: float = 0.7 if rel > 0.08 else (0.4 if rel > 0.01 else 0.25)
                    self._hz_smooth: float = self._hz_smooth * (1.0 - k) + hz_raw * k
            else:
                t_s: Any | None = getattr(self, "_hz_silent_since", None)
                if t_s is None:
                    self._hz_silent_since: float = now
                elif now - t_s > 0.6:
                    self._hz_smooth = 0.0

        dim = '<span style="color:#6e7681">'
        peak: float = self._peak_val
        rms: float = self._rms_val
        if self.channels == "stereo":
            peak_l: float = self._peak_l_val
            peak_r: float = self._peak_r_val
            clip_l: bool = peak_l >= 0.999
            clip_r: bool = peak_r >= 0.999
            col_l: str = "#f85149" if clip_l else self.accent
            col_r: str = "#f85149" if clip_r else self._stereo_r_hex
            mark_l: str = f" {tr('clip')}" if clip_l else ""
            mark_r: str = f" {tr('clip')}" if clip_r else ""
            self.peak_tile.set(
                tr("peak") + " · dBFS",
                f'<span style="color:{col_l}">L {self._db(peak_l)}{mark_l}</span>'
                f"{dim} · </span>"
                f'<span style="color:{col_r}">R {self._db(peak_r)}{mark_r}</span>',
            )
            self.rms_tile.set(tr("rms"), f"{self._db(rms)} dBFS")
        else:
            peak_color: str = "#f85149" if peak >= 0.999 else self.accent
            clip: str = f"  {tr('clip')}" if peak >= 0.999 else ""
            self.peak_tile.set(
                tr("peak"),
                f'<span style="color:{peak_color}">{self._db(peak)} dBFS{clip}</span>',
            )
            self.rms_tile.set(tr("rms"), f"{self._db(rms)} dBFS")

        hz_txt: str = self._format_hz(self._hz_smooth)
        if self._solo:
            vals: tuple[str, str, str, str] = (
                self._db(peak),
                self._db(rms),
                self._format_hz_plain(self._hz_smooth),
                self._hz_parts(self._hz_smooth)[0],
            )
            if vals != self._solo_vals:
                self._solo_vals: tuple[str, str, str, str] = vals
                if self._hud_on:
                    self._fit_hud()
        self.hz_tile.set(tr("hz_label"), hz_txt)
        if self.hz_tile.toolTip() != tr("hz_tip"):
            self.hz_tile.setToolTip(tr("hz_tip"))

        dt: float = now - self.t0
        if dt >= FPS_AVG_S:
            self._fps_measured: float = self.frames / max(dt, 1e-3)
            self.frames = 0
            self.t0: float = now
        fps: Any | float = getattr(self, "_fps_measured", 0.0)
        self.fps_tile.set(
            tr("fps_label"), tr("paused") if self.frozen else f"{fps:.0f}"
        )

        if now - getattr(self, "_slow_t", 0.0) < SLOW_STATS_S:
            return
        self._slow_t: float = now
        self.info_tile.set(
            tr("window_label"),
            f'{self.n} {tr("samples")} {dim}·</span> '
            f'{self.n / SAMPLERATE * 1000:.1f} {tr("ms")}',
        )
        be_name: str = tr(_BACKEND_NAME_KEYS.get(_BACKEND_ID, "backend_raster"))
        self.bknd_tile.set(tr("backend_label"), be_name)
        if self.bknd_tile.toolTip() != tr("backend_tip"):
            self.bknd_tile.setToolTip(tr("backend_tip"))
        has_err = bool(state["error"])
        if has_err != self.device_err:
            self.device_err: bool = has_err
            self.device_lbl.setStyleSheet("color: #f85149;" if has_err else "")
            self.dot.setStyleSheet(
                "background: #f85149; border-radius: 4px;" if has_err else ""
            )
        if has_err:
            full: str = f'{tr("error_prefix")} {state["error"]}'
            self.pill.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            self.device_lbl.setToolTip(f'{full}\n{tr("audio_error_click_details")}')
            self.pill.setToolTip(self.device_lbl.toolTip())
        else:
            name = state["device"] or tr("searching")
            full: str = f'{name}  ·  {SAMPLERATE / 1000:g} {tr("khz")}'
            self.pill.unsetCursor()
            self.device_lbl.setToolTip(full)
            self.pill.setToolTip(full)
        elided: str = self.device_lbl.fontMetrics().elidedText(
            full,
            QtCore.Qt.TextElideMode.ElideRight,
            max(self.device_lbl.width() - 4, 20),
        )
        if self.device_lbl.text() != elided:
            self.device_lbl.setText(elided)
