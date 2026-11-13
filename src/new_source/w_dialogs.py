"""Обработка темы, диалогов настроек и параметров окна."""

from typing import Any


import sys


import numpy as np

import pyqtgraph as pg

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from animations import (
    BUTTON_FEEDBACK_DURATION_MS,
    HOVER_EASING_CURVE,
    CascadeReveal,
    fade_in_when_shown,
)

from settings import (
    CHANNEL_LAYOUTS,
    CH_LAYOUT_TO_N,
    CHUNK,
    CONFIG,
    DEFAULT_CONFIG,
    MAX_CH,
    SAMPLERATE,
    SAMPLE_OPTIONS,
    TICKS,
    _save_config,
    cap_text,
    tr,
)

from theme import (
    DEFAULT_THEME,
    Swatch,
    THEMES,
    THEME_HEX,
    _make_icon,
    _mix_dark,
    _rgba,
    _stereo_r_color,
    system_accent,
    make_qss,
)

from audio import (
    _ui_gate,
    set_channel_count,
    set_ring_window,
    state,
)


from widgets import (
    GradientCurveItem,
    make_gradient_pen,
)

from dsp import Biquad, make_filter

from analysis import PeakHold


class _AnimatedOutline(QtWidgets.QWidget):
    def __init__(self, target: QtWidgets.QWidget, accent: QtGui.QColor) -> None:
        super().__init__(target)
        self._target = target
        self._color = QtGui.QColor("#30363d")
        self._accent = QtGui.QColor(accent)
        self._hovered = False
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(BUTTON_FEEDBACK_DURATION_MS)
        self._anim.setEasingCurve(HOVER_EASING_CURVE)
        self._anim.valueChanged.connect(self._set_color)
        target.installEventFilter(self)
        self.setGeometry(target.rect())
        self.show()
        self.raise_()

    def set_accent(self, accent: QtGui.QColor) -> None:
        self._accent = QtGui.QColor(accent)
        if self._hovered:
            self._animate(True)

    def _set_color(self, color: QtGui.QColor) -> None:
        self._color = QtGui.QColor(color)
        self.update()

    def _animate(self, hovered: bool) -> None:
        self._hovered = hovered
        self._anim.stop()
        self._anim.setStartValue(self._color)
        self._anim.setEndValue(self._accent if hovered else QtGui.QColor("#30363d"))
        self._anim.start()
        self.raise_()

    def eventFilter(self, obj, ev) -> bool:
        if obj is not self._target:
            return False
        event_type = ev.type()
        if event_type in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Show):
            self.setGeometry(self._target.rect())
            self.raise_()
        elif event_type == QtCore.QEvent.Type.Enter:
            self._animate(True)
        elif event_type == QtCore.QEvent.Type.Leave:
            self._animate(False)
        return False

    def paintEvent(self, ev) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(self._color, 1.2))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 9, 9)


class WindowDialogsMixin:
    THEME_ACCENT_ANIMATION_MS = 140

    def _show_audio_error(self) -> None:
        error = state.get("error")
        if not error:
            return

        device_name = state.get("device") or self.device_combo.currentText()
        device_id = state.get("device_id")
        error_type = state.get("error_type") or tr("audio_error_unknown_type")
        details: list[str] = [
            f'{tr("audio_error_device")}: {device_name or tr("searching")}',
            f'{tr("audio_error_device_id")}: {device_id or "—"}',
            f'{tr("audio_error_sample_rate")}: {SAMPLERATE} Hz',
            f'{tr("audio_error_block_size")}: {CHUNK}',
            f'{tr("audio_error_channels")}: {MAX_CH}',
            f'{tr("audio_error_type")}: {error_type}',
            f'{tr("audio_error_message")}: {error}',
        ]
        extra_details = state.get("error_details")
        if extra_details and extra_details != error:
            details.append(f'{tr("audio_error_raw")}: {extra_details}')

        dialog = QtWidgets.QDialog(self)
        dialog.setObjectName("audioErrorDialog")
        dialog.setWindowTitle(tr("audio_error_title"))
        dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        dialog.setMinimumWidth(360)
        dialog.setMaximumWidth(480)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)

        heading = QtWidgets.QHBoxLayout()
        heading.setSpacing(12)
        icon = QtWidgets.QLabel()
        icon.setPixmap(
            self.style()
            .standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MessageBoxCritical)
            .pixmap(28, 28)
        )
        heading.addWidget(icon, 0, QtCore.Qt.AlignmentFlag.AlignTop)

        text_col = QtWidgets.QVBoxLayout()
        text_col.setSpacing(5)
        title = QtWidgets.QLabel(tr("audio_error_title"))
        title.setStyleSheet("color: #f0f6fc; font-size: 15px; font-weight: 600;")
        error_label = QtWidgets.QLabel(str(error))
        error_label.setWordWrap(True)
        error_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        hint = QtWidgets.QLabel(tr("audio_error_hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8b949e;")
        text_col.addWidget(title)
        text_col.addWidget(error_label)
        text_col.addWidget(hint)
        heading.addLayout(text_col, 1)
        layout.addLayout(heading)

        details_view = QtWidgets.QPlainTextEdit()
        details_view.setReadOnly(True)
        details_view.setPlainText("\n".join(details))
        details_view.setFixedHeight(190)
        details_view.hide()
        layout.addWidget(details_view)

        def _toggle_details(visible) -> None:
            details_view.setVisible(visible)
            details_button.setText(
                tr("audio_error_hide_details")
                if visible
                else tr("audio_error_show_details")
            )
            QtCore.QTimer.singleShot(0, dialog.adjustSize)

        buttons = QtWidgets.QHBoxLayout()
        details_button = QtWidgets.QPushButton(tr("audio_error_show_details"))
        details_button.setCheckable(True)
        details_button.toggled.connect(_toggle_details)
        close_button = QtWidgets.QPushButton(tr("close"))
        close_button.setObjectName("errorCloseButton")
        close_button.setDefault(True)
        close_button.clicked.connect(dialog.accept)
        buttons.addWidget(details_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)
        top = _mix_dark(self.accent, 0.040)
        bottom = _mix_dark(self.accent, 0.014)
        dialog.setStyleSheet(f"""
            QDialog#audioErrorDialog {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 {top}, stop: 1 {bottom}
                );
                border: none;
            }}
            QLabel {{ color: #c9d1d9; background: transparent; }}
            QPlainTextEdit {{
                background: #0d1117;
                color: #c9d1d9;
                border: 1px solid {_rgba(self.accent, 0.35)};
                border-radius: 6px;
                padding: 8px;
                font-family: 'Cascadia Mono', 'Consolas', monospace;
                font-size: 10px;
            }}
            QPushButton {{
                color: #c9d1d9;
                background: rgba(255, 255, 255, 10);
                border: 1px solid rgba(255, 255, 255, 26);
                border-radius: 6px;
                padding: 6px 12px;
                min-height: 22px;
            }}
            QPushButton#errorCloseButton {{
                color: #0d1117;
                background: {self.accent};
                border-color: {self.accent};
                font-weight: 600;
            }}
            """)
        dialog.ensurePolished()
        dialog.adjustSize()
        if sys.platform == "win32":
            # Windows увеличивает минимальную высоту диалога на 16 пикселей.
            dialog.setMinimumHeight(dialog.height() + 16)
        fade_in_when_shown(dialog)
        dialog.exec()

    def _update_app_icon(self, color=None) -> None:
        icon: QtGui.QIcon = _make_icon(color or self.accent)
        self.setWindowIcon(icon)
        self.card.setWindowIcon(icon)
        app: QtCore.QCoreApplication | None = QtWidgets.QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)

    def _rebuild_pens(self) -> None:
        c = QtGui.QColor(self._current_color)
        c.setAlpha(255)
        quiet = QtGui.QColor(c)
        quiet.setAlpha(40)
        loud: QtGui.QColor = QtGui.QColor(c).lighter(120)
        self._solid_pen = pg.mkPen(c, width=1.7)
        self._gradient_pen: QtGui.QPen = make_gradient_pen(quiet, loud, 1.7)
        self.curve.setPen(
            self._gradient_pen if self._show_gradient else self._solid_pen
        )

        c2: QtGui.QColor = _stereo_r_color(c.name())
        self._stereo_r_hex: str = c2.name()
        quiet2 = QtGui.QColor(c2)
        quiet2.setAlpha(40)
        loud2: QtGui.QColor = QtGui.QColor(c2).lighter(120)
        self.curve_r.setPen(
            make_gradient_pen(quiet2, loud2, 1.5)
            if self._show_gradient
            else pg.mkPen(c2, width=1.5)
        )

        if hasattr(self, "peak_hold_curve"):
            ph_l = QtGui.QColor(c2)
            ph_l.setAlpha(140)
            self.peak_hold_curve.setPen(pg.mkPen(ph_l, width=1.0))
            ph_r = QtGui.QColor(c2)
            ph_r.setAlpha(140)
            self.peak_hold_curve_r.setPen(pg.mkPen(ph_r, width=1.0))

        # Многоканальные кривые
        h, s, v, a = c.getHsv()
        for i, mc in enumerate(self.multi_curves):
            factor: float = 1.0 - (i / max(self._n_ch - 1, 1)) * 0.6
            v_mod = int(v * factor)
            col: QtGui.QColor = QtGui.QColor.fromHsv(h, s, max(40, v_mod))

            if self._show_gradient:
                q = QtGui.QColor(col)
                q.setAlpha(60)
                lo: QtGui.QColor = QtGui.QColor(col).lighter(130)
                mc.setPen(make_gradient_pen(q, lo, 1.3))
            else:
                mc.setPen(pg.mkPen(col, width=1.3))

            # Пик-холд для этого канала берет тот же цвет, что и в стерео/моно
            if i < len(self.multi_peak_curves):
                ph_col = QtGui.QColor(c2)
                ph_col.setAlpha(140)
                self.multi_peak_curves[i].setPen(pg.mkPen(ph_col, width=1.0))

    def _apply_theme_color(self, color: QtGui.QColor) -> None:
        c = QtGui.QColor(color)
        c.setAlpha(255)
        self._current_color: QtGui.QColor = c
        self._rebuild_pens()
        self.glow_renderer.set_color(c)
        c2: QtGui.QColor = _stereo_r_color(c.name())
        self.glow_renderer_r.set_color(c2)
        self._set_theme_accent_widgets(c)
        self._update_freeze_icon()

        if self._solo:
            self.card.setStyleSheet(self._solo_card_qss())
            self._style_hud()

    def _set_theme_accent_widgets(self, color: QtGui.QColor) -> None:
        self.logo.set_accent(color)
        self.trigger.set_accent(color)
        self.glow.set_accent(color)
        if hasattr(self, "fade_sw"):
            self.fade_sw.set_accent(color)
        for attr in (
            "gradient_sw",
            "peak_hold_sw",
            "bpm_sw",
            "smooth_sw",
            "edge_clip_sw",
        ):
            w: Any | None = getattr(self, attr, None)
            if w is not None:
                w.set_accent(color)

    def _animate_theme_accent_widgets(
        self, start: QtGui.QColor, target: QtGui.QColor
    ) -> None:
        if (
            self._theme_anim is not None
            and self._theme_anim.state() == QtCore.QAbstractAnimation.State.Running
        ):
            start = QtGui.QColor(self._theme_anim.currentValue())
            self._theme_anim.stop()
        if start == target:
            return
        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(self.THEME_ACCENT_ANIMATION_MS)
        anim.setStartValue(start)
        anim.setEndValue(target)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.valueChanged.connect(self._set_theme_accent_widgets)
        anim.finished.connect(lambda: self._set_theme_accent_widgets(target))
        anim.start()
        self._theme_anim = anim

    def _on_theme_color(self, color: QtGui.QColor) -> None:
        if not isinstance(color, QtGui.QColor):
            return
        self.accent: str = color.name()
        self.setStyleSheet(make_qss(self.accent))
        self._apply_theme_color(color)
        self.plot.setBackground(self._plot_bg())
        for outline in getattr(self, "_settings_outlines", []):
            outline.set_accent(color)
        if getattr(self, "gain_combo", None) is not None:
            self.gain_combo.set_accent(color)
        prev: Any | None = getattr(self, "_theme_preview", None)
        if prev is not None:
            prev.setStyleSheet(f"background: {self.accent}; border-radius: 6px;")
        dlg: Any | None = getattr(self, "_theme_dialog", None)
        if dlg is not None and dlg.isVisible():
            dlg.setStyleSheet(self._theme_dialog_qss())
        self._update_app_icon(color)

    def _theme_color(self, theme_id: str) -> str:
        if theme_id == "system":
            c: str = system_accent() or self._sys_last or THEME_HEX[DEFAULT_THEME]
            self._sys_last: str = c
            return c
        if theme_id == "custom":
            return str(CONFIG.get("custom_color", DEFAULT_CONFIG["custom_color"]))
        return THEME_HEX.get(theme_id, THEME_HEX[DEFAULT_THEME])

    def _sync_system_watch(self) -> None:
        if self._sys_timer is None:
            tm = QtCore.QTimer(self)
            tm.setTimerType(QtCore.Qt.TimerType.VeryCoarseTimer)
            tm.setInterval(self.SYS_POLL_MS)
            tm.timeout.connect(self._poll_system_accent)
            self._sys_timer: QtCore.QTimer = tm
        if self.theme_id == "system":
            if not self._sys_timer.isActive():
                self._sys_timer.start()
        else:
            self._sys_timer.stop()

    def _poll_system_accent(self) -> None:
        if self.theme_id != "system":
            return
        new: str = self._theme_color("system")
        if new == self._sys_applied:
            return
        start = QtGui.QColor(self._current_color)
        self._sys_applied: str = new
        self._on_theme_color(QtGui.QColor(new))
        self._animate_theme_accent_widgets(start, QtGui.QColor(new))
        if getattr(self, "_theme_dialog", None):
            self._refresh_theme_dialog_texts()

    def _apply_theme(self, theme_id: str) -> None:
        btns: Any | None = getattr(self, "theme_btns", None)
        if btns and theme_id in btns:
            btns[theme_id].setChecked(True)
        self.theme_id: str = theme_id
        target: str = self._theme_color(theme_id)
        start = QtGui.QColor(self._current_color)
        self._sys_applied: str | None = target if theme_id == "system" else None
        self._sync_system_watch()
        self._on_theme_color(QtGui.QColor(target))
        if not getattr(self, "_loading", False):
            self._animate_theme_accent_widgets(start, QtGui.QColor(target))
        if getattr(self, "_theme_dialog", None) and self._theme_dialog.isVisible():
            self._refresh_theme_dialog_texts()
        if not getattr(self, "_loading", False):
            CONFIG["theme"] = theme_id
            _save_config(CONFIG)

    @staticmethod
    def _style_combo_popup(combo) -> None:
        view = QtWidgets.QListView()
        view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        combo.setView(view)
        try:
            popup = combo.view().window()
            if popup is None:
                return
            popup.setWindowFlags(
                popup.windowFlags()
                | QtCore.Qt.WindowType.FramelessWindowHint
                | QtCore.Qt.WindowType.NoDropShadowWindowHint
            )
            popup.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        except Exception:
            pass

    def _caption_holder(
        self, caption_key, widget
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QLabel]:
        cap = QtWidgets.QLabel(cap_text(caption_key))
        cap.setObjectName("cap")
        cap.setFixedHeight(12)
        holder = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addWidget(cap)
        box.addWidget(widget)
        return holder, cap

    def _settings_dialog_qss(self) -> str:
        a: str = self.accent
        top: str = _mix_dark(a, 0.040)
        bot: str = _mix_dark(a, 0.014)
        return f"""
            QDialog {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {top}, stop:1 {bot}); }}
            QLabel {{ background: transparent; color: #c9d1d9; }}
            QLabel#stTitle {{ color: #f0f6fc; font-size: 16px; font-weight: 700; }}
            QLabel#stCap {{ color: #6e7681; font-size: 10px; font-weight: 600; letter-spacing: 1px; padding: 2px 4px 6px 4px; }}
            QLabel#stRowLabel {{ color: #e6edf3; font-size: 13px; font-weight: 500; }}
            QFrame#stCard {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; }}
            QFrame#stSep {{ background: rgba(255,255,255,0.06); max-height: 1px; min-height: 1px; border: none; }}
            QPushButton#stDone {{ background: {a}; color: #0d1117; border: none; border-radius: 10px; padding: 10px 28px; font-weight: 700; font-size: 13px; }}
            QComboBox {{ background: rgba(255,255,255,0.08); color: #c9d1d9; border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; padding: 6px 12px; min-width: 120px; min-height: 28px; }}
            QComboBox::drop-down {{ border: none; width: 22px; }}
            QComboBox QAbstractItemView {{ background: #161b22; color: #c9d1d9; border: 1px solid #30363d; border-radius: 10px; selection-background-color: {a}; selection-color: #0d1117; }}
            QFrame#seg {{ background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; }}
            QPushButton#segbtn {{ background: transparent; color: #8b949e; border: none; border-radius: 8px; padding: 6px 12px; font-weight: 500; font-family: 'Segoe UI'; font-size: 12px; min-width: 0px; }}
            QPushButton#segbtn:checked {{ background: {a}; color: #0d1117; font-weight: 700; }}
        """

    def _settings_row(self, label_key, widget, store_cap=None) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(14, 10, 14, 10)
        h.setSpacing(12)
        lbl = QtWidgets.QLabel(tr(label_key))
        lbl.setObjectName("stRowLabel")
        h.addWidget(lbl, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        h.addStretch(1)
        h.addWidget(widget, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        if store_cap is not None:
            setattr(self, store_cap, lbl)
        return row

    def _settings_sep(self) -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setObjectName("stSep")
        line.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        return line

    def _build_settings_dialog(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(tr("settings"))
        dlg.setModal(True)
        dlg.setFixedWidth(820)
        self._settings_sections: list[tuple[QtWidgets.QLabel, QtWidgets.QFrame]] = []
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(24, 22, 24, 20)
        lay.setSpacing(14)

        title = QtWidgets.QLabel(tr("settings"))
        title.setObjectName("stTitle")
        lay.addWidget(title)

        def _section(
            title_key,
        ) -> tuple[QtWidgets.QVBoxLayout, QtWidgets.QVBoxLayout, QtWidgets.QLabel]:
            wrap = QtWidgets.QVBoxLayout()
            wrap.setSpacing(6)
            cap = QtWidgets.QLabel(tr(title_key))
            cap.setObjectName("stCap")
            wrap.addWidget(cap)
            card = QtWidgets.QFrame()
            card.setObjectName("stCard")
            card_l = QtWidgets.QVBoxLayout(card)
            card_l.setContentsMargins(0, 4, 0, 4)
            card_l.setSpacing(0)
            wrap.addWidget(card)
            self._settings_sections.append((cap, card))
            return wrap, card_l, cap

        w1, c1, self._st_cap_general = _section("settings_general")
        c1.addWidget(self._settings_row("language", self.lang_combo, "cap_lang"))
        c1.addWidget(self._settings_sep())
        c1.addWidget(
            self._settings_row("ch_layout", self.ch_layout_combo, "cap_channels")
        )
        c1.addWidget(self._settings_sep())
        c1.addWidget(self._settings_row("fps", self.fps_combo, "cap_fps"))

        w2, c2, self._st_cap_signal = _section("settings_signal")
        c2.addWidget(self._settings_row("window", self.n_combo, "cap_window"))
        c2.addWidget(self._settings_sep())
        c2.addWidget(self._settings_row("gain", self.gain_combo, "cap_gain"))
        c2.addWidget(self._settings_sep())
        c2.addWidget(self._settings_row("bandpass", self.band_combo))

        w3, c3, self._st_cap_effects = _section("settings_effects")
        c3.addWidget(self._settings_row("stabilization", self.trigger, "_cap_stab"))
        c3.addWidget(self._settings_sep())
        c3.addWidget(self._settings_row("smooth_stab", self.smooth_sw))
        c3.addWidget(self._settings_sep())
        c3.addWidget(self._settings_row("edge_clip", self.edge_clip_sw, "_cap_edge"))
        c3.addWidget(self._settings_sep())
        c3.addWidget(self._settings_row("glow", self.glow, "_cap_glow"))
        c3.addWidget(self._settings_sep())
        c3.addWidget(self._settings_row("fade", self.fade_sw, "_cap_fade"))
        c3.addWidget(self._settings_sep())
        c3.addWidget(self._settings_row("gradient", self.gradient_sw))
        c3.addWidget(self._settings_sep())
        c3.addWidget(self._settings_row("peak_hold", self.peak_hold_sw))
        c3.addWidget(self._settings_sep())
        c3.addWidget(self._settings_row("bpm", self.bpm_sw))

        cols = QtWidgets.QHBoxLayout()
        cols.setSpacing(24)

        col_left = QtWidgets.QVBoxLayout()
        col_left.setSpacing(14)
        col_left.addLayout(w1)
        col_left.addLayout(w2)
        col_left.addStretch(1)

        col_right = QtWidgets.QVBoxLayout()
        col_right.setSpacing(14)
        col_right.addLayout(w3)
        col_right.addStretch(1)

        cols.addLayout(col_left, 1)
        cols.addLayout(col_right, 1)
        lay.addLayout(cols)

        for w in (
            self.lang_combo,
            self.n_combo,
            self.gain_combo,
            self.ch_layout_combo,
            self.fps_combo,
            self.trigger,
            self.glow,
            self.fade_sw,
            self.band_combo,
            self.gradient_sw,
            self.peak_hold_sw,
            self.bpm_sw,
            self.smooth_sw,
            self.edge_clip_sw,
        ):
            w.setVisible(True)
            w.show()

        for w in (
            self.lang_combo,
            self.n_combo,
            self.fps_combo,
            self.band_combo,
        ):
            w.setFixedWidth(190)

        self.ch_layout_combo.setMinimumWidth(190)
        self.gain_combo.setMinimumWidth(210)

        for sw in (
            self.trigger,
            self.glow,
            self.fade_sw,
            self.gradient_sw,
            self.peak_hold_sw,
            self.bpm_sw,
            self.smooth_sw,
            self.edge_clip_sw,
        ):
            sw.setText("")
            sw.setMinimumWidth(sw.TW + 4)
            sw.setFixedHeight(26)

        close_btn = QtWidgets.QPushButton(tr("theme_close"))
        close_btn.setObjectName("stDone")
        close_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)

        self._settings_outlines = [
            _AnimatedOutline(widget, QtGui.QColor(self.accent))
            for widget in (
                self.lang_combo,
                self.ch_layout_combo,
                self.fps_combo,
                self.n_combo,
                self.band_combo,
                close_btn,
            )
        ]
        dlg.setStyleSheet(self._settings_dialog_qss())
        self.gain_combo.set_animated_selection(True, QtGui.QColor(self.accent))
        self._settings_dialog: QtWidgets.QDialog = dlg
        self._settings_dlg_title: QtWidgets.QLabel = title
        self._settings_dlg_close: QtWidgets.QPushButton = close_btn
        self._settings_cascade = CascadeReveal(dlg)

    def _open_settings_dialog(self) -> None:
        if not getattr(self, "_settings_dialog", None):
            self._build_settings_dialog()
        for w in (
            self.lang_combo,
            self.n_combo,
            self.gain_combo,
            self.ch_layout_combo,
            self.fps_combo,
            self.trigger,
            self.glow,
            self.fade_sw,
            self.band_combo,
            self.gradient_sw,
            self.peak_hold_sw,
            self.bpm_sw,
            self.smooth_sw,
            self.edge_clip_sw,
        ):
            w.setVisible(True)
            w.show()
        self._settings_dialog.setWindowTitle(tr("settings"))
        self._settings_dlg_title.setText(tr("settings"))
        self._settings_dlg_close.setText(tr("theme_close"))
        if getattr(self, "_st_cap_general", None):
            self._st_cap_general.setText(tr("settings_general"))
        if getattr(self, "_st_cap_signal", None):
            self._st_cap_signal.setText(tr("settings_signal"))
        if getattr(self, "_st_cap_effects", None):
            self._st_cap_effects.setText(tr("settings_effects"))
        if self.cap_lang:
            self.cap_lang.setText(tr("language"))
        if self.cap_channels:
            self.cap_channels.setText(tr("ch_layout"))
        if self.cap_window:
            self.cap_window.setText(tr("window"))
        if self.cap_gain:
            self.cap_gain.setText(tr("gain"))
        if getattr(self, "_cap_stab", None):
            self._cap_stab.setText(tr("stabilization"))
        if getattr(self, "_cap_glow", None):
            self._cap_glow.setText(tr("glow"))
        if getattr(self, "_cap_fade", None):
            self._cap_fade.setText(tr("fade"))
        if getattr(self, "_cap_edge", None):
            self._cap_edge.setText(tr("edge_clip"))
        for sw in (
            self.trigger,
            self.glow,
            self.fade_sw,
            self.gradient_sw,
            self.peak_hold_sw,
            self.bpm_sw,
            self.smooth_sw,
            self.edge_clip_sw,
        ):
            sw.setText("")
        self.trigger.setToolTip(tr("stabilization_tip"))
        self.fade_sw.setToolTip(tr("fade_tip"))
        self.edge_clip_sw.setToolTip(tr("edge_clip_tip"))
        self.band_combo.setToolTip(tr("bandpass_tip"))
        self.bpm_tile.setToolTip(tr("bpm_tip"))
        self.band_combo.setItemText(0, tr("band_off"))
        self.band_combo.setItemText(1, tr("band_low"))
        self.band_combo.setItemText(2, tr("band_mid"))
        self.band_combo.setItemText(3, tr("band_high"))

        if self.ch_layout_combo:
            for i, (_key, _n, label) in enumerate(CHANNEL_LAYOUTS):
                self.ch_layout_combo.setItemText(i, tr(label))

        for sw in (
            self.trigger,
            self.glow,
            self.fade_sw,
            self.gradient_sw,
            self.peak_hold_sw,
            self.bpm_sw,
            self.smooth_sw,
            self.edge_clip_sw,
        ):
            sw.set_accent(self.accent)
        idx: int = self.n_combo.currentIndex()
        self.n_combo.blockSignals(True)
        self.n_combo.clear()
        self.n_combo.addItems([f"{v} {tr('samples')}" for v in SAMPLE_OPTIONS])
        self.n_combo.setCurrentIndex(idx)
        self.n_combo.blockSignals(False)
        self._refresh_fps_combo()
        if self.cap_fps:
            self.cap_fps.setText(tr("fps"))
        self._settings_dialog.setStyleSheet(self._settings_dialog_qss())
        self._settings_dialog.adjustSize()
        self._settings_dialog.setMinimumHeight(self._settings_dialog.height())
        self._settings_cascade.prepare(
            [self._settings_dlg_title]
            + [[cap, card] for cap, card in self._settings_sections]
            + [self._settings_dlg_close]
        )
        self._settings_cascade.play()
        fade_in_when_shown(self._settings_dialog)
        self._settings_dialog.exec()

    def _theme_dialog_qss(self) -> str:
        a: str = self.accent
        top: str = _mix_dark(a, 0.040)
        bot: str = _mix_dark(a, 0.014)
        return f"""
            QDialog {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {top}, stop:1 {bot}); }}
            QLabel {{ background: transparent; color: #c9d1d9; }}
            QLabel#thTitle {{ color: #f0f6fc; font-size: 15px; font-weight: 700; }}
            QLabel#thCap {{ color: #6e7681; font-size: 10px; font-weight: 600; letter-spacing: 1px; }}
            QLabel#thSub {{ color: #8b949e; font-size: 12px; }}
            QFrame#thCard {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; }}
            QPushButton#thCustom {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.10); border-radius: 12px; color: #c9d1d9; text-align: left; padding: 10px 14px; font-size: 12px; }}
            QPushButton#thDone {{ background: {a}; color: #0d1117; border: none; border-radius: 10px; padding: 10px 28px; font-weight: 700; font-size: 13px; }}
            QFrame#thPreview {{ background: {a}; border-radius: 6px; }}
        """

    def _build_theme_dialog(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(tr("theme"))
        dlg.setModal(True)
        dlg.setFixedWidth(360)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(16)

        head = QtWidgets.QHBoxLayout()
        head.setSpacing(12)
        title = QtWidgets.QLabel(tr("theme"))
        title.setObjectName("thTitle")
        head.addWidget(title)
        head.addStretch(1)
        self._theme_preview = QtWidgets.QFrame()
        self._theme_preview.setObjectName("thPreview")
        self._theme_preview.setFixedSize(28, 28)
        head.addWidget(self._theme_preview)
        lay.addLayout(head)

        card = QtWidgets.QFrame()
        card.setObjectName("thCard")
        card_l = QtWidgets.QVBoxLayout(card)
        card_l.setContentsMargins(16, 14, 16, 16)
        card_l.setSpacing(12)

        cap = QtWidgets.QLabel(tr("theme_presets"))
        cap.setObjectName("thCap")
        card_l.addWidget(cap)

        grid_w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        group = QtWidgets.QButtonGroup(dlg)
        group.setExclusive(True)
        self.theme_btns = {}
        cols = 6
        for i, (tid, color) in enumerate(THEMES):
            cell = QtWidgets.QWidget()
            cell_l = QtWidgets.QVBoxLayout(cell)
            cell_l.setContentsMargins(0, 0, 0, 0)
            cell_l.setSpacing(4)
            cell_l.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            b: Swatch[str] = Swatch(color)
            b.setToolTip(tr(f"theme_{tid}"))
            b.setFixedSize(34, 34)
            b.clicked.connect(lambda _=False, t=tid: self._apply_theme(t))
            group.addButton(b)
            name = QtWidgets.QLabel(tr(f"theme_{tid}"))
            name.setObjectName("thSub")
            name.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            name.setStyleSheet("font-size: 10px; color: #6e7681;")
            cell_l.addWidget(b, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
            cell_l.addWidget(name)
            grid.addWidget(cell, i // cols, i % cols)
            self.theme_btns[tid] = b
            b._name_lbl = name

        card_l.addWidget(grid_w)
        lay.addWidget(card)

        sys_hex: str | None = system_accent()
        self._theme_system_btn = QtWidgets.QPushButton()
        self._theme_system_btn.setObjectName("thCustom")
        self._theme_system_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._theme_system_btn.clicked.connect(lambda: self._apply_theme("system"))
        system_inner = QtWidgets.QHBoxLayout(self._theme_system_btn)
        system_inner.setContentsMargins(4, 2, 8, 2)
        system_inner.setSpacing(12)
        self.system_swatch: Swatch[str] = Swatch(sys_hex or THEME_HEX[DEFAULT_THEME])
        self.system_swatch.setFixedSize(30, 30)
        self.system_swatch.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        system_inner.addWidget(self.system_swatch)
        system_txt_box = QtWidgets.QVBoxLayout()
        system_txt_box.setSpacing(1)
        self._theme_system_title = QtWidgets.QLabel(tr("theme_system"))
        self._theme_system_title.setStyleSheet(
            "font-size: 12px; font-weight: 600; color: #e6edf3; background: transparent;"
        )
        self._theme_system_hex = QtWidgets.QLabel(sys_hex or tr("theme_system_na"))
        self._theme_system_hex.setStyleSheet(
            "font-size: 11px; color: #8b949e; background: transparent; font-family: 'Cascadia Mono', 'Consolas', monospace;"
        )
        system_txt_box.addWidget(self._theme_system_title)
        system_txt_box.addWidget(self._theme_system_hex)
        system_inner.addLayout(system_txt_box)
        system_inner.addStretch(1)
        group.addButton(self.system_swatch)
        self.theme_btns["system"] = self.system_swatch
        self._theme_system_btn.setEnabled(sys_hex is not None)
        lay.addWidget(self._theme_system_btn)

        self._theme_custom_btn = QtWidgets.QPushButton()
        self._theme_custom_btn.setObjectName("thCustom")
        self._theme_custom_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._theme_custom_btn.clicked.connect(self._on_custom_theme)
        custom_inner = QtWidgets.QHBoxLayout(self._theme_custom_btn)
        custom_inner.setContentsMargins(4, 2, 8, 2)
        custom_inner.setSpacing(12)
        cc = CONFIG.get("custom_color", DEFAULT_CONFIG["custom_color"])
        self.custom_swatch: Swatch = Swatch(cc, custom=True)
        self.custom_swatch.setFixedSize(30, 30)
        self.custom_swatch.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        custom_inner.addWidget(self.custom_swatch)
        custom_txt_box = QtWidgets.QVBoxLayout()
        custom_txt_box.setSpacing(1)
        self._theme_custom_title = QtWidgets.QLabel(tr("theme_custom"))
        self._theme_custom_title.setStyleSheet(
            "font-size: 12px; font-weight: 600; color: #e6edf3; background: transparent;"
        )
        self._theme_custom_hex = QtWidgets.QLabel(cc)
        self._theme_custom_hex.setStyleSheet(
            "font-size: 11px; color: #8b949e; background: transparent; font-family: 'Cascadia Mono', 'Consolas', monospace;"
        )
        custom_txt_box.addWidget(self._theme_custom_title)
        custom_txt_box.addWidget(self._theme_custom_hex)
        custom_inner.addLayout(custom_txt_box)
        custom_inner.addStretch(1)
        group.addButton(self.custom_swatch)
        self.theme_btns["custom"] = self.custom_swatch
        lay.addWidget(self._theme_custom_btn)

        self._theme_dlg_label = QtWidgets.QLabel("")
        self._theme_dlg_label.setObjectName("thSub")
        self._theme_dlg_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._theme_dlg_label)

        close_btn = QtWidgets.QPushButton(tr("theme_close"))
        close_btn.setObjectName("thDone")
        close_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)

        self._theme_dlg_card: QtWidgets.QFrame = card
        self._theme_cascade = CascadeReveal(dlg)

        dlg.setStyleSheet(self._theme_dialog_qss())
        self._theme_dialog: QtWidgets.QDialog = dlg
        self._theme_dlg_title: QtWidgets.QLabel = title
        self._theme_dlg_close: QtWidgets.QPushButton = close_btn

    def _refresh_theme_dialog_texts(self) -> None:
        if not getattr(self, "_theme_dialog", None):
            return
        self._theme_dialog.setWindowTitle(tr("theme"))
        self._theme_dlg_title.setText(tr("theme"))
        self._theme_dlg_close.setText(tr("theme_close"))
        self._theme_custom_title.setText(tr("theme_custom"))
        self._theme_system_title.setText(tr("theme_system"))
        sys_hex: str | None = system_accent()
        self._theme_system_btn.setEnabled(sys_hex is not None)
        self._theme_system_btn.setToolTip(tr("theme_system_tip"))
        self._theme_system_hex.setText(sys_hex or tr("theme_system_na"))
        if sys_hex:
            self.system_swatch.set_color(sys_hex)
        for tid, btn in self.theme_btns.items():
            if tid == "custom":
                btn.setToolTip(tr("theme_custom_tip"))
            elif tid == "system":
                btn.setToolTip(tr("theme_system_tip"))
            else:
                btn.setToolTip(tr(f"theme_{tid}"))
                if hasattr(btn, "_name_lbl"):
                    btn._name_lbl.setText(tr(f"theme_{tid}"))
        if self.theme_id in self.theme_btns:
            self.theme_btns[self.theme_id].setChecked(True)
        if self.theme_id == "custom":
            name: str = tr("theme_custom")
            hex_c = CONFIG.get("custom_color", "")
        elif self.theme_id == "system":
            name: str = tr("theme_system")
            hex_c: str = self._theme_color("system")
        else:
            name: str = tr(f"theme_{self.theme_id}")
            hex_c: str = THEME_HEX.get(self.theme_id, "")
        self._theme_dlg_label.setText(f"{name}  ·  {hex_c}")
        self._theme_custom_hex.setText(
            str(CONFIG.get("custom_color", DEFAULT_CONFIG["custom_color"]))
        )
        chip: str = self._theme_color(self.theme_id)
        self._theme_preview.setStyleSheet(f"background: {chip}; border-radius: 6px;")
        self._theme_dialog.setStyleSheet(self._theme_dialog_qss())

    def _open_theme_dialog(self) -> None:
        if not getattr(self, "_theme_dialog", None):
            self._build_theme_dialog()
        self._refresh_theme_dialog_texts()
        self._theme_dialog.adjustSize()
        self._theme_dialog.setFixedHeight(self._theme_dialog.height())
        self._theme_cascade.prepare(
            [
                [self._theme_dlg_title, self._theme_preview],
                self._theme_dlg_card,
                self._theme_system_btn,
                self._theme_custom_btn,
                self._theme_dlg_label,
                self._theme_dlg_close,
            ]
        )
        self._theme_cascade.play()
        fade_in_when_shown(self._theme_dialog)
        self._theme_dialog.exec()

    def _on_custom_theme(self) -> None:
        initial = QtGui.QColor(
            CONFIG.get("custom_color", DEFAULT_CONFIG["custom_color"])
        )
        parent = self._theme_dialog if getattr(self, "_theme_dialog", None) else self
        color: QtGui.QColor = QtWidgets.QColorDialog.getColor(
            initial, parent, tr("theme_custom")
        )
        if not color.isValid():
            if self.theme_id == "custom":
                self.custom_swatch.setChecked(True)
            return
        hex_color: str = color.name()
        CONFIG["custom_color"] = hex_color
        self.custom_swatch.set_color(hex_color)
        self._theme_custom_hex.setText(hex_color)
        self._apply_theme("custom")
        if getattr(self, "_theme_dialog", None):
            self._refresh_theme_dialog_texts()

    def set_n(self, n) -> None:
        self.n = n
        set_ring_window(n)
        self._last_total = state.get("total_samples", 0)
        self._bpm_last_total = self._last_total
        self._metrics_total = -(10**9)
        self._alloc_scratch(n)
        self._x: np.ndarray[Any, np.dtype[np.floating]] = np.arange(n, dtype=np.float32)
        self.last: np.ndarray[Any, np.dtype[np.floating]] = np.zeros(
            n, dtype=np.float32
        )
        self.last_r: np.ndarray[Any, np.dtype[np.floating]] = np.zeros(
            n, dtype=np.float32
        )
        self.prev_tpl = None
        self._crossing_smooth = None
        self._period_smooth = 0.0
        self._lock_quality = 0.0
        self._sm_bufs.clear()
        self._peak_hold.reset()
        self._peak_hold_r.reset()
        for ph in self.multi_peak_holds:
            ph.reset()
        if self._fade_on or self._fade_blend > 0.01:
            self._rebuild_demo_caches()
        self.plot.setXRange(-n * 0.012, n, padding=0)
        self.bottom.setTickSpacing(TICKS[n], TICKS[n] / 2)
        self._reapply_ranges()
        if not getattr(self, "_loading", False):
            CONFIG["n"] = n
            _save_config(CONFIG)

    def set_gain(self, g) -> None:
        target_scale = 1.0 / float(g)
        self.lim = target_scale
        self.left.setTickSpacing(0.5 * target_scale, 0.25 * target_scale)
        if getattr(self, "_loading", False):
            self._apply_gain_scale(target_scale)
        else:
            animation = getattr(self, "_gain_scale_animation", None)
            if animation is None:
                animation = QtCore.QVariantAnimation(self)
                animation.setDuration(240)
                animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
                animation.valueChanged.connect(self._apply_gain_scale)
                self._gain_scale_animation = animation

            animation.stop()
            animation.setStartValue(self._gain_display_scale)
            animation.setEndValue(target_scale)
            animation.start()
            CONFIG["gain"] = g
            _save_config(CONFIG)

    def _apply_gain_scale(self, scale: float) -> None:
        scale = float(scale)
        self.plot.setYRange(-1.06 * scale, 1.06 * scale, padding=0)
        self._gain_display_scale = scale
        self.lim_hi.setPos(scale)
        self.lim_lo.setPos(-scale)

    def on_freeze(self, on) -> None:
        self.frozen = on
        if not on:
            _ui_gate[0] = True
        self.freeze_btn.setText(self._freeze_text(on))
        self._update_freeze_icon()
        self.solo_freeze.setChecked(on)

    def _on_trigger_toggle(self, on) -> None:
        self.prev_tpl = None
        self._crossing_smooth = None
        self._period_smooth = 0.0
        self._lock_quality = 0.0
        CONFIG["stabilization"] = bool(on)
        _save_config(CONFIG)

    def _on_glow_toggle(self, on: bool) -> None:
        self.glow_renderer.set_enabled(on)
        self.glow_renderer_r.set_enabled(on and self._n_ch == 2)
        CONFIG["glow"] = bool(on)
        _save_config(CONFIG)

    def _on_fade_toggle(self, on: bool) -> None:
        self._fade_on = bool(on)
        self._fade_target: float = 1.0 if self._fade_on else 0.0
        if self._fade_on:
            self._rebuild_demo_caches()
        CONFIG["fade"] = self._fade_on
        _save_config(CONFIG)

    def _on_gradient_toggle(self, on) -> None:
        self._show_gradient = bool(on)
        self._rebuild_pens()
        CONFIG["gradient"] = self._show_gradient
        _save_config(CONFIG)

    def _on_peak_hold_toggle(self, on) -> None:
        self._show_peak_hold = bool(on)
        self.peak_hold_curve.setVisible(self._show_peak_hold and self._n_ch <= 2)
        self.peak_hold_curve_r.setVisible(self._show_peak_hold and self._n_ch == 2)
        if not self._show_peak_hold:
            self.peak_hold_curve.clear()
            self.peak_hold_curve_r.clear()
            for c in self.multi_peak_curves:
                c.setVisible(False)
        CONFIG["peak_hold"] = self._show_peak_hold
        _save_config(CONFIG)

    def _on_smooth_toggle(self, on) -> None:
        self._smooth_stab = bool(on)
        self._sm_bufs.clear()
        self._hold_samples = 0
        CONFIG["smooth_stab"] = self._smooth_stab
        _save_config(CONFIG)

    def _sync_edge_clip_enabled(self, *_args) -> None:
        """«Об края» доступно только при включённой стабилизации
        и включённой плавной стабилизации."""
        self.edge_clip_sw.setEnabled(
            self.trigger.isChecked() and self.smooth_sw.isChecked()
        )

    def _on_edge_clip_toggle(self, on) -> None:
        self._edge_clip_on = bool(on)
        self._sm_bufs.clear()
        self._sm_scale.clear()
        CONFIG["edge_clip"] = self._edge_clip_on
        _save_config(CONFIG)

    def _on_bpm_toggle(self, on) -> None:
        self._bpm_on = bool(on)
        CONFIG["bpm_on"] = self._bpm_on
        _save_config(CONFIG)

    def _on_band_change(self, idx) -> None:
        if idx < 0:
            return
        key = self.band_combo.itemData(idx)
        if not key:
            return
        self._band_kind = key
        self._filters: list[None | Biquad] = [make_filter(key) for _ in range(MAX_CH)]
        CONFIG["bandpass"] = key
        _save_config(CONFIG)

    def _on_layout_change(self, idx) -> None:
        if idx < 0:
            return
        key = self.ch_layout_combo.itemData(idx)
        if key:
            self._apply_ch_layout(key)

    def _apply_ch_layout(self, layout: str) -> None:
        if layout == self.ch_layout and self._layout_applied:
            return
        self.ch_layout: str = layout
        self._n_ch: int = CH_LAYOUT_TO_N.get(layout, 1)
        self.channels: str = "mono" if self._n_ch == 1 else "stereo"
        set_channel_count(self._n_ch)
        self._alloc_scratch(self.n)

        for c in self.multi_curves:
            try:
                self.plot.removeItem(c)
            except Exception:
                pass
        self.multi_curves = []

        for c in self.multi_peak_curves:
            try:
                self.plot.removeItem(c)
            except Exception:
                pass
        self.multi_peak_curves = []
        self.multi_peak_holds = []

        if self._n_ch > 2:
            for i in range(self._n_ch):
                c = GradientCurveItem()
                c.setZValue(-0.1 - 0.01 * i)
                self.plot.addItem(c)
                self.multi_curves.append(c)

                ph = PeakHold(decay=0.94, smooth=0.5, window=15)
                self.multi_peak_holds.append(ph)

                ph_c = pg.PlotCurveItem(pen=None)
                ph_c.setZValue(-0.8 - 0.01 * i)
                ph_c.setVisible(self._show_peak_hold)
                self.plot.addItem(ph_c)
                self.multi_peak_curves.append(ph_c)

            self.curve.clear()
            self.curve_r.clear()
            self.curve.setVisible(False)
            self.curve_r.setVisible(False)
            self.peak_hold_curve.setVisible(False)
            self.peak_hold_curve_r.setVisible(False)
        else:
            self.curve.setVisible(True)
            self.curve_r.setVisible(self._n_ch == 2)
            self.peak_hold_curve.setVisible(self._show_peak_hold)
            self.peak_hold_curve_r.setVisible(self._show_peak_hold and self._n_ch == 2)

        self.trigger.setEnabled(True)

        if self._n_ch > 2:
            if self.glow_renderer._item is not None:
                self.glow_renderer._item.clear()
            if self.glow_renderer_r._item is not None:
                self.glow_renderer_r._item.clear()

        self.prev_tpl = None
        self._crossing_smooth = None
        self._period_smooth = 0.0
        self._lock_quality = 0.0
        self._last_total = state.get("total_samples", 0)
        self._bpm_last_total = self._last_total
        self._metrics_total = -(10**9)
        self._sm_bufs.clear()
        self._peak_hold.reset()
        self._peak_hold_r.reset()
        for ph in self.multi_peak_holds:
            ph.reset()

        for f in self._filters:
            if f is not None:
                f.reset()

        self.glow_renderer_r.set_enabled(
            self.glow_renderer.is_enabled() and self._n_ch == 2
        )

        self._layout_applied = True
        if not getattr(self, "_loading", False):
            CONFIG["ch_layout"] = layout
            _save_config(CONFIG)
        self._rebuild_pens()