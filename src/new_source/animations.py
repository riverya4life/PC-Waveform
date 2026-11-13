"""Reusable animations for interactive UI controls."""

from typing import Any, Callable

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


WINDOW_OPEN_DURATION_MS = 300
SOLO_TRANSITION_DURATION_MS = 300
BUTTON_FEEDBACK_DURATION_MS = 300
HOVER_EASING_CURVE = QtCore.QEasingCurve.Type.InOutSine
BUTTON_INTRO_DURATION_MS = 300


def fade_in(
    widget: QtWidgets.QWidget,
    duration: int = WINDOW_OPEN_DURATION_MS,
) -> QtCore.QPropertyAnimation:
    """Fade a newly shown widget from transparent to fully visible."""
    widget.setWindowOpacity(0.0)
    animation = QtCore.QPropertyAnimation(widget, b"windowOpacity", widget)
    animation.setDuration(duration)
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
    animation.start()
    return animation


def fade_out(
    widget: QtWidgets.QWidget,
    duration: int = SOLO_TRANSITION_DURATION_MS,
) -> QtCore.QPropertyAnimation:
    """Fade a visible widget out."""
    animation = QtCore.QPropertyAnimation(widget, b"windowOpacity", widget)
    animation.setDuration(duration)
    animation.setStartValue(widget.windowOpacity())
    animation.setEndValue(0.0)
    animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
    animation.start()
    return animation


def fade_in_when_shown(
    widget: QtWidgets.QWidget,
    duration: int = WINDOW_OPEN_DURATION_MS,
) -> None:
    """Prepare a top-level widget to fade in as its event loop starts."""
    widget.setWindowOpacity(0.0)

    def start_animation() -> None:
        setattr(widget, "_open_animation", fade_in(widget, duration))

    QtCore.QTimer.singleShot(0, start_animation)


class ButtonAnimation(QtCore.QObject):
    """Adds an animated accent overlay and optional entrance fade to a control."""

    def __init__(
        self,
        button: QtWidgets.QWidget,
        *,
        animate_entrance: bool = True,
    ) -> None:
        super().__init__(button)
        self._button = button
        self._entrance_effect: QtWidgets.QGraphicsOpacityEffect | None = None
        self._entrance_animation: QtCore.QPropertyAnimation | None = None
        if animate_entrance:
            self._entrance_effect = QtWidgets.QGraphicsOpacityEffect(button)
            self._entrance_effect.setOpacity(0.0)
            button.setGraphicsEffect(self._entrance_effect)
            self._entrance_animation = QtCore.QPropertyAnimation(
                self._entrance_effect, b"opacity", self
            )
            self._entrance_animation.setDuration(BUTTON_INTRO_DURATION_MS)
            self._entrance_animation.setStartValue(0.0)
            self._entrance_animation.setEndValue(1.0)
            self._entrance_animation.setEasingCurve(
                QtCore.QEasingCurve.Type.OutCubic
            )

        self._overlay = _HoverOverlay(button, self._accent_for(button))
        self._hover_animation = QtCore.QVariantAnimation(self)
        self._hover_animation.setDuration(BUTTON_FEEDBACK_DURATION_MS)
        self._hover_animation.setEasingCurve(HOVER_EASING_CURVE)
        self._hover_animation.valueChanged.connect(self._overlay.set_progress)
        button.installEventFilter(self)
        setattr(button, "_button_animation", self)

    def show(self, delay_ms: int = 0) -> None:
        if self._entrance_animation is not None:
            QtCore.QTimer.singleShot(delay_ms, self._entrance_animation.start)

    @staticmethod
    def _accent_for(widget: QtWidgets.QWidget) -> QtGui.QColor:
        current: QtWidgets.QWidget | None = widget
        while current is not None:
            accent = getattr(current, "accent", None)
            if accent:
                color = QtGui.QColor(str(accent))
                if color.isValid():
                    return color
            current = current.parentWidget()
        return QtGui.QColor("#c9d1d9")

    def _animate_hover(self, progress: float) -> None:
        if (
            self._hover_animation.state() == QtCore.QAbstractAnimation.State.Running
            and float(self._hover_animation.endValue()) == progress
        ):
            return
        if self._overlay.progress == progress:
            return
        distance = abs(progress - self._overlay.progress)
        self._hover_animation.stop()
        self._hover_animation.setDuration(
            max(90, int(BUTTON_FEEDBACK_DURATION_MS * distance))
        )
        self._hover_animation.setStartValue(self._overlay.progress)
        self._hover_animation.setEndValue(progress)
        self._hover_animation.start()

    def handle_event(self, event: QtCore.QEvent) -> None:
        event_type = event.type()
        if event_type == QtCore.QEvent.Type.Enter:
            self._overlay.set_accent(self._accent_for(self._button))
            self._animate_hover(1.0)
        elif event_type == QtCore.QEvent.Type.Leave:
            self._animate_hover(0.0)
        elif event_type == QtCore.QEvent.Type.MouseButtonPress:
            self._animate_hover(1.35)
        elif event_type == QtCore.QEvent.Type.MouseButtonRelease:
            self._animate_hover(1.0 if self._button.underMouse() else 0.0)
        elif event_type == QtCore.QEvent.Type.EnabledChange:
            self._animate_hover(1.0 if self._button.underMouse() else 0.0)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Resize:
            self._overlay.setGeometry(self._button.rect())
        elif event.type() == QtCore.QEvent.Type.Show:
            self._overlay.setGeometry(self._button.rect())
            self._overlay.raise_()
        self.handle_event(event)
        return super().eventFilter(watched, event)


class _HoverOverlay(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, accent: QtGui.QColor) -> None:
        super().__init__(parent)
        self._accent = QtGui.QColor(accent)
        self.progress = 0.0
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent; border: none;")
        self.setGeometry(parent.rect())
        self.hide()

    def set_accent(self, accent: QtGui.QColor) -> None:
        self._accent = QtGui.QColor(accent)

    def set_progress(self, progress: Any) -> None:
        self.progress = float(progress)
        if self.progress > 0.0 and not self.isVisible():
            self.raise_()
            self.show()
        elif self.progress <= 0.0:
            self.hide()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if self.progress <= 0.0:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = QtCore.QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        if rect.isEmpty():
            return
        strength = min(1.0, max(0.0, self.progress))
        alpha = min(32, int(22 * strength))
        fill = QtGui.QColor(self._accent)
        fill.setAlpha(alpha)
        outline = QtGui.QColor(self._accent)
        outline.setAlpha(min(170, int(110 * strength)))
        painter.setBrush(fill)
        painter.setPen(QtGui.QPen(outline, 1.0))
        painter.drawRoundedRect(rect, 8.0, 8.0)


def animate_button(
    button: QtWidgets.QPushButton,
    intro_delay_ms: int = 0,
) -> ButtonAnimation:
    """Animate a button's entrance and pointer interaction."""
    animation = ButtonAnimation(button)
    animation.show(intro_delay_ms)
    return animation


class ButtonHoverFilter(QtCore.QObject):
    """Lazily animates hover feedback for interactive buttons and combo boxes."""

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        is_abstract_button = isinstance(watched, QtWidgets.QAbstractButton)
        is_combo_box = isinstance(watched, QtWidgets.QComboBox)
        if type(watched).__name__ in ("ToggleSwitch", "Swatch", "IconButton"):
            return super().eventFilter(watched, event)
        if is_abstract_button and watched.objectName() == "segbtn":
            # У сегментированных переключателей собственный «плавающий» ховер.
            return super().eventFilter(watched, event)
        is_hover_control = is_abstract_button or is_combo_box
        if is_hover_control and event.type() in (
            QtCore.QEvent.Type.Enter,
            QtCore.QEvent.Type.MouseButtonPress,
        ):
            animation = getattr(watched, "_button_animation", None)
            if animation is None:
                animation = ButtonAnimation(watched, animate_entrance=False)
            animation.handle_event(event)
        return super().eventFilter(watched, event)


def install_button_hover_effects(
    app: QtWidgets.QApplication,
) -> ButtonHoverFilter:
    """Enable animated hover feedback for existing and future application buttons."""
    hover_filter = ButtonHoverFilter(app)
    app.installEventFilter(hover_filter)
    setattr(app, "_button_hover_filter", hover_filter)
    return hover_filter


class ValueAnimation:
    """Owns a QVariantAnimation and provides a consistent start/retarget API."""

    def __init__(
        self,
        parent: QtCore.QObject,
        duration: int,
        on_value_changed: Callable[[Any], None],
        easing_curve: QtCore.QEasingCurve.Type = QtCore.QEasingCurve.Type.OutCubic,
    ) -> None:
        self._animation = QtCore.QVariantAnimation(parent)
        self._animation.setDuration(duration)
        self._animation.setEasingCurve(easing_curve)
        self._animation.valueChanged.connect(on_value_changed)

    def animate(self, start: Any, end: Any) -> None:
        self._animation.stop()
        self._animation.setStartValue(start)
        self._animation.setEndValue(end)
        self._animation.start()

    def stop(self) -> None:
        self._animation.stop()


class _SlideFadeEffect(QtWidgets.QGraphicsEffect):
    """Появление: прозрачность + лёгкий подъём снизу. Не трогает layout —
    виджет только рисуется со смещением, поэтому размеры окна не «дёргаются»."""

    def __init__(self, parent: QtCore.QObject, distance: int = 14) -> None:
        super().__init__(parent)
        self._progress = 0.0
        self._distance = int(distance)

    def set_progress(self, value: float) -> None:
        value = min(1.0, max(0.0, float(value)))
        if value != self._progress:
            self._progress = value
            self.update()

    def boundingRectFor(self, rect: QtCore.QRectF) -> QtCore.QRectF:
        return rect.adjusted(0.0, 0.0, 0.0, float(self._distance))

    def draw(self, painter: QtGui.QPainter) -> None:
        if self._progress >= 1.0:
            self.drawSource(painter)
            return
        if self._progress <= 0.0:
            return
        pixmap, offset = self.sourcePixmap(
            QtCore.Qt.CoordinateSystem.DeviceCoordinates
        )
        if pixmap.isNull():
            return
        dpr = painter.device().devicePixelRatioF()
        dy = int(round((1.0 - self._progress) * self._distance * dpr))
        painter.save()
        painter.setWorldTransform(QtGui.QTransform())
        painter.setOpacity(painter.opacity() * self._progress)
        painter.drawPixmap(offset + QtCore.QPoint(0, dy), pixmap)
        painter.restore()


class CascadeReveal(QtCore.QObject):
    """Каскадное появление блоков окна: каждый следующий блок стартует чуть
    позже предыдущего и плавно выезжает снизу с проявлением.

    Один общий таймер на всю анимацию; по окончании эффекты снимаются, чтобы
    не тратить ресурсы на перерисовку через pixmap."""

    def __init__(
        self,
        parent: QtCore.QObject,
        item_ms: int = 380,
        stagger_ms: int = 60,
        distance: int = 14,
    ) -> None:
        super().__init__(parent)
        self._item_ms = max(1, int(item_ms))
        self._stagger_ms = int(stagger_ms)
        self._distance = int(distance)
        self._curve = QtCore.QEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._groups: list[list[tuple[QtWidgets.QWidget, _SlideFadeEffect]]] = []
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.Linear)
        self._anim.valueChanged.connect(self._on_tick)
        self._anim.finished.connect(self._clear)

    def prepare(self, groups: list[Any]) -> None:
        """Спрятать блоки (прогресс 0). groups — виджеты или списки виджетов;
        виджеты из одного списка появляются одновременно."""
        self._anim.stop()
        self._clear()
        for group in groups:
            widgets = list(group) if isinstance(group, (list, tuple)) else [group]
            entries = []
            for widget in widgets:
                effect = _SlideFadeEffect(widget, self._distance)
                effect.set_progress(0.0)
                widget.setGraphicsEffect(effect)
                entries.append((widget, effect))
            self._groups.append(entries)

    def play(self) -> None:
        if not self._groups:
            return
        total = self._stagger_ms * (len(self._groups) - 1) + self._item_ms
        self._anim.setDuration(total)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(float(total))
        # Старт после показа окна, когда уже запущен его цикл событий.
        QtCore.QTimer.singleShot(0, self._anim.start)

    def _on_tick(self, value: Any) -> None:
        elapsed = float(value)
        for index, entries in enumerate(self._groups):
            t = (elapsed - index * self._stagger_ms) / self._item_ms
            progress = self._curve.valueForProgress(min(1.0, max(0.0, t)))
            for _widget, effect in entries:
                effect.set_progress(progress)

    def _clear(self) -> None:
        for entries in self._groups:
            for widget, effect in entries:
                try:
                    if widget.graphicsEffect() is effect:
                        widget.setGraphicsEffect(None)
                except RuntimeError:
                    pass
        self._groups = []