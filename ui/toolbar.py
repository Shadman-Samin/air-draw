"""
Toolbar panel containing brushes, colors, sliders, and controls.

Organizes options into clean logical sections using the dark theme QSS.
Integrates directly with the central EventHub for state updates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.events import events
from drawing.color_manager import ColorManager

if TYPE_CHECKING:
    from settings.settings_manager import SettingsManager


class DrawingToolbar(QFrame):
    """
    Scrollable control toolbar for brushes, colors, eraser, and history.
    """

    def __init__(self, settings: SettingsManager | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("toolbar_frame")
        self._settings = settings
        self.color_manager = ColorManager()

        self._init_ui()
        self._connect_ui_signals()
        self._connect_event_signals()

    def _init_ui(self) -> None:
        """Construct scrollable panel with grouped controls."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        self.main_layout = QVBoxLayout(panel)
        self.main_layout.setSpacing(12)
        self.main_layout.setContentsMargins(12, 12, 12, 12)

        # ── Tracking mode ──
        self.main_layout.addWidget(self._create_section_title("TRACKING MODE"))
        mode_layout = QHBoxLayout()
        self.btn_hand_mode = QPushButton("Hand")
        self.btn_hand_mode.setCheckable(True)
        self.btn_hand_mode.setChecked(True)
        self.btn_color_mode = QPushButton("Color")
        self.btn_color_mode.setCheckable(True)
        mode_layout.addWidget(self.btn_hand_mode)
        mode_layout.addWidget(self.btn_color_mode)
        self.main_layout.addLayout(mode_layout)

        # ── Display options ──
        self.main_layout.addWidget(self._create_section_title("DISPLAY"))
        wb_on = False
        multi_on = True
        if self._settings is not None:
            wb_on = bool(self._settings.get("whiteboard.enabled", False))
            multi_on = bool(self._settings.get("multi_hand.enabled", True))
        self.chk_whiteboard = QCheckBox("Virtual Whiteboard")
        self.chk_whiteboard.setChecked(wb_on)
        self.chk_multi_hand = QCheckBox("Multi-hand Drawing")
        self.chk_multi_hand.setChecked(multi_on)
        self.main_layout.addWidget(self.chk_whiteboard)
        self.main_layout.addWidget(self.chk_multi_hand)

        # ── Drawing tools (2-row grid so buttons fit sidebar width) ──
        self.main_layout.addWidget(self._create_section_title("DRAWING TOOL"))
        tool_grid = QGridLayout()
        tool_grid.setSpacing(6)
        self._tool_buttons: dict[str, QPushButton] = {}
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        tools = [
            ("freehand", "Pen", 0, 0),
            ("line", "Line", 0, 1),
            ("rectangle", "Rect", 0, 2),
            ("circle", "Circle", 1, 0),
            ("arrow", "Arrow", 1, 1),
        ]
        for tool_id, label, row, col in tools:
            btn = QPushButton(label)
            btn.setObjectName("tool_btn")
            btn.setCheckable(True)
            if tool_id == "freehand":
                btn.setChecked(True)
            self._tool_group.addButton(btn)
            self._tool_buttons[tool_id] = btn
            tool_grid.addWidget(btn, row, col)
        self.main_layout.addLayout(tool_grid)

        # ── Brush size ──
        self.main_layout.addWidget(self._create_section_title("BRUSH SETTINGS"))
        self.brush_size_label = QLabel("Brush Size: 6px")
        self.main_layout.addWidget(self.brush_size_label)
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(1, 40)
        self.brush_slider.setValue(6)
        self.main_layout.addWidget(self.brush_slider)

        # ── Color palette ──
        self.main_layout.addWidget(self._create_section_title("BRUSH COLOR"))
        palette_grid_widget = QWidget()
        palette_layout = QVBoxLayout(palette_grid_widget)
        palette_layout.setContentsMargins(0, 0, 0, 0)
        palette_layout.setSpacing(6)

        preset_list = self.color_manager.palette[:24]
        row_layout = QHBoxLayout()
        row_layout.setSpacing(4)
        for idx, bgr_color in enumerate(preset_list):
            if idx > 0 and idx % 8 == 0:
                palette_layout.addLayout(row_layout)
                row_layout = QHBoxLayout()
                row_layout.setSpacing(4)
            btn = QPushButton()
            btn.setObjectName("color_preset_btn")
            hex_color = ColorManager.bgr_to_hex(bgr_color)
            btn.setStyleSheet(f"background-color: {hex_color};")
            btn.clicked.connect(lambda checked, col=bgr_color: self._on_color_preset_clicked(col))
            row_layout.addWidget(btn)
        palette_layout.addLayout(row_layout)
        self.main_layout.addWidget(palette_grid_widget)

        self.btn_custom_color = QPushButton("More Colors...")
        self.btn_custom_color.setObjectName("accent_btn")
        self.main_layout.addWidget(self.btn_custom_color)

        # ── Eraser ──
        self.main_layout.addWidget(self._create_section_title("ERASER SETTINGS"))
        self.eraser_size_label = QLabel("Eraser Size: 28px")
        self.main_layout.addWidget(self.eraser_size_label)
        self.eraser_slider = QSlider(Qt.Orientation.Horizontal)
        self.eraser_slider.setRange(5, 80)
        self.eraser_slider.setValue(28)
        self.main_layout.addWidget(self.eraser_slider)

        # ── History & save ──
        self.main_layout.addWidget(self._create_section_title("CANVAS"))
        history_layout = QHBoxLayout()
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setEnabled(False)
        self.btn_redo = QPushButton("Redo")
        self.btn_redo.setEnabled(False)
        history_layout.addWidget(self.btn_undo)
        history_layout.addWidget(self.btn_redo)
        self.main_layout.addLayout(history_layout)

        self.btn_clear = QPushButton("Clear Canvas")
        self.main_layout.addWidget(self.btn_clear)
        self.btn_save = QPushButton("Save as Image...")
        self.btn_save.setObjectName("accent_btn")
        self.main_layout.addWidget(self.btn_save)

        self.main_layout.addStretch()
        scroll.setWidget(panel)
        outer.addWidget(scroll)

    def _create_section_title(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("title_label")
        return lbl

    def _connect_ui_signals(self) -> None:
        self.btn_hand_mode.clicked.connect(self._on_hand_mode_clicked)
        self.btn_color_mode.clicked.connect(self._on_color_mode_clicked)
        self.brush_slider.valueChanged.connect(self._on_brush_slider_changed)
        self.eraser_slider.valueChanged.connect(self._on_eraser_slider_changed)
        self.btn_custom_color.clicked.connect(self._on_custom_color_clicked)
        self.btn_clear.clicked.connect(lambda: events.canvas_cleared.emit())
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_undo.clicked.connect(self._on_undo_clicked)
        self.btn_redo.clicked.connect(self._on_redo_clicked)
        self.chk_whiteboard.toggled.connect(events.whiteboard_mode_changed.emit)
        self.chk_multi_hand.toggled.connect(events.multi_hand_changed.emit)
        self._tool_group.buttonClicked.connect(self._on_tool_button_clicked)

    def _connect_event_signals(self) -> None:
        events.undo_available.connect(self.set_undo_enabled)
        events.redo_available.connect(self.set_redo_enabled)

    def _on_tool_button_clicked(self, button: QPushButton) -> None:
        for tool_id, btn in self._tool_buttons.items():
            if btn is button:
                events.drawing_tool_changed.emit(tool_id)
                break

    def _on_hand_mode_clicked(self) -> None:
        self.btn_hand_mode.setChecked(True)
        self.btn_color_mode.setChecked(False)
        events.tracking_mode_changed.emit("hand")

    def _on_color_mode_clicked(self) -> None:
        self.btn_color_mode.setChecked(True)
        self.btn_hand_mode.setChecked(False)
        events.tracking_mode_changed.emit("color")

    def _on_brush_slider_changed(self, value: int) -> None:
        self.brush_size_label.setText(f"Brush Size: {value}px")
        events.brush_size_changed.emit(value)

    def _on_eraser_slider_changed(self, value: int) -> None:
        self.eraser_size_label.setText(f"Eraser Size: {value}px")
        events.eraser_size_changed.emit(value)

    def _on_color_preset_clicked(self, bgr_color: tuple[int, int, int]) -> None:
        self.color_manager.current_color = bgr_color
        events.color_changed.emit(bgr_color)

    def _on_custom_color_clicked(self) -> None:
        curr = self.color_manager.current_color
        rgb = ColorManager.bgr_to_rgb(curr)
        q_color = QColorDialog.getColor(QColor(*rgb), self, "Select Brush Color")
        if q_color.isValid():
            selected_bgr = ColorManager.rgb_to_bgr(
                (q_color.red(), q_color.green(), q_color.blue()),
            )
            self.color_manager.current_color = selected_bgr
            events.color_changed.emit(selected_bgr)

    def _on_undo_clicked(self) -> None:
        events.status_message.emit("Undo performed", 1000)

    def _on_redo_clicked(self) -> None:
        events.status_message.emit("Redo performed", 1000)

    def _on_save_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Drawing",
            "drawing.png",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if path:
            events.canvas_save_requested.emit(path)

    @pyqtSlot(bool)
    def set_undo_enabled(self, enabled: bool) -> None:
        self.btn_undo.setEnabled(enabled)

    @pyqtSlot(bool)
    def set_redo_enabled(self, enabled: bool) -> None:
        self.btn_redo.setEnabled(enabled)
