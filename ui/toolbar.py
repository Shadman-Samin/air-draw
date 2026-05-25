"""
Toolbar panel containing brushes, colors, sliders, and controls.

Organizes options into clean logical sections using the dark theme QSS.
Integrates directly with the central EventHub for state updates.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.events import events
from drawing.color_manager import ColorManager


class DrawingToolbar(QFrame):
    """
    Control toolbar for managing brushes, colors, eraser, and history.

    Features:
    - Custom styled preset color palette buttons
    - Fine color selector via standard dialog
    - Size sliders for brush and eraser
    - Tracking engine switches (Hand / Color)
    - Thread-safe action routing
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("toolbar_frame")
        self.color_manager = ColorManager()
        
        self._init_ui()
        self._connect_ui_signals()
        self._connect_event_signals()

    def _init_ui(self) -> None:
        """Construct visual elements and layouts."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(14)
        self.main_layout.setContentsMargins(12, 12, 12, 12)

        # ----------------------------------------------------
        # SECTION 1: Tracking Mode Selectors
        # ----------------------------------------------------
        self.main_layout.addWidget(self._create_section_title("TRACKING MODE"))
        
        mode_layout = QHBoxLayout()
        self.btn_hand_mode = QPushButton("Hand Tracker")
        self.btn_hand_mode.setCheckable(True)
        self.btn_hand_mode.setChecked(True)
        
        self.btn_color_mode = QPushButton("Color Marker")
        self.btn_color_mode.setCheckable(True)
        
        # Exclusively toggle modes
        mode_layout.addWidget(self.btn_hand_mode)
        mode_layout.addWidget(self.btn_color_mode)
        self.main_layout.addLayout(mode_layout)

        # ----------------------------------------------------
        # SECTION 2: Brush Settings
        # ----------------------------------------------------
        self.main_layout.addWidget(self._create_section_title("BRUSH SETTINGS"))
        
        # Brush Size Slider
        self.brush_size_label = QLabel("Brush Size: 6px")
        self.main_layout.addWidget(self.brush_size_label)
        
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(1, 40)
        self.brush_slider.setValue(6)
        self.main_layout.addWidget(self.brush_slider)

        # ----------------------------------------------------
        # SECTION 3: Preset Palette (4x6 Matrix)
        # ----------------------------------------------------
        self.main_layout.addWidget(self._create_section_title("BRUSH COLOR"))
        
        palette_grid_widget = QWidget()
        palette_layout = QVBoxLayout(palette_grid_widget)
        palette_layout.setContentsMargins(0, 0, 0, 0)
        palette_layout.setSpacing(6)
        
        # We build 3 rows of 8 color circles
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
            
            # Style the circular button with its custom solid color
            hex_color = ColorManager.bgr_to_hex(bgr_color)
            btn.setStyleSheet(f"background-color: {hex_color};")
            
            # Bind the color click using custom lambda parameters
            btn.clicked.connect(lambda checked, col=bgr_color: self._on_color_preset_clicked(col))
            row_layout.addWidget(btn)
            
        palette_layout.addLayout(row_layout)
        self.main_layout.addWidget(palette_grid_widget)

        # Custom picker
        self.btn_custom_color = QPushButton("More Colors...")
        self.btn_custom_color.setObjectName("accent_btn")
        self.main_layout.addWidget(self.btn_custom_color)

        # ----------------------------------------------------
        # SECTION 4: Eraser Settings
        # ----------------------------------------------------
        self.main_layout.addWidget(self._create_section_title("ERASER SETTINGS"))
        
        self.eraser_size_label = QLabel("Eraser Size: 28px")
        self.main_layout.addWidget(self.eraser_size_label)
        
        self.eraser_slider = QSlider(Qt.Orientation.Horizontal)
        self.eraser_slider.setRange(5, 80)
        self.eraser_slider.setValue(28)
        self.main_layout.addWidget(self.eraser_slider)

        # ----------------------------------------------------
        # SECTION 5: Canvas Controls
        # ----------------------------------------------------
        self.main_layout.addWidget(self._create_section_title("CANVAS HISTORY"))
        
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

        # Add vertical spacer at the end to pack everything to the top
        self.main_layout.addStretch()

    def _create_section_title(self, text: str) -> QLabel:
        """Create standard premium section headers."""
        lbl = QLabel(text)
        lbl.setObjectName("title_label")
        return lbl

    def _connect_ui_signals(self) -> None:
        """Wire UI controls to methods emitting events."""
        # Modes
        self.btn_hand_mode.clicked.connect(self._on_hand_mode_clicked)
        self.btn_color_mode.clicked.connect(self._on_color_mode_clicked)
        
        # Sliders
        self.brush_slider.valueChanged.connect(self._on_brush_slider_changed)
        self.eraser_slider.valueChanged.connect(self._on_eraser_slider_changed)
        
        # Color picker
        self.btn_custom_color.clicked.connect(self._on_custom_color_clicked)
        
        # Actions
        self.btn_clear.clicked.connect(lambda: events.canvas_cleared.emit())
        self.btn_undo.clicked.connect(self._on_undo_clicked)
        self.btn_redo.clicked.connect(self._on_redo_clicked)

    def _connect_event_signals(self) -> None:
        """Wire event listeners from system managers to UI controls."""
        events.undo_available.connect(self.set_undo_enabled)
        events.redo_available.connect(self.set_redo_enabled)

    # --- UI Actions Event Emitters ---
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
        """Open native color dialog to select user color."""
        curr = self.color_manager.current_color
        rgb = ColorManager.bgr_to_rgb(curr)
        q_color = QColorDialog.getColor(QColor(*rgb), self, "Select Brush Color")
        if q_color.isValid():
            selected_bgr = ColorManager.rgb_to_bgr((q_color.red(), q_color.green(), q_color.blue()))
            self.color_manager.current_color = selected_bgr
            events.color_changed.emit(selected_bgr)

    def _on_undo_clicked(self) -> None:
        # We hook directly to canvas manager's actions inside main window,
        # but emitting canvas_stack_changed triggers an event
        events.status_message.emit("Undo performed", 1000)

    def _on_redo_clicked(self) -> None:
        events.status_message.emit("Redo performed", 1000)

    # --- Slot handlers for inbound events ---
    @pyqtSlot(bool)
    def set_undo_enabled(self, enabled: bool) -> None:
        self.btn_undo.setEnabled(enabled)

    @pyqtSlot(bool)
    def set_redo_enabled(self, enabled: bool) -> None:
        self.btn_redo.setEnabled(enabled)
