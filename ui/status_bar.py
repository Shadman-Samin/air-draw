"""
Custom Status Bar component.

Displays system log/status messages, active FPS values, and system state.
Integrates with the global EventHub to react to system-wide signals.
"""

from __future__ import annotations

from PyQt6.QtCore import QTimer, pyqtSlot
from PyQt6.QtWidgets import QLabel, QStatusBar

from core.events import events


class SystemStatusBar(QStatusBar):
    """
    Polished status bar showing system status updates, CV framerates, and errors.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("status_panel")
        
        # UI Subcomponents
        self.status_label = QLabel("System Initialized.")
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: #818cf8; font-weight: bold; margin-right: 12px;")
        
        # Add labels to layout
        self.addWidget(self.status_label, 1)
        self.addPermanentWidget(self.fps_label)
        
        # Fade timer for status messages
        self._fade_timer = QTimer(self)
        self._fade_timer.setSingleShot(True)
        self._fade_timer.timeout.connect(self._on_fade_timeout)
        
        # Connect signals
        self._connect_signals()

    def _connect_signals(self) -> None:
        """Register listeners to global system events."""
        events.status_message.connect(self.show_message)
        events.fps_updated.connect(self.update_fps)

    @pyqtSlot(str, int)
    def show_message(self, message: str, timeout_ms: int = 3000) -> None:
        """
        Slot: Show a temporary status update that fades out.
        """
        self.status_label.setText(message)
        
        if timeout_ms > 0:
            self._fade_timer.stop()
            self._fade_timer.start(timeout_ms)

    @pyqtSlot(float)
    def update_fps(self, fps: float) -> None:
        """
        Slot: Update the permanent FPS counter label.
        """
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _on_fade_timeout(self) -> None:
        """Revert temporary message back to default active mode."""
        self.status_label.setText("Active.")
