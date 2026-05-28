"""
Main Application Window.

Combines all custom UI elements (Toolbar, Video Canvas Widget, Status Bar)
and manages the lifetime of the background processing pipeline thread.
"""

from __future__ import annotations

import os
from pathlib import Path
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QSplitter, QWidget

from core.events import events
from core.pipeline import ProcessingPipeline
from settings.settings_manager import SettingsManager
from ui.status_bar import SystemStatusBar
from ui.toolbar import DrawingToolbar
from ui.video_widget import VideoWidget


class MainWindow(QMainWindow):
    """
    Main shell container for the Air Draw application.

    Orchestrates the layout of toolbar, video canvas, and status logs.
    Handles startup configuration and clean thread teardown on exit.
    """

    def __init__(self, settings: SettingsManager):
        super().__init__()
        self.settings = settings
        
        self.setWindowTitle("Air Draw (Virtual Pen) — Enterprise Studio")
        self.resize(1500, 850)
        self.setMinimumSize(1100, 650)
        
        # Load Theme
        self._load_stylesheet()
        
        # 1. Background capture thread pipeline
        self.pipeline = ProcessingPipeline(self.settings)
        
        # 2. Construct layout & subcomponents
        self._init_ui()
        
        # 3. Connect thread hooks
        self._connect_signals()
        
        # 4. Start background capture
        self.pipeline.start()

    def _init_ui(self) -> None:
        """Construct side panels and main canvas splitting sections."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Splitter to allow user dragging sizing
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        
        # Left sidebar toolbar panel
        self.toolbar = DrawingToolbar(self)
        self.toolbar.setMaximumWidth(380)
        self.toolbar.setMinimumWidth(280)
        
        # Right high-performance video canvas display
        self.video_widget = VideoWidget(self)
        
        # Add widgets to horizontal splitter
        splitter.addWidget(self.toolbar)
        splitter.addWidget(self.video_widget)
        
        # Default ratio 25% sidebar, 75% canvas
        splitter.setSizes([320, 1100])
        
        layout.addWidget(splitter)
        
        # System-wide Status Bar
        self.status_bar = SystemStatusBar(self)
        self.setStatusBar(self.status_bar)

    def _connect_signals(self) -> None:
        """Wire UI controls directly to background pipeline slots thread-safely."""
        # Wiring History controllers
        self.toolbar.btn_undo.clicked.connect(self.pipeline.undo)
        self.toolbar.btn_redo.clicked.connect(self.pipeline.redo)

    def _load_stylesheet(self) -> None:
        """Load and apply premium dark theme stylesheet."""
        qss_path = Path(__file__).parent / "styles" / "dark_theme.qss"
        if qss_path.exists():
            try:
                with open(qss_path, "r", encoding="utf-8") as f:
                    self.setStyleSheet(f.read())
            except Exception as e:
                print(f"[Error] Failed to load stylesheet: {e}")
        else:
            print(f"[Warning] Stylesheet not found at: {qss_path}")

    def closeEvent(self, event) -> None:
        """
        Intercept close event to stop background camera thread gracefully.
        Prevents app hang ups or memory leak dump on program exit.
        """
        events.status_message.emit("Shutting down pipelines...", 2000)
        
        # Shutdown capture thread and wait
        self.pipeline.stop()
        self.pipeline.wait(3000)  # Wait up to 3 seconds for safe thread join
        
        # Accept closure
        event.accept()
