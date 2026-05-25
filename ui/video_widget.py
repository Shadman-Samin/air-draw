"""
Video widget for rendering camera feed and composited canvas.

Converts OpenCV BGR image arrays into QImages and renders them
efficiently using QPainter, preserving aspect ratio and scaling.
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QMutex, QMutexLocker, pyqtSlot
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QWidget

from core.events import events


class VideoWidget(QWidget):
    """
    Custom widget for drawing high-performance processed video frames.

    Features:
    - Thread-safe frame caching via QMutex
    - Preserve aspect ratio rendering with centering
    - Fallback placeholder screen when camera is inactive
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        
        # Pixmap cache and thread protection
        self._pixmap = QPixmap()
        self._mutex = QMutex()
        
        # Connect capture pipeline frame signals
        events.frame_processed.connect(self.update_frame)

    @pyqtSlot(QImage, object, object)
    def update_frame(self, q_img: QImage, tracking_result: object, gesture_state: object) -> None:
        """
        Slot: Called from CV worker thread with processed frame data.
        Converts thread-safe QImage to QPixmap and triggers redraw.
        """
        # Convert to Pixmap in UI thread
        pixmap = QPixmap.fromImage(q_img)

        # Thread-safe write to pixmap cache
        with QMutexLocker(self._mutex):
            self._pixmap = pixmap

        # Request repaint on main GUI thread
        self.update()

    def paintEvent(self, event) -> None:
        """Draw the latest cached pixmap, preserving aspect ratio."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        
        # Lock mutex to read pixmap cache safely
        with QMutexLocker(self._mutex):
            pixmap = self._pixmap

        if pixmap.isNull():
            # Draw elegant standby/idle screen
            self._draw_standby_screen(painter)
            return

        # Calculate bounding rect preserving aspect ratio
        widget_w, widget_h = self.width(), self.height()
        pix_w, pix_h = pixmap.width(), pixmap.height()
        
        # Scale factor
        scale = min(widget_w / pix_w, widget_h / pix_h)
        new_w = int(pix_w * scale)
        new_h = int(pix_h * scale)
        
        # Center coordinates
        x = (widget_w - new_w) // 2
        y = (widget_h - new_h) // 2
        
        # Draw background black borders
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        
        # Draw frame scaled
        painter.drawPixmap(x, y, new_w, new_h, pixmap)

    def _draw_standby_screen(self, painter: QPainter) -> None:
        """Renders an attractive standby indicator when camera capture is offline."""
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        
        # Draw glowing retro overlay
        font = painter.font()
        font.setPointSize(16)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.darkGray)
        
        message = "AWAITING CAMERA PIPELINE ACTIVE..."
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            message
        )
