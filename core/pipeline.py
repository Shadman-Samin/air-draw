"""
Threaded Processing Pipeline worker.

Coordinates OpenCV camera capture, runs the selected tracking engine,
smooths movement inputs via FilterChain, executes GestureRecognizer,
manages Canvas drawing states, composites drawing layers on the camera feed,
and broadcasts results to the UI thread via EventHub signals.
"""

from __future__ import annotations

import time
import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QMutexLocker, QThread, pyqtSlot
from PyQt6.QtGui import QImage

from app.constants import TrackingMode
from canvas.canvas_manager import CanvasManager
from canvas.canvas_renderer import CanvasRenderer
from core.events import events
from filters.filter_chain import FilterChain
from gestures.gesture_recognizer import GestureRecognizer
from gestures.gesture_types import GestureState, GestureType
from settings.settings_manager import SettingsManager
from tracking.color_tracker import ColorTracker
from tracking.hand_tracker import HandTracker
from tracking.tracking_result import Point2D, TrackingResult


class ProcessingPipeline(QThread):
    """
    Background worker QThread for video acquisition and frame processing.

    Decoupling the computer vision loop from the UI thread ensures
    buttery-smooth, low-latency performance (30+ FPS) regardless of load.
    """

    def __init__(self, settings: SettingsManager, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._running = False
        
        # Core CV Engines
        self.hand_tracker = HandTracker(
            max_hands=self.settings.get("hand_tracking.max_num_hands"),
            detection_confidence=self.settings.get("hand_tracking.min_detection_confidence"),
            tracking_confidence=self.settings.get("hand_tracking.min_tracking_confidence"),
        )
        self.color_tracker = ColorTracker(
            hsv_lower=self.settings.get("color_tracking.hsv_lower"),
            hsv_upper=self.settings.get("color_tracking.hsv_upper"),
            min_contour_area=self.settings.get("color_tracking.min_contour_area"),
            max_contour_area=self.settings.get("color_tracking.max_contour_area"),
        )
        
        # State tracking mode
        self.current_mode = TrackingMode.HAND
        self.active_tracker = self.hand_tracker
        
        # Core systems
        self.gesture_recognizer = GestureRecognizer()
        self.filter_chain = FilterChain()
        self.canvas_manager = CanvasManager(
            width=self.settings.get("camera.width"),
            height=self.settings.get("camera.height"),
        )
        self.renderer = CanvasRenderer(
            width=self.settings.get("camera.width"),
            height=self.settings.get("camera.height"),
        )
        
        # Video Capture Device
        self.cap = None
        
        # Thread safety controls
        self._camera_index = self.settings.get("camera.device_index")
        self._target_width = self.settings.get("camera.width")
        self._target_height = self.settings.get("camera.height")
        self._mirror = self.settings.get("camera.mirror")
        self._canvas_mutex = QMutex()
        
        # Drawing session tracking
        self._is_drawing_stroke = False
        
        # Wire settings change listeners
        self._connect_signals()

    def _connect_signals(self) -> None:
        """Register listeners to global settings events."""
        events.color_changed.connect(self.set_brush_color)
        events.brush_size_changed.connect(self.set_brush_size)
        events.eraser_size_changed.connect(self.set_eraser_size)
        events.brush_type_changed.connect(self.set_brush_type)
        events.tracking_mode_changed.connect(self.set_tracking_mode)
        events.canvas_cleared.connect(self.clear_canvas)

    @pyqtSlot(tuple)
    def set_brush_color(self, bgr_color: tuple[int, int, int]) -> None:
        """Slot: update drawing color in canvas manager."""
        self.canvas_manager.brush_color = bgr_color

    @pyqtSlot(int)
    def set_brush_size(self, size: int) -> None:
        """Slot: update brush size."""
        self.canvas_manager.brush_size = size

    @pyqtSlot(int)
    def set_eraser_size(self, size: int) -> None:
        """Slot: update eraser size."""
        self.canvas_manager.eraser_size = size

    @pyqtSlot(str)
    def set_brush_type(self, brush_type: str) -> None:
        """Slot: update brush engine brush type."""
        pass

    @pyqtSlot()
    def undo(self) -> None:
        """Slot: performs drawing undo safely under lock."""
        with QMutexLocker(self._canvas_mutex):
            self.canvas_manager.undo()

    @pyqtSlot()
    def redo(self) -> None:
        """Slot: performs drawing redo safely under lock."""
        with QMutexLocker(self._canvas_mutex):
            self.canvas_manager.redo()

    @pyqtSlot()
    def clear_canvas(self) -> None:
        """Slot: clears canvas safely under lock."""
        with QMutexLocker(self._canvas_mutex):
            self.canvas_manager.clear_canvas()

    @pyqtSlot(str)
    def set_tracking_mode(self, mode_str: str) -> None:
        """Slot: switch active tracking engine thread-safely."""
        if mode_str.lower() == "hand":
            self.current_mode = TrackingMode.HAND
            self.active_tracker = self.hand_tracker
        else:
            self.current_mode = TrackingMode.COLOR
            self.active_tracker = self.color_tracker
        
        # Reset filters to avoid projection jump
        self.filter_chain.reset()
        self._is_drawing_stroke = False
        events.status_message.emit(f"Switched tracking to: {mode_str.upper()}", 2000)

    def stop(self) -> None:
        """Gracefully request thread shutdown."""
        self._running = False

    def run(self) -> None:
        """Main CV loop executed in the background thread."""
        self._running = True
        
        try:
            # Initialize active tracking resources
            self.hand_tracker.start()
            self.color_tracker.start()
            
            # Open video capture device
            self.cap = cv2.VideoCapture(self._camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_height)
            
            # Validate capture device
            if not self.cap.isOpened():
                events.status_message.emit(
                    f"Error: Camera index {self._camera_index} could not be opened.", 5000
                )
                self._running = False
                return

            events.status_message.emit("Capture pipeline running.", 3000)
            
            # Performance/FPS Counters
            last_time = time.perf_counter()
            frame_count = 0
            fps = 0.0
            
            # Warmup delay for camera exposure
            time.sleep(0.3)

            while self._running:
                start_loop_time = time.perf_counter()
                ret, frame = self.cap.read()
                if not ret:
                    # Capture frame failure (e.g. unplugged camera)
                    events.status_message.emit("Error: Frame capture failure.", 2000)
                    time.sleep(0.03)  # don't thrash CPU
                    continue

                # Frame size adjustments if necessary
                h, w = frame.shape[:2]
                if w != self._target_width or h != self._target_height:
                    frame = cv2.resize(frame, (self._target_width, self._target_height))
                    w, h = self._target_width, self._target_height

                # 1. Flip horizontally if requested (default behavior for virtual mirror feel)
                if self._mirror:
                    frame = cv2.flip(frame, 1)

                # 2. Tracking: Feed frame to selected tracker
                timestamp_ms = int(time.perf_counter() * 1000)
                tracking_result = self.active_tracker.process_frame(frame, timestamp_ms)

                # 3. Gesture Recognition
                gesture_state = GestureState(gesture=GestureType.PAUSE)
                raw_pt = None
                smoothed_pt = None

                if tracking_result.has_hands:
                    hand = tracking_result.primary_hand
                    if hand is not None:
                        # In HAND mode, GestureRecognizer processes standard hand state
                        # In COLOR mode, HandState contains virtual hand with points clustered on centroid,
                        # which triggers GestureType.DRAW by default because the fingers are "closed" (clustered).
                        if self.current_mode == TrackingMode.HAND:
                            gesture_state = self.gesture_recognizer.update(tracking_result)
                        else:
                            # In color mode, we are always drawing when object is visible
                            gesture_state = GestureState(gesture=GestureType.DRAW)
                        
                        raw_pt = hand.index_tip

                # 4. Filter coordinate and update Canvas Drawing
                if raw_pt is not None:
                    # Apply smoothing filter chain to index tip position
                    smoothed_pt = self.filter_chain.process(raw_pt)
                else:
                    self.filter_chain.reset()

                # Perform drawing operations and alpha layer rendering under a thread safety lock
                with QMutexLocker(self._canvas_mutex):
                    # Delegate drawing actions to canvas manager's robust state machine
                    self.canvas_manager.handle_input(smoothed_pt or raw_pt, gesture_state)

                    # 5. Composite Drawing Layers over Camera Feed
                    overlay_frame = self.renderer.render_frame(frame, self.canvas_manager.layer_stack)

                # Query state changes and broadcast undo/redo availability signals
                events.undo_available.emit(len(self.canvas_manager._undo_stack) > 0)
                events.redo_available.emit(len(self.canvas_manager._redo_stack) > 0)

                # 6. Draw visual annotations (Landmarks, pointer cursor, status HUD) on frame
                self._render_hud_overlays(overlay_frame, tracking_result, gesture_state, smoothed_pt or raw_pt)

                # 7. FPS Calculations
                frame_count += 1
                now = time.perf_counter()
                elapsed = now - last_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    events.fps_updated.emit(fps)
                    frame_count = 0
                    last_time = now

                # 8. Emit completed frame back to UI main thread
                # Convert BGR (OpenCV default) to RGB and create QImage in worker thread
                rgb_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                q_img = QImage(
                    rgb_frame.data,
                    w,
                    h,
                    ch * w,
                    QImage.Format.Format_RGB888
                ).copy()
                
                events.frame_processed.emit(q_img, tracking_result, gesture_state)

                # 9. Frame-rate limiter (target ~30-60 fps depending on settings)
                target_dt = 1.0 / self.settings.get("camera.fps")
                loop_duration = time.perf_counter() - start_loop_time
                sleep_time = max(0.001, target_dt - loop_duration)
                time.sleep(sleep_time)
        except Exception as e:
            print(f"[Error] Exception in background processing pipeline: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                
            self.hand_tracker.stop()
            self.color_tracker.stop()
            events.status_message.emit("Capture pipeline stopped.", 2000)

    def _render_hud_overlays(
        self,
        frame: np.ndarray,
        tracking_result: TrackingResult,
        gesture_state: GestureState,
        active_point: Point2D | None,
    ) -> None:
        """
        Draws visual feedback overlays directly onto BGR frame:
        - Hand landmarks (in Hand mode)
        - Color object tracking boundary boxes (in Color mode)
        - Customized hover cursors indicating brush color/state
        - Text indicators for the active gesture mode
        """
        h, w = frame.shape[:2]

        # Draw tracking guides
        if tracking_result.has_hands:
            hand = tracking_result.primary_hand
            
            # --- Hand Landmarks Rendering ---
            if self.current_mode == TrackingMode.HAND and self.settings.get("ui.show_landmarks"):
                # Connect key landmarks with color schemes based on gesture
                color_map = {
                    GestureType.DRAW: (50, 255, 50),     # Bright Green
                    GestureType.ERASE: (50, 50, 255),    # Red
                    GestureType.CURSOR: (255, 200, 50),  # Cyan
                    GestureType.PAUSE: (150, 150, 150),  # Grey
                    GestureType.CLEAR: (255, 50, 255),   # Magenta
                }
                l_color = color_map.get(gesture_state.gesture, (255, 255, 255))
                
                # Draw joint segments
                self._draw_hand_skeleton(frame, hand, l_color)

            # --- Color Mode Guide Rendering ---
            elif self.current_mode == TrackingMode.COLOR:
                # Draw a bounding indicator or circle around target object
                if active_point is not None:
                    cx, cy = active_point.as_int_tuple()
                    cv2.circle(frame, (cx, cy), 12, (255, 255, 50), 2, cv2.LINE_AA)
                    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

        # --- Interactive Hover Pointer Cursors ---
        if active_point is not None:
            cx, cy = active_point.as_int_tuple()
            if 0 <= cx < w and 0 <= cy < h:
                if gesture_state.gesture == GestureType.DRAW:
                    # Draw a cursor that previews brush color and size
                    bgr_col = self.canvas_manager.brush_color
                    brush_size = self.canvas_manager.brush_size
                    # Preview outline
                    cv2.circle(frame, (cx, cy), max(brush_size // 2, 2), bgr_col, -1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), max(brush_size // 2, 2) + 2, (255, 255, 255), 1, cv2.LINE_AA)
                elif gesture_state.gesture == GestureType.ERASE:
                    # Eraser circle guide
                    er_size = self.canvas_manager.eraser_size
                    cv2.circle(frame, (cx, cy), er_size, (50, 50, 255), 1, cv2.LINE_AA)
                    cv2.drawMarker(frame, (cx, cy), (50, 50, 255), cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
                elif gesture_state.gesture == GestureType.CURSOR:
                    # Subtle cyan reticle
                    cv2.circle(frame, (cx, cy), 8, (255, 200, 50), 1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 2, (255, 200, 50), -1, cv2.LINE_AA)

        # --- Gesture Status Banner (Bottom Left) ---
        if self.settings.get("ui.show_gesture_overlay"):
            status_text = f"MODE: {gesture_state.gesture.name}"
            # Text background panel
            cv2.rectangle(frame, (15, h - 45), (200, h - 15), (20, 20, 20), -1)
            cv2.rectangle(frame, (15, h - 45), (200, h - 15), (100, 100, 100), 1)
            
            # Status colors
            status_colors = {
                GestureType.DRAW: (0, 255, 0),       # Green
                GestureType.ERASE: (0, 0, 255),      # Red
                GestureType.CURSOR: (255, 255, 0),   # Cyan
                GestureType.PAUSE: (150, 150, 150),  # Grey
                GestureType.CLEAR: (255, 0, 255),    # Magenta
            }
            txt_color = status_colors.get(gesture_state.gesture, (255, 255, 255))
            cv2.putText(
                frame,
                status_text,
                (25, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                txt_color,
                2,
                cv2.LINE_AA,
            )

    def _draw_hand_skeleton(self, frame: np.ndarray, hand: object, color: tuple[int, int, int]) -> None:
        """Draw skeleton links between key landmarks."""
        # Standard MediaPipe connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (9, 10), (10, 11), (11, 12),         # Middle
            (13, 14), (14, 15), (15, 16),        # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17)            # Palm
        ]
        
        # Draw lines
        for start_idx, end_idx in connections:
            p1 = hand.get_landmark(start_idx)
            p2 = hand.get_landmark(end_idx)
            if p1 and p2:
                cv2.line(frame, p1.as_int_tuple(), p2.as_int_tuple(), color, 2, cv2.LINE_AA)

        # Draw joints
        for pt in hand.landmarks:
            cv2.circle(frame, pt.as_int_tuple(), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt.as_int_tuple(), 5, color, 1, cv2.LINE_AA)
