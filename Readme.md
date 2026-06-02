# Air Draw 🎨✋

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-red.svg)](https://developers.google.com/mediapipe)

**Air Draw** is a real-time virtual drawing application that leverages advanced hand tracking to turn your webcam into a canvas. Draw in 3D space (projected to 2D) using intuitive hand gestures — no physical mouse or touchscreen required.

Built with a high-performance background processing pipeline, multi-hand support, and intelligent gesture recognition.

---

## ✨ Key Features

- **🚀 High Performance:** Decoupled background CV pipeline ensures a smooth 60 FPS UI experience.
- **🙌 Multi-Hand Support:** Draw with two hands simultaneously. Each hand has its own filter chain, gesture state, and color.
- **📐 Geometric Tools:** Integrated shape engine for drawing precise **Lines**, **Rectangles**, **Circles**, and **Arrows**.
- **🎯 Intelligent Smoothing:** Hybrid filtering using **Kalman Filters**, **EMA (Exponential Moving Average)**, and **Deadzone thresholds** to eliminate fingertip jitter.
- **✨ Gesture-Driven UI:** Control the entire app without touching the keyboard — undo, redo, and clear canvas are all gesture-activated.
- **📥 Persistent Settings:** All configurations (brush size, colors, tracking sensitivity) are saved automatically to your user profile.
- **🎬 Whiteboard Mode:** Toggle between "Mirror" mode and a professional "Whiteboard" overlay.

---

## ✋ Gesture Controls

The app distinguishes between your **Primary** hand (full control) and **Secondary** hand (drawing only).

### 🥇 Primary Hand (Master Control)
| Gesture | Action |
|:---:|---|
| **☝️ Index Up** | **Draw / Set Shape Anchor** |
| **✌️ Index + Middle** | **Cursor Mode** (Move without drawing) |
| **🖐️ Open Palm** | **Pause Tracking** |
| **✊ Closed Fist** | **Eraser Mode** |
| **🤟 Three Fingers** | **Undo** (Single-shot) |
| **🖖 Four Fingers** | **Redo** (Single-shot) |
| **⏳ Palm Hold** | **Clear Canvas** (Sustained hold) |

### 🥈 Secondary Hand (Multi-Hand Mode)
| Gesture | Action |
|:---:|---|
| **☝️ Index Up** | **Secondary Draw** (Blue by default) |
| **✊ Closed Fist** | **Erase** |
| **🖐️ Any Other** | **Cursor Mode** |

---

## 🛠️ Tech Stack

- **MediaPipe Tasks:** Industry-leading `HandLandmarker` for sub-millisecond hand landmark detection.
- **OpenCV:** Core image processing, frame manipulation, and AA (Anti-Aliased) rendering.
- **PyQt6:** Robust desktop framework for the main window, custom widgets, and event dispatching.
- **NumPy & SciPy:** vectorized math for coordinate scaling, distance calculations, and filter implementations.

---

## 📂 Architecture Overview

```bash
air-draw/
├── main.py                  # Entry point: Initializes app and spawns worker threads
├── core/                    # The "Brain": Processing pipeline & event hub
│   ├── pipeline.py          # Background CV loop (QThread)
│   └── multi_hand_controller # Routes tracking data to individual hand sessions
├── canvas/                  # The "Paper": Layer management & renderer
│   ├── canvas_manager.py    # Per-hand state machine & undo/redo logic
│   └── layer_stack.py       # Supports multiple transparent BGRA layers
├── drawing/                 # The "Pen": Brush engine & shape math
├── tracking/                # The "Eyes": HandLandmarker & ColorTracker wrappers
├── gestures/                # The "Logic": Finite State Machine for gesture detection
└── filters/                 # The "Steady Hand": Kalman & EMA filter implementations
```

---

## ⚙️ Installation & Setup

### 1. Requirements
- **Python 3.12+**
- **Webcam** (720p+ recommended)
- **hand_landmarker.task** (MediaPipe Model)

### 2. Quick Start
```bash
# 1. Clone repository
git clone https://github.com/Shadman-Samin/air-draw.git
cd air-draw

# 2. Download MediaPipe model
# Download 'hand_landmarker.task' from Google's MediaPipe documentation
# and place it in the root folder.

# 3. Setup Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 4. Install Dependencies
pip install -r requirements.txt

# 5. Run
python main.py
```

---

## 🐞 Troubleshooting

- **Low FPS?** Ensure your room is well-lit. Tracking accuracy drops significantly in low light.
- **Not detecting hand?** Keep your hand within the 640x480 center tracking area for best results.
- **Camera error?** Verify that no other apps (Zoom, Teams) are using the webcam.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 👨‍💻 Author

**Shadman Samin** - [GitHub Profile](https://github.com/Shadman-Samin)

---

*If you find this project interesting, consider giving it a ⭐ on GitHub!*
