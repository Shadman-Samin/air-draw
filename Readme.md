# Air Draw

Real-time virtual drawing application using hand tracking via webcam.  
Draw in the air with your fingers — no mouse, no touchscreen required.

Built with **Python**, **MediaPipe Tasks** (HandLandmarker), **OpenCV**, and **PyQt6**.

---

## Features

- Hand tracking via webcam with up to **2 hands** simultaneously
- Multi-hand drawing — each hand draws independently with its own color
- Shape tools — line, rectangle, circle, arrow
- Eraser mode
- Gesture-controlled undo / redo / clear canvas
- Adjustable brush size and color
- Smoothing filters (Kalman, EMA, deadzone) for jitter-free strokes
- Virtual whiteboard mode
- Real-time FPS display
- Persistent settings saved to `~/.airdraw/settings.json`

---

## Project Structure

```
air-draw/
├── main.py                  # Entry point
├── app/                     # Application bootstrap and constants
├── canvas/                  # Drawing canvas, layer stack, rendering
├── core/                    # Processing pipeline, multi-hand controller, events
├── drawing/                 # Brush engine, stroke builder, shape renderer
├── filters/                 # Smoothing filters (Kalman, EMA, deadzone)
├── gestures/                # Gesture recognition per hand
├── settings/                # Persistent JSON settings manager
├── tracking/                # Hand tracker (MediaPipe) and color tracker
├── ui/                      # PyQt6 widgets (main window, toolbar, video, status)
└── tests/                   # Accuracy, performance, and shape tests
```

---

## Installation

### 1. Clone

```bash
git clone https://github.com/Shadman-Samin/air-draw.git
cd air-draw
```

### 2. Download MediaPipe model

Place `hand_landmarker.task` in the project root.  
Download from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models).

### 3. Create virtual environment (recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

- Show your hand to the camera
- Use index finger to draw
- Switch tools via the toolbar
- Toggle multi-hand for simultaneous drawing

---

## Gesture Controls

### Primary hand (full gesture set)

| Gesture | Action |
|---|---|
| Index finger up | Draw / shape anchor |
| Index + middle up | Cursor (move, no drawing) |
| Open palm (all fingers) | Pause |
| Closed fist | Eraser mode |
| Three fingers | Undo (single-shot) |
| Four fingers | Redo (single-shot) |
| Palm hold (sustained) | Clear canvas |

### Secondary hand (multi-hand mode only)

| Gesture | Action |
|---|---|
| Index finger up | Draw (secondary color) |
| Closed fist | Erase |
| Any other | Cursor |

---

## Shape Tools

Select from the toolbar: **line**, **rectangle**, **circle**, **arrow**.

1. Point with index finger to set the anchor
2. Move your finger to define the shape
3. Release (change gesture) to commit

---

## Tech Stack

| Component | Library |
|---|---|
| Hand tracking | MediaPipe Tasks HandLandmarker |
| Video processing | OpenCV |
| GUI | PyQt6 |
| Numerical | NumPy, SciPy |
| Image export | Pillow |

---

## Requirements

- Python 3.12+
- Webcam
- Good lighting for accurate tracking

---

## License

MIT License
