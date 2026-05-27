# Air Draw 🎨✋

A real-time virtual drawing application built with Python, OpenCV, and hand tracking.  
Draw in the air using your fingers through your webcam — no mouse, no touchscreen required.

---

# 📌 Features

- ✍️ Draw in the air using hand gestures
- 🖐️ Real-time hand tracking
- 🎨 Multiple drawing colors
- 🧽 Eraser mode
- 📷 Webcam-based interaction
- ⚡ Smooth and responsive drawing experience
- 🪄 Gesture-controlled UI
- 💻 Lightweight and easy to run

---

# 🧠 How It Works

The application uses:

- **OpenCV** for video processing and drawing
- **MediaPipe** for hand detection and landmark tracking
- Finger position tracking to simulate drawing in the air

The webcam captures live video frames, detects your hand landmarks, and tracks fingertip movement to create virtual drawings on the screen.

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Video processing & rendering |
| MediaPipe | Hand tracking & landmark detection |
| NumPy | Frame manipulation & calculations |

---

# 📂 Project Structure

```bash
air-draw/
│
├── main.py                # Main application file
├── requirements.txt       # Dependencies
├── assets/                # Images/icons (if any)
├── utils/                 # Helper modules
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone the repository

```bash
git clone https://github.com/Shadman-Samin/air-draw.git
cd air-draw
```

## 2️⃣ Create virtual environment (recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

```bash
python main.py
```

After launching:

1. Your webcam will open
2. Show your hand to the camera
3. Use finger gestures to draw in the air

---

# ✋ Gesture Controls

| Gesture | Action |
|---|---|
| Index finger up | Draw |
| Index + middle finger up | Selection mode |
| Eraser gesture | Erase drawing |
| Closed hand | Stop drawing |

> Actual gestures may vary depending on implementation.

---

# 📸 Screenshots

Add screenshots here for better presentation.

Example:

```md
![Demo](assets/demo.png)
```

---

# 🚀 Future Improvements

- Save drawings as images
- Adjustable brush size
- Shape drawing support
- Gesture-based undo/redo
- Multi-hand support
- AI-assisted drawing recognition
- Virtual whiteboard mode

---

# 🧪 Requirements

- Python 3.9+
- Webcam
- Good lighting for accurate hand tracking

---

# 📦 Dependencies

Example dependencies:

```txt
opencv-python
mediapipe
numpy
```

Install manually if needed:

```bash
pip install opencv-python mediapipe numpy
```

---

# 🐞 Common Issues

## Webcam not opening

Make sure:
- Webcam permissions are enabled
- No other application is using the camera

## Hand not detected properly

- Improve room lighting
- Keep your hand inside the camera frame
- Avoid cluttered backgrounds

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Added new feature"
```

4. Push to your branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Made by [Shadman Samin](https://github.com/Shadman-Samin)

If you like this project, consider giving it a ⭐ on GitHub.

---

# 🌟 Demo Ideas

Possible use cases:

- Virtual whiteboard
- Teaching & presentations
- Fun drawing application
- Gesture interaction experiments
- Computer vision learning project

---

# 📚 Learning Resources

- [OpenCV Documentation](https://opencv.org/)
- [MediaPipe Documentation](https://developers.google.com/mediapipe)
- [Python Official Website](https://www.python.org/)
