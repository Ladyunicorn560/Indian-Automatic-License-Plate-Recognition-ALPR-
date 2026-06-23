# 🚗 Indian ALPR — Automatic License Plate Recognition

A high-performance, real-time Automatic License Plate Recognition (ALPR) system built specifically for **Indian number plates**.

---

## ✨ Features

- 🔍 **YOLO v9** — state-of-the-art license plate detection
- 🔤 **CCT OCR** — fast, accurate character recognition (65+ countries, incl. India)
- 🇮🇳 **Indian plate post-processing** — cleans output to `MHXXABXXXX` format
- 📹 **Threaded IP Camera** — RTSP stream with auto-reconnect (no frame drops)
- 💾 **Auto-save** — annotated images saved to `output/` on every detection
- 📋 **Plate log** — timestamped detection history with CSV export
- 🌐 **Streamlit UI** — clean, dark-mode web interface

---

## 📦 Installation

```bash
pip install "fast-alpr[onnx]"   # CPU
# or
pip install "fast-alpr[onnx-gpu]"   # NVIDIA GPU
```

Additional dependencies:
```bash
pip install streamlit opencv-python-headless numpy pillow pandas
```

---

## 🚀 Run the App

```bash
streamlit run indian_alpr_demo.py --server.port 8502
```

Open **http://localhost:8502** in your browser.

---

## 🗂 Project Structure

```
├── indian_alpr_demo.py   # Streamlit web app (main entry point)
├── camera.py             # Threaded IPCamera class with RTSP auto-reconnect
├── fast_alpr/            # Core ALPR engine (detector + OCR pipeline)
│   ├── alpr.py           # ALPR class — orchestrates detection + OCR
│   ├── base.py           # Abstract base classes
│   ├── default_detector.py
│   └── default_ocr.py
├── assets/               # Sample images
├── output/               # Auto-saved annotated frames (created at runtime)
└── requirements.txt
```

---

## 📹 IP Camera (RTSP)

The app uses a **threaded, non-blocking** camera reader:

```python
from camera import IPCamera

cam = IPCamera("rtsp://admin:password@192.168.1.x:554/Streaming/Channels/101")
if cam.connect():
    frame = cam.get_frame()   # always latest, never blocks
    cam.disconnect()
```

Features:
- Background thread reads frames at camera speed
- Auto-reconnects if the RTSP stream drops
- Thread-safe `get_frame()` via `threading.Lock`

---

## ⚙️ Configuration (Sidebar)

| Setting | Description |
|---|---|
| Detector Model | YOLO v9 model size (tiny / small) |
| OCR Model | CCT model size (xs / s) |
| Confidence Threshold | Minimum score to display a detection |
| Draw Bounding Boxes | Toggle plate outlines on image |
| Clean Indian Plate Format | Strips spaces/symbols, uppercases text |
| Auto-save detections | Saves every annotated frame to `output/` |

---


