# VisionAI Pro 2026

A Streamlit-based computer vision dashboard for image upload, privacy-preserving face blur, edge detection, and optional YOLO object detection.

## Features

- Image upload processing
- Privacy face blur using classical OpenCV or YOLO face cropping
- Canny edge detection
- Optional live webcam stream via `streamlit-webrtc`
- Detection logging and CSV export
- Lazy loading of YOLO model weights for graceful fallback

## Requirements

- Python 3.10+
- `streamlit`
- `opencv-python-headless`
- `pandas`
- `numpy`
- `Pillow`
- `ultralytics` (optional for YOLO object detection)
- `streamlit-webrtc` and `av` (optional for live webcam streaming)

## Installation

```bash
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

## Notes

- If `ultralytics` or the YOLO weight file is unavailable, object detection is disabled and the app falls back to classical CV.
- Live webcam mode requires `streamlit-webrtc` and `av`.

## GitHub remote setup

If you have not already configured a remote, run:

```bash
git remote add origin https://github.com/<your-user>/<your-repo>.git
```

Then push the repository:

```bash
git push -u origin master
```
