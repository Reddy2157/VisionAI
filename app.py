import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from pathlib import Path

# --- 1. Page Configuration & Theme ---
st.set_page_config(page_title="VisionAI Pro 2026", layout="wide", page_icon="👁️")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Load Deep Learning Model (Cached) ---

def import_optional(name):
    try:
        return __import__(name)
    except ImportError:
        return None


def lazy_load_yolo():
    ultralytics = import_optional('ultralytics')
    if ultralytics is None:
        return None

    YOLO = getattr(ultralytics, 'YOLO', None)
    if YOLO is None:
        return None

    weights = [Path('yolo11n.pt'), Path('yolov8n.pt')]
    for w in weights:
        if w.exists():
            return YOLO(str(w))

    return None


@st.cache_resource
def load_yolo():
    return lazy_load_yolo()


model = load_yolo()
model_available = model is not None

# --- 3. Persistent Session State ---
if 'history' not in st.session_state:
    st.session_state.history = []

def log_detection(label, conf):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.append({"Time": timestamp, "Object": label, "Confidence": f"{conf:.2f}"})

# --- 4. Sidebar Controls ---
webrtc_module = import_optional('streamlit_webrtc')
av = import_optional('av')
webrtc_available = webrtc_module is not None and av is not None

st.sidebar.title("🎮 Control Center")
available_app_modes = ["Image Upload"]
if webrtc_available:
    available_app_modes.append("Live Webcam Stream")
app_mode = st.sidebar.selectbox("Choose Input Source", available_app_modes)

vision_tasks = ["Privacy Face Blur", "Canny Edge (Classical)"]
if model_available:
    vision_tasks.insert(0, "Object Detection (DL)")
task = st.sidebar.radio("Vision Task", vision_tasks)

if not model_available:
    st.sidebar.warning("Object Detection is disabled because ultralytics or the model weights are unavailable.")
if not webrtc_available:
    st.sidebar.warning("Live Webcam Stream is disabled because streamlit-webrtc or av is not installed.")

if st.sidebar.button("🗑️ Clear History"):
    st.session_state.history = []
    st.rerun()

# --- 5. Logic: Image Processing Functions ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_frame(img, task_type):
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if task_type == "Object Detection (DL)":
        if model is None:
            return img_bgr, None
        results = model(img_bgr, conf=0.4)
        return results[0].plot(), results[0].boxes

    elif task_type == "Privacy Face Blur":
        if model is not None:
            results = model(img_bgr, classes=[0])
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_h = int((y2 - y1) * 0.3)
                if y1 + face_h < img_bgr.shape[0]:
                    roi = img_bgr[y1:y1 + face_h, x1:x2]
                    blurred = cv2.GaussianBlur(roi, (99, 99), 30)
                    img_bgr[y1:y1 + face_h, x1:x2] = blurred
            return img_bgr, results[0].boxes

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        for (x, y, w, h) in faces:
            face = img_bgr[y:y+h, x:x+w]
            blurred = cv2.GaussianBlur(face, (99, 99), 30)
            img_bgr[y:y+h, x:x+w] = blurred
        return img_bgr, None

    elif task_type == "Canny Edge (Classical)":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return edges_bgr, None

# --- 6. Main Application UI ---
st.title("👁️ VisionAI Hub: Multi-Model Computer Vision")
st.caption("Integrated Classical CV + Deep Learning + WebRTC Deployment")

if app_mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Run Processing
        output_img, boxes = process_frame(img_array, task)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("AI Result")
            st.image(output_img, channels="BGR", use_container_width=True)
            
        # Logging
        if boxes is not None and model_available:
            for box in boxes:
                log_detection(model.names[int(box.cls[0])], float(box.conf[0]))

else: # Live Webcam Mode
    if not webrtc_available:
        st.warning("Live webcam mode is unavailable because streamlit-webrtc or av is not installed.")
    else:
        st.subheader("📡 Real-time AI Stream")

        class VideoProcessor(webrtc_module.VideoTransformerBase):
            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                processed, _ = process_frame(img_rgb, task)
                return av.VideoFrame.from_ndarray(processed, format="bgr24")

        webrtc_module.webrtc_streamer(
            key="vision-stream",
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

# --- 7. Analytics & Export ---
st.divider()
st.subheader("📊 Detection Analytics")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df.tail(10), use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export Full Report (CSV)", data=csv, file_name="vision_report.csv", mime="text/csv")
else:
    st.info("No detections logged yet. Upload an image or start the webcam!")