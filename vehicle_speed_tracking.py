import streamlit as st
import cv2
import time
import numpy as np
import tempfile
from sort import *
from ultralytics import YOLO
from PIL import Image

# --- INITIAL CONFIGURATION ---
st.set_page_config(page_title="Vehicle Speed Monitor", layout="wide")

# Calibration Constants
CALIBRATION_DISTANCE = 5  # meters
CALIBRATION_PIXELS = 200   # pixels
PIXEL_TO_METER = CALIBRATION_DISTANCE / CALIBRATION_PIXELS
FRAME_RATE = 30

# Load YOLO Model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Load Class Names
try:
    with open('classes.txt', 'r') as f:
        classnames = f.read().splitlines()
except FileNotFoundError:
    st.error("Error: 'classes.txt' not found. Please ensure it is in the project folder.")
    st.stop()

# --- SIDEBAR & UI ---
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.6)
video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

st.title("🚗 Vehicle Speed Tracking System")
col1, col2 = st.columns([3, 1])

with col1:
    video_placeholder = st.empty()  # Where the video frame goes

with col2:
    st.subheader("Live Logs")
    log_placeholder = st.empty()  # Where the speed logs go
    logs = []

# --- TRACKING LOGIC ---
if video_file is not None:
    # Save uploaded file to temp path
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    tracker = Sort(max_age=30)
    prev_positions = {}
    
    start_btn = st.sidebar.button("Start Tracking")
    stop_btn = st.sidebar.button("Stop Tracking")

    if start_btn:
        while cap.isOpened():
            if stop_btn:
                st.warning("Tracking stopped by user.")
                break
                
            ret, frame = cap.read()
            if not ret:
                st.info("End of video.")
                break

            # Resize for display
            frame = cv2.resize(frame, (800, 450))
            detections = np.empty((0, 5))
            
            # YOLO Detection
            results = model(frame, stream=True)
            for info in results:
                for box in info.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = box.cls[0]
                    label = classnames[int(cls)]

                    if label in ['car', 'truck', 'bus'] and conf > conf_threshold:
                        detections = np.vstack((detections, [int(x1), int(y1), int(x2), int(y2), conf]))

            # Update Tracker
            track_result = tracker.update(detections)
            frame_time = 1 / FRAME_RATE

            for result in track_result:
                x1, y1, x2, y2, obj_id = map(int, result)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                # Speed Calculation
                if obj_id in prev_positions:
                    prev_x, prev_y, _ = prev_positions[obj_id]
                    dist_px = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                    speed_kmph = (dist_px * PIXEL_TO_METER) / frame_time * 3.6
                else:
                    speed_kmph = 0

                prev_positions[obj_id] = (center_x, center_y, time.time())

                # Draw Results
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {obj_id} {speed_kmph:.1f}km/h', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Add to Log
                log_entry = f"ID: {obj_id} | Speed: {speed_kmph:.1f} km/h"
                logs.append(log_entry)
                if len(logs) > 15: logs.pop(0) # Keep only last 15 logs

            # Update Streamlit UI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            log_placeholder.text("\n".join(logs))

        cap.release()