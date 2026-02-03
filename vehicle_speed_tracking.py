import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from sort import * # Ensure sort.py is in your GitHub folder
from ultralytics import YOLO
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(page_title="Vehicle Speed Monitor", layout="wide")

# Constants
CALIBRATION_DISTANCE = 5 
CALIBRATION_PIXELS = 200 
PIXEL_TO_METER = CALIBRATION_DISTANCE / CALIBRATION_PIXELS

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Load classes
try:
    with open('classes.txt', 'r') as f:
        classnames = f.read().splitlines()
except:
    classnames = ["person", "bicycle", "car", "motorcycle", "bus", "truck"] # Fallback

# --- SESSION STATE ---
# This prevents the app from "forgetting" what it's doing
if 'tracking_active' not in st.session_state:
    st.session_state.tracking_active = False

def start_btn(): st.session_state.tracking_active = True
def stop_btn(): st.session_state.tracking_active = False

# --- UI LAYOUT ---
st.title("🚗 Vehicle Speed Monitoring System")

sidebar = st.sidebar
video_file = sidebar.file_uploader("Step 1: Upload Video", type=["mp4", "avi", "mov"])
conf_thresh = sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

col_video, col_logs = st.columns([3, 1])

with col_video:
    video_placeholder = st.empty()  # This replaces the Tkinter Canvas

with col_logs:
    st.subheader("Live Speed Logs")
    log_placeholder = st.empty()
    logs_list = []

# --- CONTROL BUTTONS ---
c1, c2 = sidebar.columns(2)
c1.button("▶ Start", on_click=start_btn, use_container_width=True)
c2.button("⏹ Stop", on_click=stop_btn, use_container_width=True)

# --- PROCESSING LOOP ---
if video_file is not None and st.session_state.tracking_active:
    # Save upload to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_time = 1 / fps
    
    tracker = Sort(max_age=30)
    prev_positions = {}

    while cap.isOpened() and st.session_state.tracking_active:
        ret, frame = cap.read()
        if not ret:
            st.info("Video processing complete.")
            break

        # Resize for web performance
        frame = cv2.resize(frame, (800, 450))
        
        # YOLO Detection
        results = model(frame, stream=True, conf=conf_thresh, verbose=False)
        detections = np.empty((0, 5))

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                label = classnames[cls] if cls < len(classnames) else "vehicle"
                
                if label in ['car', 'truck', 'bus']:
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        # Update SORT Tracker
        track_result = tracker.update(detections)

        for result in track_result:
            x1, y1, x2, y2, obj_id = map(int, result)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Speed Calculation
            if obj_id in prev_positions:
                px, py = prev_positions[obj_id]
                dist_px = np.sqrt((cx - px)**2 + (cy - py)**2)
                speed_kmph = (dist_px * PIXEL_TO_METER) / frame_time * 3.6
            else:
                speed_kmph = 0
            
            prev_positions[obj_id] = (cx, cy)

            # Drawing (OpenCV)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{obj_id} {int(speed_kmph)}km/h", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Update Logs
            if speed_kmph > 1: # Only log moving vehicles
                log_entry = f"ID: {obj_id} | Speed: {int(speed_kmph)} km/h"
                if log_entry not in logs_list:
                    logs_list.insert(0, log_entry)
                    if len(logs_list) > 10: logs_list.pop()

        # Update the Web UI
        video_placeholder.image(frame, channels="BGR", use_container_width=True)
        log_placeholder.text("\n".join(logs_list))

    cap.release()
