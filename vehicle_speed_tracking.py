import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
from PIL import Image
import time
import tempfile

# --- 1. Configuration & Constants ---
CALIBRATION_DISTANCE = 5  # meters
CALIBRATION_PIXELS = 200  # pixels
PIXEL_TO_METER = CALIBRATION_DISTANCE / CALIBRATION_PIXELS
FRAME_RATE = 30

st.set_page_config(page_title="Vehicle Speed Tracker", layout="wide")

# --- 2. Model Loading (Cached) ---
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

@st.cache_resource
def get_class_names():
    # Fallback if classes.txt is missing
    try:
        with open('classes.txt', 'r') as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"]

model = load_yolo_model()
classnames = get_class_names()

# --- 3. Sidebar UI ---
st.sidebar.title("Control Panel")
video_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
run_tracking = st.sidebar.checkbox("Start Tracking")

# --- 4. Main Interface ---
st.title("🚀 Vehicle Speed Monitoring System")
col1, col2 = st.columns([3, 1])

with col1:
    st_frame = st.empty()  # Placeholder for video

with col2:
    st.subheader("Live Logs")
    log_container = st.empty() # Placeholder for logs

# --- 5. Processing Logic ---
if video_file and run_tracking:
    # Handle the file upload via a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    tracker = Sort(max_age=30)
    prev_positions = {}
    logs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("End of video reached.")
            break

        # Processing Resize
        frame = cv2.resize(frame, (800, 450))
        
        # YOLO Inference
        results = model(frame, stream=True, verbose=False)
        detections = np.empty((0, 5))

        for info in results:
            boxes = info.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                if cls < len(classnames):
                    object_detected = classnames[cls]
                    if object_detected in ['car', 'truck', 'bus'] and conf > 0.6:
                        detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        # SORT Tracking
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

            # Draw Visuals
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{obj_id} {speed_kmph:.1f}km/h", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Update Logs (Keep only last 10 for performance)
            log_entry = f"ID: {obj_id} | Speed: {speed_kmph:.1f} km/h"
            logs.insert(0, log_entry)
            logs = logs[:10]

        # Update UI components
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
        log_container.code("\n".join(logs))

    cap.release()
else:
    st.info("Please upload a video and check 'Start Tracking' in the sidebar.")
