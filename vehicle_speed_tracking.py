import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import time
import tempfile

# --- Setup ---
st.set_page_config(page_title="Vehicle Speed Tracker", layout="wide")
st.title("🚗 Live Vehicle Tracking & Speed Estimation")

# Calibration
CALIBRATION_DISTANCE = 5 
CALIBRATION_PIXELS = 200 
PIXEL_TO_METER = CALIBRATION_DISTANCE / CALIBRATION_PIXELS
FRAME_RATE = 30

# Initialize Model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()
tracker = Sort(max_age=30)

# UI Elements
video_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
start_btn = st.sidebar.button("Start Processing")
stop_btn = st.sidebar.button("Stop")

# This placeholder is CRITICAL for the video effect
container = st.empty() 
log_placeholder = st.sidebar.empty()

if video_file and start_btn:
    # 1. Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    prev_positions = {}

    # 2. The Video Loop
    while cap.isOpened():
        if stop_btn:
            break
            
        ret, frame = cap.read()
        if not ret:
            st.write("Video Processing Complete.")
            break

        # Resize for web performance
        frame = cv2.resize(frame, (960, 540))
        results = model(frame, stream=True, verbose=False)
        
        detections = np.empty((0, 5))
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Filter for vehicles (car=2, bus=5, truck=7 in COCO)
                if cls in [2, 3, 5, 7] and conf > 0.5:
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        # Update Tracker
        tracks = tracker.update(detections)
        
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Calculate Speed
            if track_id in prev_positions:
                px, py = prev_positions[track_id]
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                speed = (dist * PIXEL_TO_METER) * FRAME_RATE * 3.6
                
                cv2.putText(frame, f"ID {track_id}: {int(speed)} km/h", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            prev_positions[track_id] = (cx, cy)

        # 3. DISPLAY THE FRAME
        # Convert BGR (OpenCV) to RGB (Streamlit/PIL)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        container.image(frame, channels="RGB", use_container_width=True)

        # Small sleep to match frame rate (optional, helps UI stability)
        # time.sleep(0.01) 

    cap.release()
