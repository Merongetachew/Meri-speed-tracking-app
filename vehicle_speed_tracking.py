import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import tempfile

# --- Setup ---
st.set_page_config(page_title="Vehicle Speed Tracker", layout="wide")
st.title("🚗 Live Vehicle Tracking & Speed Estimation")

# Calibration & Limits
CALIBRATION_DISTANCE = 5 
CALIBRATION_PIXELS = 200 
PIXEL_TO_METER = CALIBRATION_DISTANCE / CALIBRATION_PIXELS
FRAME_RATE = 30
SPEED_LIMIT = 80  # Speed limit in km/h

# Initialize Model
@st.cache_resource
def load_model():
    # 'yolov8n.pt' is the lightest and fastest model
    return YOLO('yolov8n.pt')

model = load_model()
tracker = Sort(max_age=30)

# UI Elements
video_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
start_btn = st.sidebar.button("Start Processing")

# Display placeholders
container = st.empty() 

if video_file and start_btn:
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    prev_positions = {}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.success("Video Processing Complete.")
            break

        frame_count += 1
        
        # 1. REDUCE RESOLUTION (640x360 is the best for performance)
        frame = cv2.resize(frame, (640, 360))

        # 2. FRAME SKIPPING (Process AI every 2nd frame to save CPU)
        if frame_count % 2 == 0:
            # imgsz=320 makes the AI detection even faster
            results = model(frame, stream=True, verbose=False, imgsz=320)
            
            detections = np.empty((0, 5))
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Filter for vehicles
                    if cls in [2, 3, 5, 7] and conf > 0.4:
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
                    
                    # Multiply by 2 because we are skipping every other frame
                    speed = (dist * PIXEL_TO_METER) * (FRAME_RATE / 2) * 3.6
                    
                    # Speed Limit Logic
                    color = (0, 255, 0) # Green
                    if speed > SPEED_LIMIT:
                        color = (255, 0, 0) # Red for speeders
                        label = f"OVER LIMIT: {int(speed)} km/h"
                    else:
                        label = f"ID {track_id}: {int(speed)} km/h"
                    
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                prev_positions[track_id] = (cx, cy)

        # 3. DISPLAY
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        container.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
