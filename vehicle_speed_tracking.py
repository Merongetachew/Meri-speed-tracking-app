import streamlit as st
import cv2
import numpy as np
import tempfile
from sort import * # Ensure sort.py is in your GitHub
from ultralytics import YOLO

st.set_page_config(page_title="Speed Monitor", layout="wide")

@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

model = load_yolo()

st.title("Vehicle Tracker Debug Version")

uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi"])
run_tracking = st.sidebar.checkbox("Run Tracking")

video_display = st.empty()
status_text = st.sidebar.empty()

if uploaded_file and run_tracking:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    tracker = Sort(max_age=30)
    prev_positions = {}
    
    # Constants
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    PIXEL_TO_METER = 5 / 200

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Resize small for web speed
        frame = cv2.resize(frame, (640, 360))
        
        # 2. Simple YOLO Detect
        results = model.predict(frame, conf=0.4, verbose=False)
        
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in [2, 3, 5, 7]: # Car, Motorcycle, Bus, Truck
                    b = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections.append([b[0], b[1], b[2], b[3], conf])

        # 3. Safe Tracker Update
        try:
            if len(detections) > 0:
                tracks = tracker.update(np.array(detections))
            else:
                tracks = tracker.update(np.empty((0, 5)))
        except Exception as e:
            st.error(f"Tracker Error: {e}")
            tracks = []

        # 4. Speed Logic
        for trk in tracks:
            x1, y1, x2, y2, obj_id = map(int, trk)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            speed = 0
            if obj_id in prev_positions:
                px, py = prev_positions[obj_id]
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                speed = (dist * PIXEL_TO_METER) * fps * 3.6
            
            prev_positions[obj_id] = (cx, cy)

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{obj_id} {int(speed)}km/h", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 5. Force Update
        video_display.image(frame, channels="BGR")
        status_text.text(f"Processing... (FPS: {int(fps)})")

    cap.release()
