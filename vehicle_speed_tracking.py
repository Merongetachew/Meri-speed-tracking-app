import streamlit as st
import cv2
import numpy as np
import tempfile
from sort import * from ultralytics import YOLO

st.set_page_config(page_title="Fast Speed Tracker", layout="wide")

@st.cache_resource
def load_yolo():
    # 'n' (Nano) is the fastest model for web servers
    return YOLO('yolov8n.pt')

model = load_yolo()

st.sidebar.title("Optimization Settings")
video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi"])

# Increase this slider if the video is still "sticking"
skip_frames = st.sidebar.slider("Speed Optimization (Skip Frames)", 1, 10, 3)

view = st.empty()

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    tracker = Sort(max_age=20)
    prev_positions = {}
    
    # Calibration
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    PIXEL_TO_METER = 5 / 200
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        count += 1
        # SKIP FRAMES: Essential for web apps to prevent lagging
        if count % skip_frames != 0:
            continue

        # 1. Resize to 480p (Small is faster for the browser to render)
        frame = cv2.resize(frame, (640, 360)) 
        
        # 2. YOLO Predict (optimized for CPU)
        results = model.predict(frame, conf=0.5, verbose=False)
        
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in [2, 3, 5, 7]: # Car, Motorcycle, Bus, Truck
                    b = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections.append([b[0], b[1], b[2], b[3], conf])

        # 3. Update SORT Tracker
        if len(detections) > 0:
            tracks = tracker.update(np.array(detections))
        else:
            tracks = tracker.update(np.empty((0, 5)))

        # 4. Drawing & Speed Calculation
        for trk in tracks:
            x1, y1, x2, y2, obj_id = map(int, trk)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            if obj_id in prev_positions:
                px, py = prev_positions[obj_id]
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                
                # Speed formula adjusted for frame skipping
                # speed = (distance_px * scale) / (seconds_between_processed_frames) * km_h_conversion
                time_gap = (1 / fps) * skip_frames
                speed = (dist * PIXEL_TO_METER) / time_gap * 3.6
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{obj_id} {int(speed)}km/h", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            prev_positions[obj_id] = (cx, cy)

        # 5. Push frame to web UI
        view.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    st.success("Video processing complete!")
