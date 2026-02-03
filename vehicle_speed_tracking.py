import streamlit as st
import cv2
import numpy as np
import tempfile
from sort import *
from ultralytics import YOLO

# --- APP SETUP ---
st.set_page_config(page_title="Speed Tracker", layout="wide")

@st.cache_resource
def get_model():
    return YOLO('yolov8n.pt') # Using Nano for speed

model = get_model()

# --- UI ---
st.sidebar.header("Settings")
video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi"])
run_app = st.sidebar.checkbox("Run Tracking")

col_left, col_right = st.columns([3, 1])
view = col_left.empty()
log_box = col_right.empty()

# --- LOGIC ---
if video_file and run_app:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    # Get actual video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1: fps = 30
    
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    prev_positions = {}
    
    # Constants
    M_PER_PX = 5 / 200 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Resize immediately (Smaller = Much Faster)
        frame = cv2.resize(frame, (640, 360))
        
        # 2. YOLO - We use .predict for better control
        results = model.predict(frame, conf=0.5, verbose=False, device='cpu')
        
        # 3. Format detections for SORT: [x1, y1, x2, y2, score]
        detections = []
        for r in results:
            for box in r.boxes:
                # Class 2=car, 3=motorcycle, 5=bus, 7=truck (COCO dataset)
                cls = int(box.cls[0])
                if cls in [2, 3, 5, 7]:
                    b = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections.append([b[0], b[1], b[2], b[3], conf])
        
        # Convert to numpy for SORT
        if len(detections) > 0:
            detections = np.array(detections)
        else:
            detections = np.empty((0, 5))

        # 4. Update Tracker
        tracks = tracker.update(detections)
        
        # 5. Speed & Drawing
        for trk in tracks:
            x1, y1, x2, y2, obj_id = map(int, trk)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            speed = 0
            if obj_id in prev_positions:
                px, py = prev_positions[obj_id]
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                # Speed = (Distance * Scale) / (1 / FPS) * 3.6 (for km/h)
                speed = (dist * M_PER_PX) * fps * 3.6
            
            prev_positions[obj_id] = (cx, cy)

            # Visuals
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{obj_id} {int(speed)}km/h", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 6. CRITICAL: Show the frame
        # We convert to RGB because Streamlit expects it
        view.image(frame, channels="BGR", use_container_width=True)

    cap.release()
else:
    st.info("Upload a video and check 'Run Tracking' to start.")
