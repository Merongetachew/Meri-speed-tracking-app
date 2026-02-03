import streamlit as st
import cv2
import numpy as np
import tempfile
from sort import * # Ensure sort.py is in your GitHub repo!
from ultralytics import YOLO

# --- CONFIG ---
st.set_page_config(page_title="Vehicle Speed Monitor", layout="wide")

# Use st.cache_resource so the model doesn't reload every time you click a button
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# --- STATE MANAGEMENT ---
if 'run' not in st.session_state:
    st.session_state.run = False

def start_tracking(): st.session_state.run = True
def stop_tracking(): st.session_state.run = False

# --- UI ---
st.title("🚗 Vehicle Speed Tracking System")
video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi"])
conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)

st.sidebar.button("Start Tracking", on_click=start_tracking)
st.sidebar.button("Stop Tracking", on_click=stop_tracking)

col1, col2 = st.columns([3, 1])
video_placeholder = col1.empty()
log_placeholder = col2.empty()

# --- LOGIC ---
if video_file and st.session_state.run:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    tracker = Sort(max_age=30)
    
    # Constants for calculation
    FRAME_RATE = cap.get(cv2.CAP_PROP_FPS) or 30
    PIXEL_TO_METER = 5 / 200 
    
    prev_positions = {}
    logs = []

    while cap.isOpened() and st.session_state.run:
        ret, frame = cap.read()
        if not ret: break

        # 1. Faster Processing: Resize frame immediately
        frame = cv2.resize(frame, (640, 360)) 
        
        # 2. YOLO Detection
        results = model(frame, verbose=False, conf=conf_threshold)
        detections = np.empty((0, 5))
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Check for car/truck/bus (classes 2, 5, 7 in COCO)
                if cls in [2, 3, 5, 7]:
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        # 3. Sort Tracking
        if len(detections) > 0:
            track_result = tracker.update(detections)
            for res in track_result:
                x1, y1, x2, y2, obj_id = map(int, res)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Speed calc
                speed = 0
                if obj_id in prev_positions:
                    px, py = prev_positions[obj_id]
                    dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                    speed = (dist * PIXEL_TO_METER) * FRAME_RATE * 3.6
                
                prev_positions[obj_id] = (cx, cy)

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {obj_id}: {int(speed)}km/h", (x1, y1-5), 0, 0.6, (255,255,255), 2)

        # 4. Streamlit Update
        video_placeholder.image(frame, channels="BGR")
        
    cap.release()
