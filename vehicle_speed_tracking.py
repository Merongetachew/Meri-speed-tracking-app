import streamlit as st
import cv2
import numpy as np
import tempfile
from sort import *
from ultralytics import YOLO

# --- CONFIG ---
st.set_page_config(page_title="Vehicle Speed Monitor", layout="wide")

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# --- UI ---
st.title("🚗 Live Vehicle Tracking")
video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi"])
start_btn = st.sidebar.button("Start Tracking")

col1, col2 = st.columns([3, 1])
video_placeholder = col1.empty()
log_placeholder = col2.empty()

if video_file and start_btn:
    # 1. Save and Open Video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    # 2. Setup Tracker
    tracker = Sort(max_age=30)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    PIXEL_TO_METER = 5 / 200 
    prev_positions = {}
    logs = []

    # 3. Processing Loop
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # SKIP FRAMES to keep video moving (Process every 2nd frame)
        if frame_count % 2 != 0:
            continue

        # Resize to make it light for the web browser
        frame = cv2.resize(frame, (640, 360))
        
        # YOLO Detection (Optimized)
        results = model(frame, verbose=False, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                if cls in [2, 3, 5, 7] and conf > 0.4: # Car, Bus, Truck
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        # SORT Update
        track_result = tracker.update(detections)
        
        for res in track_result:
            x1, y1, x2, y2, obj_id = map(int, res)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Speed logic
            speed = 0
            if obj_id in prev_positions:
                px, py = prev_positions[obj_id]
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                # Multiply by 2 because we are skipping every 2nd frame
                speed = (dist * PIXEL_TO_METER) * (fps / 2) * 3.6
            
            prev_positions[obj_id] = (cx, cy)

            # Draw on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {obj_id}: {int(speed)}km/h", (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # CRITICAL: Push the frame to Streamlit
        # Using BGR2RGB because Streamlit expects RGB
        video_placeholder.image(frame, channels="BGR", use_container_width=True)
        
        # Update logs sparingly to save performance
        if frame_count % 10 == 0:
            log_placeholder.code("\n".join(logs[-10:]))

    cap.release()
    st.success("Processing Finished")
