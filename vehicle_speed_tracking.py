import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import tempfile

st.set_page_config(page_title="Fast Speed Tracker", layout="wide")

# --- Optimized Constants ---
FRAME_RATE = 30
SPEED_LIMIT = 80 
# Adjusted calibration for 640px width
PIXEL_TO_METER = 5 / 120 

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()
tracker = Sort(max_age=20)

video_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
start_btn = st.sidebar.button("Start Tracking")
container = st.empty()

if video_file and start_btn:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    prev_positions = {}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # 1. MAJOR SPEED BOOST: Low Resolution
        # Small resolution (480p) makes the video much lighter to process
        frame = cv2.resize(frame, (640, 360))

        # 2. AI PROCESSING (Every 3rd frame to keep video moving)
        if frame_count % 3 == 0:
            # imgsz=160 is extremely fast for YOLO
            results = model(frame, stream=True, verbose=False, imgsz=160)
            
            detections = np.empty((0, 5))
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) in [2, 3, 5, 7] and box.conf[0] > 0.3:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections = np.vstack((detections, [x1, y1, x2, y2, box.conf[0].cpu().numpy()]))

            tracks = tracker.update(detections)
            
            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if track_id in prev_positions:
                    px, py = prev_positions[track_id]
                    dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                    # Adjusting speed math for the 3-frame skip
                    speed = (dist * PIXEL_TO_METER) * (FRAME_RATE / 3) * 3.6
                    
                    color = (0, 255, 0) if speed <= SPEED_LIMIT else (255, 0, 0)
                    cv2.putText(frame, f"{int(speed)} km/h", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                prev_positions[track_id] = (cx, cy)

        # 3. FAST DISPLAY
        # Using BGR2RGB is necessary for Streamlit to show colors correctly
        container.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    cap.release()
