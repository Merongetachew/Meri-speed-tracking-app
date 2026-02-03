import streamlit as st
import cv2
import numpy as np
import tempfile
from sort import * from ultralytics import YOLO

st.set_page_config(page_title="Fast Speed Tracker", layout="wide")

@st.cache_resource
def load_yolo():
    # Use the 'n' (Nano) model, it is the only one fast enough for web
    return YOLO('yolov8n.pt')

model = load_yolo()

st.sidebar.title("Fast Mode Settings")
video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi"])
# Frame skipping: Increase this if it's still too slow
skip_frames = st.sidebar.slider("Skip Frames (Higher = Faster)", 1, 10, 3)

view = st.empty()

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    tracker = Sort(max_age=20)
    prev_positions = {}
    
    # Constants
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    PIXEL_TO_METER = 5 / 200
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        count += 1
        # SKIP FRAMES: This prevents the "Stacking/Freezing"
        if count % skip_frames != 0:
            continue

        # 1. Resize to a very small size for the web
        frame = cv2.resize(frame, (480, 270)) 
        
        # 2. YOLO Detect (Optimized)
        results = model.predict(frame, conf=0.5, verbose=False, half=True)
        
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in [2, 3, 5, 7]:
                    b = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections.append([b[0], b[1], b[2], b[3], conf])

        # 3. Update Tracker
        tracks = tracker.update(np.array(detections) if detections else np.empty((0, 5)))

        # 4. Draw & Speed
        for trk in tracks:
            x1, y1, x2, y2, obj_id = map(int, trk)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            if obj_id in prev_positions:
                px, py = prev_positions[obj_id]
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                # Adjust speed for the frames we skipped
                speed = (dist * PIXEL_TO_METER) * (fps / skip_frames) * 3.6
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{obj_id} {int(speed)}km/h", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            prev_positions[obj_id] = (cx, cy)

        # 5. Display
        view.image(frame, channels="BGR", use_container_width=True)

    cap.release()
