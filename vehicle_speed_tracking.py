import streamlit as st  
import cv2  
import numpy as np  
from ultralytics import YOLO  
from sort import Sort  
import tempfile  
import os  

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

# Use a placeholder for the processed video output  
output_placeholder = st.empty()  
status_text = st.sidebar.empty()  

if video_file and start_btn:  
    # Save uploaded file to temp location  
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')  
    tfile.write(video_file.read())  
    tfile.close()  
    
    # Output video path  
    output_path = tfile.name.replace('.mp4', '_output.mp4')  
    
    cap = cv2.VideoCapture(tfile.name)  
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or FRAME_RATE  
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
    
    # Downscale for speed  
    new_width, new_height = 960, 540  
    
    # Video writer for output  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))  
    
    prev_positions = {}  
    frame_count = 0  
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    
    status_text.text(f"Processing 0/{total_frames} frames...")  
    
    while cap.isOpened():  
        ret, frame = cap.read()  
        if not ret:  
            break  
        
        frame = cv2.resize(frame, (new_width, new_height))  
        results = model(frame, stream=True, verbose=False)  
        
        detections = np.empty((0, 5))  
        for r in results:  
            for box in r.boxes:  
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  
                conf = box.conf[0].cpu().numpy()  
                cls = int(box.cls[0].cpu().numpy())  
                
                if cls in [2, 3, 5, 7] and conf > 0.5:  
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))  
        
        tracks = tracker.update(detections)  
        
        for track in tracks:  
            x1, y1, x2, y2, track_id = map(int, track)  
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  
            
            if track_id in prev_positions:  
                px, py = prev_positions[track_id]  
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)  
                speed = (dist * PIXEL_TO_METER) * fps * 3.6  
                
                cv2.putText(frame, f"ID {track_id}: {int(speed)} km/h", (x1, y1 - 10),  
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  
            prev_positions[track_id] = (cx, cy)  
        
        out.write(frame)  
        frame_count += 1  
        
        # Update status every 10
