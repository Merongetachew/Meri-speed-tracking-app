import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import tempfile

# --- 1. Custom HTML & CSS Injection ---
st.set_page_config(page_title="Vehicle Speed Tracker", layout="wide")

st.markdown("""
    <style>
        /* Main background and container styling */
        .main { background-color: #f0f2f5; }
        
        .custom-container {
            max-width: 1000px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 20px;
        }
        
        h1 { color: #1a73e8; font-family: 'Segoe UI', sans-serif; }
        .status-msg { color: #666; font-style: italic; margin-top: 10px; }
        
        /* Sidebar styling for Oromia Road Safety Branding */
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #ddd;
        }
    </style>
    
    <div class="custom-container">
        <h1>Vehicle Speed Detector</h1>
        <p>Road Safety Support: Oromia Region | Data & Surveillance</p>
    </div>
    """, unsafe_allow_html=True)

# --- 2. Setup & Calibration ---
# Note: Ensure these match your specific camera setup for Oromia roads
CALIBRATION_DISTANCE = 5 
CALIBRATION_PIXELS = 200 
PIXEL_TO_METER = CALIBRATION_DISTANCE / CALIBRATION_PIXELS
FRAME_RATE = 30

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()
tracker = Sort(max_age=30)

# --- 3. Sidebar UI (Matches your HTML Upload Section) ---
st.sidebar.header("Step 1: Upload Video")
video_file = st.sidebar.file_uploader("Upload a Traffic Video", type=['mp4', 'mov', 'avi'])
st.sidebar.markdown('<p class="status-msg">Upload a video of vehicles to begin analysis.</p>', unsafe_allow_html=True)

start_btn = st.sidebar.button("🚀 Start Analysis", use_container_width=True)
stop_btn = st.sidebar.button("⏹ Stop / Reset", use_container_width=True)

# Placeholder for the video feed
container = st.empty() 

# --- 4. Video Processing Logic ---
if video_file and start_btn:
    st.write(f"### Step 2: Processing: **{video_file.name}**")
    
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    prev_positions = {}

    while cap.isOpened():
        if stop_btn:
            st.warning("Processing stopped by user.")
            break
            
        ret, frame = cap.read()
        if not ret:
            st.success("Analysis Complete.")
            break

        # Resize for better memory management on Streamlit Cloud
        frame = cv2.resize(frame, (960, 540))
        results = model(frame, stream=True, verbose=False)
        
        detections = np.empty((0, 5))
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Filter for vehicles: car(2), motorcycle(3), bus(5), truck(7)
                if cls in [2, 3, 5, 7] and conf > 0.5:
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        tracks = tracker.update(detections)
        
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if track_id in prev_positions:
                px, py = prev_positions[track_id]
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                speed = (dist * PIXEL_TO_METER) * FRAME_RATE * 3.6
                
                # Draw Speed Label
                cv2.putText(frame, f"ID {track_id}: {int(speed)} km/h", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            prev_positions[track_id] = (cx, cy)

        # Convert for Streamlit display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        container.image(frame, channels="RGB", use_container_width=True)

    cap.release()
else:
    # This matches your "else" block in HTML
    st.info("Video display will appear here once you upload a file and click Start.")
