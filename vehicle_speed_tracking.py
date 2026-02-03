import streamlit as st
from threading import Thread
import time
import numpy as np
import cv2
from sort import Sort
from ultralytics import YOLO
from PIL import Image, ImageTk
import sys
from io import BytesIO

# ---- Helper: convert OpenCV image to PNG bytes for Streamlit display ----
def cv2_to_base64_png(cv_img):
    # cv_img is a BGR image; convert to RGB for display
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    with BytesIO() as buf:
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

# ---- Streamlit page configuration ----
st.set_page_config(page_title="Vehicle Speed Monitoring - Streamlit", layout="wide")

st.title("Vehicle Speed Monitoring System (Streamlit)")
st.markdown("""
This app processes an uploaded video to detect vehicles, track them, and estimate speeds
using a calibrated pixel-to-meter ratio. The core logic mirrors the original Tkinter version.
- Uses YOLOv8 for object detection
- Uses SORT for tracking
- Calculates speed in km/h based on pixel movement
""")

# ---- FILE UPLOADS ----
# 1) Class names file
cls_placeholder = st.sidebar.empty()
cls_file = st.sidebar.file_uploader("Upload class names (classes.txt)", type=["txt"])
if cls_file:
    classnames = cls_file.read().decode("utf-8").splitlines()
else:
    # Try to load from disk if present
    try:
        with open("classes.txt", "r") as f:
            classnames = f.read().splitlines()
    except Exception:
        classnames = []

# 2) Optional: calibration constants
st.sidebar.subheader("Calibration (meters per pixel)")
calib_dist = st.sidebar.number_input("Calibration distance (meters)", value=5.0, min_value=0.1, step=0.5)
calib_pixels = st.sidebar.number_input("Calibration pixels (in frame)", value=200.0, min_value=1.0, step=10.0)
PIXEL_TO_METER = calib_dist / calib_pixels if calib_pixels != 0 else 0.0

FRAME_RATE = 30  # keep as constant, or expose as adjustable if needed

# 3) Video upload
video_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
uploaded_video_path = None

if video_file is not None:
    # Save to a temporary path to feed to OpenCV
    import os
    tmp_path = f"/tmp/{video_file.name}"
    with open(tmp_path, "wb") as f:
        f.write(video_file.getbuffer())
    uploaded_video_path = tmp_path
    st.sidebar.success(f"Video uploaded: {video_file.name}")

# ---- Global state (per session) ----
if "stop_tracking" not in st.session_state:
    st.session_state.stop_tracking = True  # not running initially

if "frame_placeholder" not in st.session_state:
    st.session_state.frame_placeholder = st.empty()

if "log_lines" not in st.session_state:
    st.session_state.log_lines = []

if "tracker" not in st.session_state:
    st.session_state.tracker = None

if "model" not in st.session_state:
    # Load YOLO model lazily when needed
    st.session_state.model = None

# ---- Logging UI ----
st.sidebar.subheader("Logs")
log_area = st.sidebar.empty()

def log_message(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    st.session_state.log_lines.append(line)
    # Keep only last 500 lines to avoid memory blow-up
    if len(st.session_state.log_lines) > 500:
        st.session_state.log_lines = st.session_state.log_lines[-500:]
    # Update UI
    log_area.markdown("\n".join(st.session_state.log_lines))

# Initialize log area with welcome message
if st.session_state.log_lines == []:
    log_message("Initialized streaming UI.")

# ---- Threads / Processing Loop ----
# We'll implement a worker thread that processes frames and pushes to a Streamlit placeholder.

process_container = st.container()
frame_display = st.image(None, caption="Processed frame will appear here", use_column_width=True)
processed_frame = None
processing_thread = None

def load_model_once():
    if st.session_state.model is None:
        log_message("Loading YOLO model 'yolov8n.pt'...")
        st.session_state.model = YOLO('yolov8n.pt')
        log_message("Model loaded.")

def process_video_loop(video_path, pixel_to_meter, display_size=(800, 450)):
    global processed_frame
    # Initialize detector and tracker
    load_model_once()
    model = st.session_state.model
    tracker = Sort(max_age=30)
    st.session_state.tracker = tracker

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_message("Failed to open video.")
        return

    prev_positions = {}

    log_message("Tracking started (Streamlit).")

    # Video resize target
    disp_w, disp_h = display_size

    while True:
        ret, frame = cap.read()
        if not ret:
            log_message("End of video or read error.")
            break

        frame = cv2.resize(frame, (disp_w, disp_h))

        detections = np.empty((0,5))
        # Run model on frame
        results = model(frame, stream=1)

        for info in results:
            boxes = info.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                classindex = int(box.cls[0])
                object_detected = classnames[classindex] if classindex < len(classnames) else str(classindex)

                if object_detected in ['car', 'truck', 'bus'] and conf > 0.6:
                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                    new_det = np.array([x1i, y1i, x2i, y2i, conf])
                    detections = np.vstack((detections, new_det))

        track_result = tracker.update(detections)
        frame_time = 1.0 / FRAME_RATE

        for res in track_result:
            x1, y1, x2, y2, obj_id = map(int, res)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if obj_id in prev_positions:
                px, py, ptime = prev_positions[obj_id]
                dist_px = np.hypot(cx - px, cy - py)
                speed_kmph = (dist_px * pixel_to_meter) / frame_time * 3.6
            else:
                speed_kmph = 0.0

            prev_positions[obj_id] = (cx, cy, time.time())

            log_message(f"ID: {obj_id} | Class: {object_detected} | Speed: {speed_kmph:.1f} km/h")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {obj_id}  Speed: {speed_kmph:.1f} km/h',
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Convert to RGB for display
        disp_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        processed_frame = disp_img
        # Streamlit image update via a shared placeholder
        yield disp_img

        # Small sleep to simulate frame rate
        time.sleep(frame_time)

    cap.release()
    log_message("Tracking stopped.")

def start_processing():
    if not uploaded_video_path:
        st.warning("Please upload a video file first.")
        return

    # Start background thread-like generator
    frame_gen = process_video_loop(uploaded_video_path, PIXEL_TO_METER)

    st.session_state.stop_tracking = False

    # Consume frames and update UI
    for disp_img in frame_gen:
        # Update the frame display
        frame_bytes = cv2_to_base64_png(disp_img)
        # Create a dummy placeholder image for Streamlit
        frame_display.image(disp_img, channels="RGB", use_column_width=True)
        # Sleep is managed inside generator to respect frame_rate
        if st.session_state.stop_tracking:
            break

def stop_processing():
    st.session_state.stop_tracking = True
    log_message("Tracking stopped by user.")

# ---- UI: Controls (in main page, not in sidebar) ----
st.markdown("### Controls")

col1, col2, col3 = st.columns([1,1,1])

with col1:
    if st.button("Start Tracking"):
        # Launch processing in a separate thread-like pattern using a generator
        # Streamlit does not support true background threads in all deployments; we simulate using a thread if possible.
        # Here, we attempt to start the processing in a separate thread to keep UI responsive.
        def _run():
            start_processing()
        t = Thread(target=_run, daemon=True)
        t.start()
        log_message("Started processing thread.")

with col2:
    if st.button("Stop Tracking"):
        stop_processing()

with col3:
    if st.button("Reset Logs"):
        st.session_state.log_lines = []
        log_area.markdown("")

# ---- Frame display area (main window) ----
st.markdown("### Live Preview")
frame_placeholder = st.empty()

# Initialize with a blank frame
blank = np.zeros((450, 800, 3), dtype=np.uint8)
frame_placeholder.image(blank, channels="RGB", caption="Processed frame will appear here", use_column_width=True)

# Note:
# In this Streamlit version, the live update loop is approximated via a generator that yields frames.
# Depending on the Streamlit version and deployment, you may adapt to a more explicit
# asynchronous approach or use st.experimental_rerun with a timer.

# ---- End of app ----
