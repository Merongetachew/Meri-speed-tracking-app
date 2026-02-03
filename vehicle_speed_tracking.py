import streamlit as st
import cv2
import numpy as np
import time
from threading import Thread, Event
from sort import Sort
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io

# Calibration for real-world measurements
CALIBRATION_DISTANCE = 5  # in meters
CALIBRATION_PIXELS = 200  # in pixels
PIXEL_TO_METER = CALIBRATION_DISTANCE / CALIBRATION_PIXELS
FRAME_RATE = 30

# Global state (per user session)
stop_event = None
video_fp = None
tracker = Sort(max_age=30)
model = YOLO('yolov8n.pt')
log_messages = []
display_frame = None
video_seed = None

classnames = []
try:
    with open('classes.txt', 'r') as f:
        classnames = f.read().splitlines()
except Exception:
    # Fallback if classes.txt is not available
    classnames = []

def add_log(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    log_messages.append(f"[{timestamp}] {msg}")

def process_video_stream(frame_generator, stop_evt: Event, display_queue: "list[str]"):
    """
    Process frames from a generator, update display_queue with processed frames (as PIL Images)
    """
    global tracker
    cap_fps = FRAME_RATE
    prev_positions = {}

    while not stop_evt.is_set():
        try:
            frame = next(frame_generator)
        except StopIteration:
            add_log("End of uploaded video.")
            stop_evt.set()
            break
        except Exception as e:
            add_log(f"Frame read error: {e}")
            stop_evt.set()
            break

        # Resize for display
        display_width, display_height = 800, 450
        frame = cv2.resize(frame, (display_width, display_height))

        detections = np.empty((0, 5))

        # Run object detection
        results = model(frame, stream=1)

        for info in results:
            boxes = info.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                classindex = box.cls[0]
                object_detected = classnames[int(classindex)] if classindex < len(classnames) else f"class{int(classindex)}"

                if object_detected in ['car', 'truck', 'bus'] and conf > 0.6:
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    new_det = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, new_det))

        tracker_results = tracker.update(detections)

        for res in tracker_results:
            x1, y1, x2, y2, obj_id = map(int, res)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if obj_id in prev_positions:
                prev_x, prev_y, prev_time = prev_positions[obj_id]
                distance_pixels = np.hypot(center_x - prev_x, center_y - prev_y)
                frame_time = 1.0 / cap_fps
                speed_kmph = (distance_pixels * PIXEL_TO_METER) / frame_time * 3.6
            else:
                speed_kmph = 0.0

            prev_positions[obj_id] = (center_x, center_y, time.time())

            object_class = "Unknown"
            # Note: obj_id corresponds to detection id; we didn't store class per track.
            # If you want class labels per track, extend code to keep mapping.

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID: {obj_id}  Speed: {speed_kmph:.1f} km/h"
            cv2.putText(frame, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Convert to RGB for display
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Push to display queue
        display_queue.append(pil_img)

        # Maintain frame rate
        time.sleep(1.0 / cap_fps)

def frame_generator_from_video_bytes(bytes_buffer, fps=30):
    """
    Generate frames from a video bytes buffer (uploaded file)
    """
    nparr = np.frombuffer(bytes_buffer, np.uint8)
    cap = cv2.VideoCapture(cv2.imdecode(nparr, cv2.IMREAD_COLOR))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def start_processing(uploaded_bytes, stop_evt, display_queue):
    """
    Entry point to start processing: create a frame generator and run processing loop.
    """
    # Create a video capture from the uploaded bytes
    # Streamlit provides uploaded file as bytes; write to temp buffer
    if uploaded_bytes is None:
        add_log("No video uploaded.")
        stop_evt.set()
        return

    # Try to read as a video file path-like object
    # We'll write to a temporary in-memory buffer and open with OpenCV
    video_bytes = uploaded_bytes.read()
    # Use a memory buffer as a temporary file
    nparr = np.frombuffer(video_bytes, np.uint8)
    cap = cv2.VideoCapture(cv2.imdecode(nparr, cv2.IMREAD_COLOR))
    if not cap.isOpened():
        add_log("Failed to open video from uploaded bytes.")
        stop_evt.set()
        return

    def frame_gen():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

    # Run processing
    process_video_stream(frame_gen(), stop_evt, display_queue)

def to_pil_image(img: Image.Image):
    return img

def streamlit_app():
    st.title("Vehicle Speed Monitoring System (Streamlit)")
    st.write("Upload a video file and start tracking to see vehicle speeds.)

    "

    st.set_option('deprecation.showfileUploaderEncoding', False)

    # Persist state in session
    if 'stop_event' not in st.session_state:
        st.session_state.stop_event = Event()
    if 'display_queue' not in st.session_state:
        st.session_state.display_queue = []
    if 'processing_thread' not in st.session_state:
        st.session_state.processing_thread = None
    if 'uploaded_video' not in st.session_state:
        st.session_state.uploaded_video = None

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.session_state.uploaded_video = uploaded_file
        add_log(f"Video uploaded: {uploaded_file.name}")
        st.success(f"Uploaded: {uploaded_file.name}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Tracking"):
            if st.session_state.uploaded_video is None:
                st.error("Please upload a video first.")
            else:
                # Reset state
                st.session_state.stop_event.clear()
                st.session_state.display_queue.clear()
                add_log("Starting tracking...")

                # Start processing in a separate thread
                def worker():
                    start_processing(
                        st.session_state.uploaded_video,
                        st.session_state.stop_event,
                        st.session_state.display_queue
                    )

                t = Thread(target=worker, daemon=True)
                t.start()
                st.session_state.processing_thread = t

    with col2:
        if st.button("Stop Tracking"):
            st.session_state.stop_event.set()
            add_log("Tracking stopped by user.")

    st.subheader("Live Preview")
    preview_placeholder = st.empty()

    # Logs
    st.subheader("Logs")
    log_area = st.empty()

    # Update loop to render frames and logs
    while True:
        # Display latest frame if available
        if st.session_state.display_queue:
            pil_img = st.session_state.display_queue.pop(0)
            buf = io.BytesIO()
            pil_img.save(buf, format='JPEG')
            img_bytes = buf.getvalue()
            preview_placeholder.image(img_bytes, caption="Processed Frame", use_column_width=True)

        # Show logs
        if log_area:
            log_area.text("\n".join(log_messages[-100:]))

        # Break condition to avoid infinite loop in Streamlit
        if st.session_state.stop_event.is_set():
            break

        time.sleep(0.1)
    st.success("Tracking ended.")

if __name__ == "__main__":
    streamlit_app()
