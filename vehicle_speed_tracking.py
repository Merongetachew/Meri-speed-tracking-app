# --- Setup outside the loop ---
SPEED_LIMIT = 80  # Define your max speed limit here
frame_count = 0

while cap.isOpened():
    if stop_btn:
        break
        
    ret, frame = cap.read()
    if not ret:
        st.write("Video Processing Complete.")
        break

    # 1. REDUCE RESOLUTION (640x360 is much faster for web)
    frame = cv2.resize(frame, (640, 360))
    
    # 2. OPTIMIZE YOLO (imgsz=320 makes the AI 2x faster)
    results = model(frame, stream=True, verbose=False, imgsz=320)
    
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
            speed = (dist * PIXEL_TO_METER) * FRAME_RATE * 3.6
            
            # 3. SPEED LIMIT LOGIC
            color = (0, 255, 0) # Green for normal
            if speed > SPEED_LIMIT:
                color = (255, 0, 0) # Red if over speed limit
                label = f"WARNING! ID {track_id}: {int(speed)} km/h"
            else:
                label = f"ID {track_id}: {int(speed)} km/h"
            
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        prev_positions[track_id] = (cx, cy)

    # Convert and show
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    container.image(frame, channels="RGB", use_container_width=True)
