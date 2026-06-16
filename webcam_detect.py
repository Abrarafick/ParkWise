"""
ParkWise — Live Webcam Detection Script
Run this on your laptop alongside the Flask backend.
It captures webcam frames, detects parking spots using OpenCV,
and sends results to the backend API every 5 seconds.

Usage:
    python webcam_detect.py

Requirements:
    pip install opencv-python requests numpy
"""
import cv2
import numpy as np
import requests
import time
import json
import sys

API_URL = "http://localhost:5000/api/cv/detect"
CAMERA_INDEX = 0       # 0 = default webcam
DETECT_INTERVAL = 5    # seconds between detections
TOTAL_SPOTS = 20
ROWS, COLS = 2, 10

# Spot IDs
SPOT_IDS = [f"A{i}" for i in range(1, 11)] + [f"B{i}" for i in range(1, 11)]

# Colors (BGR)
COLOR_FREE = (34, 197, 94)   # green
COLOR_OCC  = (60, 60, 220)   # red

bg_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40)

def detect_from_frame(frame):
    h, w = frame.shape[:2]
    fg = bg_sub.apply(frame)
    detections = {}
    idx = 0
    cell_h, cell_w = h // ROWS, w // COLS
    for r in range(ROWS):
        for c in range(COLS):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            roi = fg[y1:y2, x1:x2]
            motion = np.count_nonzero(roi) / roi.size
            spot_id = SPOT_IDS[idx]
            detections[spot_id] = motion > 0.08
            idx += 1
    return detections

def draw_overlay(frame, detections):
    h, w = frame.shape[:2]
    cell_h, cell_w = h // ROWS, w // COLS
    idx = 0
    for r in range(ROWS):
        for c in range(COLS):
            spot_id = SPOT_IDS[idx]
            occupied = detections.get(spot_id, False)
            color = COLOR_OCC if occupied else COLOR_FREE
            y1, y2 = r * cell_h + 3, (r + 1) * cell_h - 3
            x1, x2 = c * cell_w + 3, (c + 1) * cell_w - 3
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = "OCC" if occupied else "FREE"
            cv2.putText(frame, spot_id, (x1 + 5, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            cv2.putText(frame, label,    (x1 + 5, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)
            idx += 1
    return frame

def send_to_backend(detections):
    try:
        resp = requests.post(API_URL, json={"detections": detections}, timeout=3)
        if resp.status_code == 200:
            print(f"[ParkWise CV] Sent detections: {sum(detections.values())} occupied / {TOTAL_SPOTS} total")
    except Exception as e:
        print(f"[ParkWise CV] Backend not reachable: {e}")

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Webcam not found. Check CAMERA_INDEX in script.")
        sys.exit(1)

    print("✅ ParkWise CV Detection Started")
    print("   Press 'q' to quit | 's' to take snapshot")
    print(f"   Sending to: {API_URL} every {DETECT_INTERVAL}s")
    print()

    last_sent = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame"); break

        detections = detect_from_frame(frame)
        frame_out  = draw_overlay(frame, detections)

        occ_count  = sum(detections.values())
        free_count = TOTAL_SPOTS - occ_count

        # HUD
        cv2.rectangle(frame_out, (0, 0), (320, 50), (27, 58, 92), -1)
        cv2.putText(frame_out, f"ParkWise CV", (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255,255,255), 1)
        cv2.putText(frame_out, f"FREE: {free_count}  OCC: {occ_count}", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,255,100), 1)

        cv2.imshow("ParkWise — Live Parking Detection", frame_out)

        # Send to backend every DETECT_INTERVAL seconds
        now = time.time()
        if now - last_sent >= DETECT_INTERVAL:
            send_to_backend(detections)
            last_sent = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame_out)
            print(f"📸 Snapshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("ParkWise CV stopped.")

if __name__ == "__main__":
    main()
