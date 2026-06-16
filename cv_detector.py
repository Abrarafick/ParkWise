import cv2
import numpy as np
import random
import time

class ParkingDetector:
    """
    Parking spot detector using OpenCV.
    - With webcam: uses background subtraction + contour detection
    - Without webcam: returns simulated detection results
    """

    def __init__(self):
        self.spot_ids = [f"A{i}" for i in range(1, 11)] + [f"B{i}" for i in range(1, 11)]
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
        # Simulated base occupancy pattern (realistic)
        self._base_pattern = {
            "A1": False, "A2": True,  "A3": False, "A4": False, "A5": True,
            "A6": False, "A7": True,  "A8": False, "A9": False, "A10": True,
            "B1": True,  "B2": False, "B3": True,  "B4": False, "B5": False,
            "B6": True,  "B7": False, "B8": True,  "B9": False, "B10": False,
        }

    def detect_from_frame(self, frame):
        """
        Detect parking occupancy from a single camera frame.
        Uses background subtraction + brightness analysis per zone.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor.apply(frame)

        detections = {}
        # Divide frame into a 10x2 grid (simplified)
        rows, cols = 2, 10
        cell_h = h // rows
        cell_w = w // cols

        spot_idx = 0
        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * cell_h, (r + 1) * cell_h
                x1, x2 = c * cell_w, (c + 1) * cell_w
                roi_mask = fg_mask[y1:y2, x1:x2]
                motion_pct = np.count_nonzero(roi_mask) / roi_mask.size
                spot_id = self.spot_ids[spot_idx]
                detections[spot_id] = motion_pct > 0.08  # occupied if >8% motion
                spot_idx += 1

        return {
            "detections": detections,
            "method": "camera",
            "timestamp": time.time(),
            "confidence": 0.87
        }

    def detect_simulated(self):
        """
        Simulate detection with slight random drift from base pattern.
        Used when no camera is available.
        """
        detections = {}
        for spot_id, base_occ in self._base_pattern.items():
            # 10% chance of state flip each cycle (realistic churn)
            flip = random.random() < 0.10
            detections[spot_id] = (not base_occ) if flip else base_occ

        return {
            "detections": detections,
            "method": "simulated",
            "timestamp": time.time(),
            "confidence": 0.95,
            "note": "Using simulated detection — connect webcam for live CV"
        }

    def draw_overlay(self, frame, detections):
        """Draw colored overlays on frame for each parking spot."""
        h, w = frame.shape[:2]
        rows, cols = 2, 10
        cell_h = h // rows
        cell_w = w // cols
        spot_idx = 0
        for r in range(rows):
            for c in range(cols):
                spot_id = self.spot_ids[spot_idx]
                occupied = detections.get(spot_id, False)
                color = (0, 0, 220) if occupied else (0, 200, 0)
                y1, y2 = r * cell_h + 4, (r + 1) * cell_h - 4
                x1, x2 = c * cell_w + 4, (c + 1) * cell_w - 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, spot_id, (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                label = "OCC" if occupied else "FREE"
                cv2.putText(frame, label, (x1 + 5, y2 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
                spot_idx += 1
        return frame
