"""
Multi-Object Tracking (MOT) Demo
--------------------------------
Combines YOLOv8 detections with Hungarian Assignment for object tracking.
Demonstrates CV pipeline integration with detection + tracking.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

class Tracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, detections):
        """Update tracks using Hungarian assignment."""
        det_centroids = np.array([d[:2] for d in detections]) if detections else []
        track_centroids = np.array([t["centroid"] for t in self.tracks.values()]) if self.tracks else []

        if len(det_centroids) and len(track_centroids):
            cost = np.linalg.norm(track_centroids[:, None] - det_centroids[None, :], axis=2)
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            row_idx, col_idx = [], []

        assigned = set()
        # Update assigned tracks
        for r, c in zip(row_idx, col_idx):
            tid = list(self.tracks.keys())[r]
            self.tracks[tid]["centroid"] = det_centroids[c]
            self.tracks[tid]["bbox"] = detections[c][2:]  # keep bbox too
            assigned.add(c)

        # Add new tracks
        for i, det in enumerate(detections):
            if i not in assigned:
                self.tracks[self.next_id] = {"centroid": det[:2], "bbox": det[2:]}
                self.next_id += 1

        return self.tracks

def run_mot(video_source="video_examples/sample.mp4", model_path="yolov8n.pt", save_out=True):
    cap = cv2.VideoCapture(0 if video_source == "0" else video_source)
    model = YOLO(model_path)
    tracker = Tracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []
        for box in results.boxes.xywh.cpu().numpy():
            x, y, w, h = box
            detections.append([x, y, w, h])  # store centroid + bbox

        tracks = tracker.update(detections)

        # Draw tracks
        for tid, data in tracks.items():
            x, y = map(int, data["centroid"])
            w, h = map(int, data["bbox"])
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("MOT Demo", frame)

        # quit with ESC or X
        if (cv2.waitKey(1) & 0xFF == 27) or cv2.getWindowProperty("MOT Demo", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_mot()
