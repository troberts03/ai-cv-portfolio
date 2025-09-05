import argparse
import sys
from pathlib import Path
import numpy as np
import cv2

def iou_xyxy(a, b):
    # a,b: [x1,y1,x2,y2]
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union

def load_yolo(model_name):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    return YOLO(model_name)

class Track:
    def __init__(self, tid, box, label, score):
        self.id = tid
        self.box = np.array(box, dtype=float)
        self.label = int(label)
        self.score = float(score)
        self.missed = 0

def assign_tracks(tracks, detections, iou_thresh=0.2):
    # Build cost matrix using (1 - IoU)
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    cost = np.zeros((len(tracks), len(detections)), dtype=float)
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            cost[i, j] = 1.0 - iou_xyxy(t.box, d["box"])
    # Hungarian assignment
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    matches, unmatched_t, unmatched_d = [], [], []
    assigned_d = set()
    for r, c in zip(row_ind, col_ind):
        if 1.0 - cost[r, c] >= iou_thresh:
            matches.append((r, c))
            assigned_d.add(c)
        else:
            unmatched_t.append(r)
    for t_idx in range(len(tracks)):
        if t_idx not in [m[0] for m in matches] and t_idx not in unmatched_t:
            unmatched_t.append(t_idx)
    for d_idx in range(len(detections)):
        if d_idx not in assigned_d:
            unmatched_d.append(d_idx)
    return matches, unmatched_t, unmatched_d

def parse_args():
    ap = argparse.ArgumentParser(description="Simple MOT demo with YOLO detections")
    ap.add_argument("--source", required=True, help="video path or webcam index (e.g., 0)")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.2, help="IoU threshold for matching")
    ap.add_argument("--view", action="store_true", help="Show live window")
    ap.add_argument("--save", default="runs/mot/mot_demo.mp4", help="Output video path")
    return ap.parse_args()

def main():
    args = parse_args()
    cap_source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print("Failed to open source:", args.source)
        return

    model = load_yolo(args.model)
    tracks = []
    next_id = 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        # Run YOLO
        yres = model.predict(frame, conf=args.conf, verbose=False)
        dets = []
        if yres and hasattr(yres[0], "boxes") and yres[0].boxes is not None:
            for b in yres[0].boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                cls = int(b.cls[0].item()) if b.cls is not None else 0
                conf = float(b.conf[0].item()) if b.conf is not None else 0.0
                dets.append({"box": xyxy, "cls": cls, "score": conf})

        # Assign
        matches, un_t, un_d = assign_tracks(tracks, dets, iou_thresh=args.iou)

        # Update matched
        for ti, di in matches:
            tracks[ti].box = np.array(dets[di]["box"], dtype=float)
            tracks[ti].label = dets[di]["cls"]
            tracks[ti].score = dets[di]["score"]
            tracks[ti].missed = 0

        # Add new tracks
        for di in un_d:
            d = dets[di]
            tracks.append(Track(next_id, d["box"], d["cls"], d["score"]))
            next_id += 1

        # Age unmatched tracks
        alive = []
        for ti in range(len(tracks)):
            if ti in [m[0] for m in matches]:
                alive.append(tracks[ti])
            else:
                tracks[ti].missed += 1
                if tracks[ti].missed <= 15:
                    alive.append(tracks[ti])
        tracks = alive

        # Draw
        vis = frame.copy()
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t.box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"ID {t.id} cls {t.label}", (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if out is None:
            out = cv2.VideoWriter(args.save, fourcc, 30, (w, h))
        out.write(vis)

        if args.view:
            cv2.imshow("MOT Demo", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"[OK] Saved: {args.save}")

if __name__ == "__main__":
    main()
