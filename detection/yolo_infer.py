import argparse
from pathlib import Path
import sys
import cv2

def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 inference on images/video/webcam")
    p.add_argument("--source", required=True, help="Path to image/video/folder or webcam index (e.g., 0)")
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics model name or path")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--save_dir", default="runs/detect", help="Output directory")
    return p.parse_args()

def load_model(name):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    return YOLO(name)

def main():
    args = parse_args()
    model = load_model(args.model)
    # Ultralytics handles most I/O types automatically
    results = model.predict(source=args.source, conf=args.conf, save=True, project=args.save_dir, name="yolo_infer")
    # Print a quick summary
    for r in results:
        path = Path(getattr(r, 'path', ''))
        print(f"[OK] Processed: {path.name}  boxes={len(r.boxes) if hasattr(r, 'boxes') else 0}")
    print(f"Outputs saved under: {Path(args.save_dir) / 'yolo_infer'}")

if __name__ == '__main__':
    main()
