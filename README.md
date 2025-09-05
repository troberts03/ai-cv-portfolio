# AI Computer Vision Portfolio Demos

A compact set of **runnable** computer-vision demos suitable for portfolio work:

1. **YOLOv8 Video Inference** — quick object detection on images/video/webcam.
2. **Simple Multi-Object Tracking (MOT)** — Hungarian assignment + IoU/centroid for ID persistence.
3. **COCO → YOLO Converter** — convert COCO annotation JSON to YOLO format.

Tested with Python 3.10+ on macOS/Linux/Windows.

---

## Environment

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 1) YOLOv8 Video Inference

Run YOLOv8 on a video, image folder, or webcam. The script will save annotated outputs to `runs/`.

```bash
python detection/yolo_infer.py --source path/to/video.mp4
# or webcam
python detection/yolo_infer.py --source 0
# or an image folder
python detection/yolo_infer.py --source path/to/images/
```

**Notes**
- The first run downloads a pre-trained model. You can change the model via `--model` (e.g., `yolov8n.pt`, `yolov8s.pt`).

---

## 2) Simple Multi-Object Tracking (MOT)

A minimal tracker that:
- takes detector outputs (from YOLO) frame-by-frame,
- matches detections to existing tracks with Hungarian assignment using a cost based on IoU and/or centroid distance,
- maintains stable IDs across frames.

Run directly on a video. The script will run YOLO for you and overlay track IDs:

```bash
python tracking/mot_demo.py --source path/to/video.mp4 --model yolov8n.pt
```

You can also pipe your own detection JSON per frame using `--detections` for advanced use.

---

## 3) COCO → YOLO Converter

Convert a COCO annotation JSON to YOLO text files (one `.txt` per image).

```bash
python tools/coco_to_yolo.py   --coco path/to/annotations.json   --images path/to/images_dir   --out path/to/yolo_labels_dir
```

**Assumptions**
- COCO `images` entries contain `file_name`, `width`, and `height` (if width/height missing, the script can read image sizes if available).

---

**License:** MIT
