# AI Computer Vision Portfolio

This repository showcases core **Computer Vision skills** using YOLOv8, COCO datasets, and MOT tracking pipelines.  
Includes end-to-end demos for object detection, tracking, and dataset conversion.  

## Features
- **YOLOv8 Inference GUI**  
  - Run object detection on images, video, or webcam.  
  - Interactive Tkinter carousel to step through multiple images.  

- **Multi-Object Tracking (MOT)**  
  - Combines YOLOv8 detections with Hungarian Assignment.  
  - Tracks objects across frames in a demo video.  
  - Saves annotated video output for review.  

- **COCO to YOLO Converter**  
  - Converts COCO-format annotations into YOLO `.txt` labels.  
  - Includes **minimal COCO sample dataset** for testing.  

## Installation
```bash
git clone https://github.com/troberts03/ai-cv-portfolio.git
cd ai-cv-portfolio
pip install -r requirements.txt
```

## Run YOLO Inference
Run detection on sample images in a GUI with Prev/Next buttons:

`python yolo_infer.py --source image_examples --model yolov8n.pt`

Press Next / Prev to navigate images.

Close window with X or ESC.

## Run MOT Tracking
Run multi-object tracking on a demo video:

python mot_demo.py --video video_examples/mot_sample.mp4 --model yolov8n.pt

Tracks objects across frames and assigns IDs.

Saves annotated output as examples/mot_output.mp4.

## Convert COCO → YOLO
Test conversion on included minimal COCO dataset:

python coco_to_yolo.py --json coco_examples/annotations.json --out coco_examples/yolo_labels

Outputs YOLO .txt labels for each image.

## Skills Demonstrated
Object detection (YOLOv8)

Multi-object tracking (Hungarian Algorithm + CV pipeline integration)

Dataset annotation handling (COCO → YOLO)

GUI integration (Tkinter)

Video processing with OpenCV

ML workflow documentation + reproducibility

## Notes
Pretrained weights (yolov8n.pt) will be automatically downloaded by ultralytics.

Replace sample images/videos with your own for custom testing.