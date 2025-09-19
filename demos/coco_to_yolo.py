"""
COCO → YOLO Dataset Converter
-----------------------------
Converts COCO-format annotations to YOLOv8 format.
Shows preprocessing skills for custom dataset training.
"""

import argparse
import json
import os

def convert_coco_to_yolo(coco_file, output_dir):
    with open(coco_file, "r") as f:
        coco = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    for img in coco["images"]:
        img_id = img["id"]
        fname = img["file_name"]
        width, height = img["width"], img["height"]

        annots = [a for a in coco["annotations"] if a["image_id"] == img_id]
        yolo_lines = []
        for a in annots:
            cat_id = a["category_id"]
            x, y, w, h = a["bbox"]

            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w /= width
            h /= height
            yolo_lines.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        out_file = os.path.join(output_dir, fname.replace(".jpg", ".txt"))
        with open(out_file, "w") as f:
            f.write("\n".join(yolo_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO dataset to YOLOv8 format")
    parser.add_argument("--coco", default="coco_examples/annotations.json",
                    help="Path to COCO annotations JSON (default: coco_examples/annotations.json)")
    parser.add_argument("--out", default="labels", help="Output directory for YOLO labels")
    args = parser.parse_args()

    convert_coco_to_yolo(args.coco, args.out)
