import argparse
import json
from pathlib import Path
import os
from collections import defaultdict

try:
    from PIL import Image
except Exception:
    Image = None

def load_sizes(images_dir, file_name):
    if Image is None:
        return None, None
    p = Path(images_dir) / file_name
    if not p.exists():
        return None, None
    with Image.open(p) as im:
        w, h = im.size
    return w, h

def convert_bbox_to_yolo(bbox, img_w, img_h):
    # COCO bbox: [x, y, width, height]
    x, y, w, h = bbox
    xc = x + w / 2.0
    yc = y + h / 2.0
    return [
        xc / img_w,
        yc / img_h,
        w / img_w,
        h / img_h,
    ]

def main():
    ap = argparse.ArgumentParser(description="Convert COCO annotations to YOLO format")
    ap.add_argument("--coco", required=True, help="Path to COCO JSON")
    ap.add_argument("--images", required=True, help="Directory with images")
    ap.add_argument("--out", required=True, help="Output directory for YOLO labels")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.coco, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build maps
    id_to_image = {im["id"]: im for im in coco.get("images", [])}
    id_to_cat = {c["id"]: c for c in coco.get("categories", [])}
    img_to_anns = defaultdict(list)
    for ann in coco.get("annotations", []):
        img_to_anns[ann["image_id"]].append(ann)

    for img_id, im in id_to_image.items():
        file_name = im["file_name"]
        img_w = im.get("width")
        img_h = im.get("height")
        if img_w is None or img_h is None:
            img_w, img_h = load_sizes(args.images, file_name)
            if img_w is None or img_h is None:
                print(f"[WARN] Missing size for {file_name}, skipping")
                continue

        yolo_lines = []
        for ann in img_to_anns.get(img_id, []):
            cat_id = ann["category_id"]
            # Map category ids to 0..N-1 by index order of categories
            # Build a stable index:
            cat_idx = sorted(id_to_cat.keys()).index(cat_id)
            bbox = ann["bbox"]
            xcycwh = convert_bbox_to_yolo(bbox, img_w, img_h)
            yolo_lines.append(f"{cat_idx} " + " ".join(f"{v:.6f}" for v in xcycwh))

        # Write .txt next to image name
        label_path = out_dir / (Path(file_name).stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as lf:
            lf.write("\n".join(yolo_lines))

        print(f"[OK] Wrote {label_path} ({len(yolo_lines)} objects)")

if __name__ == "__main__":
    main()
