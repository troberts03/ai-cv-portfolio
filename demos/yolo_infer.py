"""
YOLOv8 Inference Demo (Tkinter GUI)
-----------------------------------
Run YOLOv8 inference on a folder of images.
Provides a GUI with Prev/Next buttons to click through results.
"""

import argparse
import os
import glob
import cv2
from ultralytics import YOLO
from tkinter import Tk, Label, Button, Frame, messagebox
from PIL import Image, ImageTk

class YOLOCarousel:
    def __init__(self, root, model_path, folder, max_size=(800, 600)):
        self.root = root
        self.root.title("YOLOv8 Inference Carousel")

        self.model = YOLO(model_path)
        self.max_w, self.max_h = max_size

        # Load only valid image files
        all_files = glob.glob(os.path.join(folder, "*.*"))
        self.images = [f for f in sorted(all_files) if f.lower().endswith((".jpg", ".png", ".jpeg")) and os.path.isfile(f)]
        if not self.images:
            raise ValueError(f"No images found in {folder}")
        self.idx = 0

        # UI layout
        self.label = Label(self.root)
        self.label.pack(padx=10, pady=10)

        btn_frame = Frame(self.root)
        btn_frame.pack(pady=10)

        self.btn_prev = Button(btn_frame, text="Prev", width=12, command=self.show_prev)
        self.btn_prev.pack(side="left", padx=10)

        self.btn_next = Button(btn_frame, text="Next", width=12, command=self.show_next)
        self.btn_next.pack(side="right", padx=10)

        self.show_image()

    def show_image(self):
        img_path = self.images[self.idx]

        if not os.path.exists(img_path):
            messagebox.showwarning("Missing File", f"Image not found:\n{img_path}\nSkipping...")
            self.show_next()
            return

        results = self.model(img_path)
        annotated = results[0].plot()

        # Convert BGR → RGB
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(annotated)

        # Resize to fit window
        im_pil.thumbnail((self.max_w, self.max_h), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=im_pil)

        self.label.config(image=imgtk)
        self.label.image = imgtk  # prevent GC
        self.root.title(f"YOLOv8 Inference ({self.idx+1}/{len(self.images)}) - {os.path.basename(img_path)}")

    def show_next(self):
        self.idx = (self.idx + 1) % len(self.images)
        self.show_image()

    def show_prev(self):
        self.idx = (self.idx - 1) % len(self.images)
        self.show_image()

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Inference (Tkinter GUI)")
    parser.add_argument("--source", type=str, default="image_examples",
                        help="Folder of images (default: image_examples)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model (default: yolov8n.pt)")
    args = parser.parse_args()

    root = Tk()
    app = YOLOCarousel(root, args.model, args.source)
    root.mainloop()

if __name__ == "__main__":
    main()
