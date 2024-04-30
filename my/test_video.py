import numpy as np
import cv2
from ultralytics import YOLO

def plot_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox.astype(np.int32)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

# Load a pretrained YOLOv8n model
model = YOLO('yolov8x.pt')

video_path = '/home/wiser-renjie/remote_datasets/traffic/video1.mp4'

results = model.predict(source=video_path, save=True, save_txt=True, imgsz=(1024, 2048), conf=0.5)