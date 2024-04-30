import cv2
import numpy as np
from ultralytics import YOLO

def pad_image(img, pad=3):
    padded= cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(127.5, 127.5, 127.5))
    return padded

# Load a pretrained YOLOv8n model
model = YOLO('/home/wiser-renjie/projects/yolov8/my/runs/detect/train2/weights/best.pt')

# img = cv2.imread('/home/wiser-renjie/datasets/test_partial/test/images/00000035.jpg')
img = cv2.imread('/home/wiser-renjie/datasets/test_partial/val/images/00004219.jpg')

img_padded = pad_image(img, pad=96)

results = model.predict(img_padded, save=True, imgsz=320, conf=0.5)