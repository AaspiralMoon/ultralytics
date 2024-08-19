import numpy as np
import cv2
from ultralytics import YOLO


model = YOLO('yolov8n.pt')

img = cv2.imread('/home/wiser-renjie/projects/yolov8/bus.jpg')
img = cv2.resize(img, (640, 640))

results = model.predict(img, save_txt=False, save_conf=False, save=False, classes=[0], imgsz=(1152, 1920), conf=0.3)