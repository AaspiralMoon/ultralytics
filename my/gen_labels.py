import os
import os.path as osp
import numpy as np
import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    img_root = '/home/wiser-renjie/datasets/temp/val/images'
    
    model = YOLO('yolov8x.pt')

    results = model.predict(source=img_root, save=True, save_txt=True, classes=[2], imgsz=(1024, 2048), conf=0.5)