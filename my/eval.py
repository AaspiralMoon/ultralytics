import os
import os.path as osp
import numpy as np
import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    img_root = '/home/wiser-renjie/datasets/test_partial/test/images'
    
    model = YOLO('yolov8x.pt')

    results = model.predict(source=img_root, save=True, save_txt=True, imgsz=128, conf=0.5)