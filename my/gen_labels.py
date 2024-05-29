import os
import os.path as osp
import numpy as np
import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/traffic/video1_30fps'
    
    model = YOLO('yolov8s.pt')

    results = model.predict(source=img_root, save=True, save_txt=True, imgsz=(2160, 3840), conf=0.3)