import cv2
import os
import re
import time
import random
import os.path as osp
import numpy as np
from ultralytics import YOLO
from test_crop import load_image, get_idx

if __name__ == '__main__':
    result_root = '/home/wiser-renjie/projects/yolov8/my/runs/detect/'
    save_path = osp.join(result_root, 'predict'+get_idx(result_root))
    img_path = '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/train/jena/jena_000066_000002_leftImg8bit.png'
    
    batch_size = 50
    
    imgs = []

    img = load_image(img_path, (128, 128))
    
    for i in range(batch_size):
        imgs.append(img)

    Yolox = YOLO('yolov8x.pt')
    
    # ----------- wamrup ------------
    for i in range (10):
        _ = Yolox.predict(imgs, save=False, imgsz=(img.shape[0], img.shape[1]), conf=0.5)
    
    # ----------- eval ------------
    # batch
    t1 = time.time()
    _ = Yolox.predict(imgs, save=False, imgsz=(img.shape[0], img.shape[1]), conf=0.5)
    t2 = time.time()
    print('Batch processing {} images, total: {} ms, average: {} ms'.format(batch_size, (t2-t1)*1000, (t2-t1)*1000/batch_size))
    
    # loop
    t1 = time.time()
    for i in range(batch_size):
        _ = Yolox.predict(img, save=False, imgsz=(img.shape[0], img.shape[1]), conf=0.5)
    t2 = time.time()
    print('loop processing {} images, total: {} ms, average: {} ms'.format(batch_size, (t2-t1)*1000, (t2-t1)*1000/batch_size))