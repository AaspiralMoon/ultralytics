import cv2
import os
import re
import time
import random
import os.path as osp
import numpy as np
from ultralytics import YOLO
from test_crop import load_image


if __name__ == '__main__':
    img_path = '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/train/jena/jena_000066_000002_leftImg8bit.png'
    
    batch_size = 4
    
    imgs = []

    img = load_image(img_path, (512, 256))
    
    for i in range(batch_size):
        imgs.append(img)

    Yolox = YOLO('yolov8x.pt')
    
    # ----------- wamrup ------------
    for i in range (10):
        _ = Yolox.predict(img, save=False, imgsz=(img.shape[0], img.shape[1]), conf=0.5)
    
    # ----------- eval ------------
    # batch
    t1 = time.time()
    _ = Yolox.predict(imgs, save=False, imgsz=(img.shape[0], img.shape[1]), conf=0.5)
    t2 = time.time()
    
    
    # loop
    t3 = time.time()
    for i in range(batch_size):
        _ = Yolox.predict(img, save=False, imgsz=(img.shape[0], img.shape[1]), conf=0.5)
    t4 = time.time()
    
    
    print('Batch processing {} images, total: {} ms, average: {} ms'.format(batch_size, round((t2-t1)*1000, 2), round((t2-t1)*1000/batch_size, 2)))
    print('Loop processing {} images, total: {} ms, average: {} ms'.format(batch_size, round((t4-t3)*1000, 2), round((t4-t3)*1000/batch_size, 2)))