import os
import cv2
import time
import numpy as np
import threading
import queue
from ultralytics import YOLO as yolo
from utils import mkdir_if_missing, tlbr2tlwh

def run_detector(img, model, H, W, queue):
    start = time.time()
    results = model.predict(img, save=False, imgsz=(H, W), classes=[0], conf=0.3)
    bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
    confs = results[0].boxes.conf.cpu().numpy().astype(np.float32)
    clses = results[0].boxes.cls.cpu().numpy().astype(np.int32)
    out = np.hstack((clses[:, None], bboxes, confs[:, None]))
    end = time.time()
    detection_time = (end - start)*1000
    queue.put((out, detection_time))

if __name__ == '__main__':
    img_path1 = '/home/wiser-renjie/remote_datasets/wildtrack/decoded_images/cam7/00000446.jpg'
    img_path2 = '/home/wiser-renjie/remote_datasets/wildtrack/decoded_images/cam7/00000448.jpg'
    
    H, W = 1152, 1920
    
    model = yolo('yolov8x.pt')
    
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    img1 = cv2.resize(img1, (W, H))
    img2 = cv2.resize(img2, (W, H))

    for i in range(5):  # warmup
        results1 = model.predict(img1, save=False, imgsz=(H, W), classes=[0], conf=0.3)
        bboxes1 = results1[0].boxes.xyxy.cpu().numpy().astype(np.int32)
    
    t1 = time.time()
    tracker = cv2.legacy.MultiTracker_create()
    for bbox in bboxes1:
        x1, y1, w, h = bbox
        tracker.add(cv2.legacy.TrackerMedianFlow_create(), img1, (x1, y1, w, h))
    t2 = time.time()
    print('Init time: {} ms\n'.format((t2 - t1) * 1000))
    
    t3 = time.time()
    results1 = model.predict(img2, save=False, imgsz=(H, W), classes=[0], conf=0.3)
    bboxes1 = results1[0].boxes.xyxy.cpu().numpy().astype(np.int32)
    t4 = time.time()
    print('Detection time: {} ms\n'.format((t4 - t3) * 1000))
    
    t5 = time.time()
    success, bboxes = tracker.update(img2)
    t6 = time.time()
    print('Tracking time: {} ms\n'.format((t6 - t5) * 1000))
    
    print('Sequential time: {} ms\n'.format((t6 - t3) * 1000))
    
    q = queue.Queue()
    
    t7 = time.time()
    detector_thread = threading.Thread(target=run_detector, args=(img2, model, H, W, q))

    detector_thread.start()
    
    success, tracker_bboxes = tracker.update(img2)
    
    detector_thread.join()
    
    detector_bboxes = q.get()
    
    t8 = time.time()
    print('Multi-threading time: {} ms\n'.format((t8 - t7) * 1000))
