import os
import cv2
import time
import numpy as np
import os.path as osp
from ultralytics import YOLO as yolo
from utils import STrack

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)
    return d

def xyxy2xywh(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1]
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/wildtrack/decoded_images/cam7'
    result_root = '/home/wiser-renjie/projects/yolov8/my/runs/my'
    exp_id = 'cam7_medianflow'
    result_path = mkdir_if_missing(osp.join(result_root, exp_id))
    
    interval = 10
        
    model = yolo('yolov8x.pt')
    tracker = cv2.legacy.MultiTracker_create()
    trackers = []
    
    for i, filename in enumerate(sorted(os.listdir(img_root))):
        print('\n ----------------- Frame : {} ------------------- \n'.format(i))
        img0 = cv2.imread(osp.join(img_root, filename))
        
        H, W = 1152, 1920
        
        img = cv2.resize(img0, (W, H))
        img0_copy = img0.copy()
        
        if i % interval == 0:
            trackers.clear()
            results = model.predict(img, save=False, imgsz=(H, W), classes=[0], conf=0.5)
            # results = model.predict(img, save=False, imgsz=(H, W), conf=0.5)
            bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
            bboxes = xyxy2xywh(bboxes)
            
            t1 = time.time()
            for bbox in bboxes:
                x1, y1, w, h = bbox
                tracker = cv2.legacy.TrackerMedianFlow_create()
                tracker.init(img, (x1, y1, w, h))
                trackers.append(tracker)
            t2 = time.time()
            # print('Init time: {} ms\n'.format((t2-t1)*1000))

        else:
            bboxes = []
            t3 = time.time()
            for j, tracker in enumerate(trackers):
                ok, out = tracker.update(img)
                bboxes.append(out)
            bboxes = np.array(bboxes)
            t4 = time.time()
            print('Track time: {} ms\n'.format((t4-t3)*1000))

        color = (0, 0, 255) if i % interval == 0 else (255, 0, 0)
        # Draw bounding boxes
        for bbox in bboxes.astype(np.int32):
            x1, y1, w, h = bbox 
            cv2.rectangle(img0_copy, (x1, y1), (x1 + w, y1 + h), color, 2)
        
        cv2.imwrite(osp.join(result_path, filename), img0_copy)