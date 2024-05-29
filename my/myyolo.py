import os
import cv2
import time
import numpy as np
import os.path as osp
from ultralytics import YOLO as yolo
from myMedianTracker import xyxy2xywh, transform_bbox

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)
    return d

if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/traffic/video3_30fps'
    result_root = '/home/wiser-renjie/projects/yolov8/my/runs/my'
    exp_id = 'video3_30fps_yolon'
    result_path = mkdir_if_missing(osp.join(result_root, exp_id))
    
    model = yolo('yolov8n.pt')
    
    for i, filename in enumerate(sorted(os.listdir(img_root))):
        if i == 3000:
            break
        img = cv2.imread(osp.join(img_root, filename))
        
        results = model.predict(img, save=False, imgsz=(img.shape[0], img.shape[1]), conf=0.5)
        bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
        bboxes = xyxy2xywh(bboxes)

        for bbox in bboxes:
            x1, y1, w, h = bbox 
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
        
        cv2.imwrite(osp.join(result_path, filename), img)