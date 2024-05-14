import os
import cv2
import numpy as np
import os.path as osp
from ultralytics import YOLO as yolo
from OBDS import OBDS_single

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)
    return d
        
if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/traffic/video1'
    result_root = '/home/wiser-renjie/projects/yolov8/my/runs/my'
    exp_id = 'OBDS1'
    result_path = mkdir_if_missing(osp.join(result_root, exp_id))
    
    interval = 5
        
    model = yolo('yolov8x.pt')

    for i, filename in enumerate(sorted(os.listdir(img_root))):
        img = cv2.imread(osp.join(img_root, filename))
        
        bboxes = []
        if i % interval == 0:
            results = model.predict(img, save=False, classes=[2], conf=0.5)
            refs = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
            bboxes_prev = refs
            
        else:
            for bbox_prev, ref in zip(bboxes_prev, refs):
                target = img[ref[1]:ref[3], ref[0]:ref[2]]
                box = OBDS_single(img, target, bbox_prev)
                bboxes.append(box)            
            bboxes_prev = bboxes
            
        # Draw bounding boxes
        for box in bboxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        cv2.imwrite(osp.join(result_path, filename), img)