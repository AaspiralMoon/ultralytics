import os
import cv2
import time
import numpy as np
import os.path as osp
from ultralytics import YOLO as yolo
from utils import STrack
from myMedianTracker import mkdir_if_missing, xyxy2xywh, transform_bbox

if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/traffic/video1_30fps'
    # img_root = '/home/wiser-renjie/remote_datasets/MOT17/MOT17-04-SDP/img1'
    result_root = '/home/wiser-renjie/projects/yolov8/my/runs/my'
    exp_id = 'test'
    result_path = mkdir_if_missing(osp.join(result_root, exp_id))
    
    interval = 10
    light_track_frames = 3
    assert light_track_frames <= interval
    
    model = yolo('yolov8x.pt')
    trackers = []
    
    for i, filename in enumerate(sorted(os.listdir(img_root))):
        # if i == interval:
        #     break
        img0 = cv2.imread(osp.join(img_root, filename))
        H0, W0 = img0.shape[:2]
        
        img0_copy = img0.copy()
        
        H, W = 640, 640
        img = cv2.resize(img0, (H, W))
        
        print('\n ----------------- Frame : {} ------------------- \n'.format(i))
        if i % interval == 0:
            trackers.clear()
            results = model.predict(img, save=False, imgsz=(H, W), classes=[2], conf=0.5)
            bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
            bboxes = xyxy2xywh(bboxes)
            
            dets = np.array([STrack(tlwh) for tlwh in bboxes], dtype=object)
            
            t1 = time.time()
            for bbox in bboxes:
                x1, y1, w, h = bbox
                tracker = cv2.legacy.TrackerMedianFlow_create()
                tracker.init(img, (x1, y1, w, h))
                trackers.append(tracker)
            t2 = time.time()
            print('Init time: {} ms\n'.format((t2-t1)*1000))
            
        else:
            if i % interval <= light_track_frames:
                t3 = time.time()
                
                oks = []
                bboxes = []
                
                for tracker in trackers:
                    ok, bbox = tracker.update(img)
                    oks.append(ok)
                    bboxes.append(bbox)
                    
                bboxes = np.array(bboxes)
                
                # bboxes = bboxes[oks]
                # dets = dets[oks]
                
                t4 = time.time()
                print('Light Tracker: {} ms\n'.format((t4-t3)*1000))
                for bbox, det in zip(bboxes, dets):
                    det.predict()
                    det.update(bbox)
            else:
                bboxes = []
                for det in dets:
                    det.predict()
                    new_x = det.mean[0] + det.mean[4]
                    new_y = det.mean[1] + det.mean[5]
                    new_a = det.mean[2] + det.mean[6]
                    new_h = det.mean[3] + det.mean[7]
                    new_w = new_a * new_h
                    bbox = [new_x - new_w/2, new_y - new_h/2, new_w, new_h]
                    # det.update(bbox)
                    bboxes.append(bbox)
                bboxes = np.array(bboxes)


        if i % interval == 0: 
            color = (0, 0, 255)                # red for detector
        elif i % interval <= light_track_frames:
            color = (255, 0, 0)                # blue for light tracker
        else:
            color = (0, 255, 0)                # green for kalman
        # Draw bounding boxes
        for bbox in transform_bbox(bboxes, H0, W0, H, W).astype(np.int32):
            x1, y1, w, h = bbox 
            cv2.rectangle(img0_copy, (x1, y1), (x1 + w, y1 + h), color, 2)
            
        # for det in dets:
        #     bbox = det.bbox
        #     bbox = transform_bbox(bbox, H0, W0, H, W).astype(np.int32)
        #     x1, y1, w, h = bbox 
        #     cv2.rectangle(img0_copy, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        
        cv2.imwrite(osp.join(result_path, filename), img0_copy)