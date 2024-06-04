import os
import os.path as osp
import cv2
import time
import numpy as np
from ultralytics import YOLO
from test_cluster import dbscan_clustering
from myMedianTracker import xyxy2xywh, mkdir_if_missing, transform_bbox
from utils import STrack


def tlwh2xywhn(x, H, W):  
    y = np.empty_like(x)
    if y.size != 0:
        y[..., 0] = (x[..., 0] + x[..., 2] / 2) / W
        y[..., 1] = (x[..., 1] + x[..., 3] / 2) / H
        y[..., 2] = x[..., 2] / W
        y[..., 3] = x[..., 3] / H
    return y

def detection_rate_adjuster(cluster_num):
    if cluster_num > 4 :
        detection_rate = 8
    elif cluster_num > 3:
        detection_rate = 9
    elif cluster_num > 2:
        detection_rate = 10
    else:
        detection_rate = 11
    return detection_rate

if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/traffic/video1_30fps'
    save_root = '/home/wiser-renjie/projects/yolov8/my/runs/my'
    cam_id = 'video1_30fps'
    save_path = mkdir_if_missing(osp.join(save_root, cam_id))
    
    interval = 10
    light_track_frames = 3
    
    model = YOLO('yolov8x.pt')
    tracker = cv2.legacy.MultiTracker_create()
    trackers = []
    
    
    for i, img_filename in enumerate(sorted(os.listdir(img_root))):
        
        img_path = osp.join(img_root, img_filename)
        
        img0 = cv2.imread(img_path)
        
        H0, W0 = img0.shape[:2]
        
        img0_copy = img0.copy()
        
        H, W = 640, 640
        img = cv2.resize(img0, (H, W))
        
        if i % interval == 0:
            trackers.clear()
            results = model.predict(img, save_txt=False, save=False, classes=[2], imgsz=640, conf=0.5)
            bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
            scores = results[0].boxes.conf.cpu().numpy().astype(np.float32)
            
            # cluster_dic, cluster_num = dbscan_clustering(bboxes)
            # interval = detection_rate_adjuster(cluster_num)
            # print(f'Updated interval to {interval}')
            
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
                
                bboxes = []
                
                for j, tracker in enumerate(trackers):
                    ok, bbox = tracker.update(img)
                    if not ok:
                        dets[j].predict()
                        bbox = dets[j].bbox
                    bboxes.append(bbox)
                    
                bboxes = np.array(bboxes)
                
                t4 = time.time()
                print('Light Tracker: {} ms\n'.format((t4-t3)*1000))
                
                for bbox, det in zip(bboxes, dets):
                    det.predict()
                    det.update(bbox)
                    
            else:
                bboxes = []
                for det in dets:
                    det.predict()
                    bbox = det.bbox
                    # det.update(bbox)
                    bboxes.append(bbox)
                bboxes = np.array(bboxes)
    
    
        bboxes = transform_bbox(bboxes, H0, W0, H, W)
        
        if i % interval == 0: 
            color = (0, 0, 255)                # red for detector
            print('red')
        elif i % interval <= light_track_frames:
            color = (255, 0, 0)                # blue for light tracker
            print('blue')
        else:
            color = (0, 255, 0)                # green for kalman
            print('green')
        # Draw bounding boxes
        for bbox in bboxes.astype(np.int32):
            x1, y1, w, h = bbox 
            cv2.rectangle(img0_copy, (x1, y1), (x1 + w, y1 + h), color, 2)
        
        bboxes = tlwh2xywhn(bboxes, H0, W0)
        bboxes = np.insert(bboxes, 0, 0, axis=1)
        
        cv2.imwrite(osp.join(save_path, img_filename.replace('png', 'jpg')), img0_copy)
        
        # np.savetxt(osp.join(save_path, img_filename.replace('png', 'txt')), bboxes)