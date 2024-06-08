import os
import os.path as osp
import cv2
import time
import numpy as np
from ultralytics import YOLO
from test_cluster import dbscan_clustering
from myMedianTracker import mkdir_if_missing, transform_bbox
from utils import STrack, tlwh2tlbr, tlbr2tlwh, tlwh2xywhn, detection_rate_adjuster, compute_union, bbox_to_blocks, merge_bboxes, revert_bboxes, check_boundary
from test_merge import get_merge_info, get_merge_img


if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/wildtrack/decoded_images/cam7'
    save_root = '/home/wiser-renjie/projects/yolov8/my/runs/my'
    cam_id = 'cam7'
    save_path = mkdir_if_missing(osp.join(save_root, cam_id))
    
    interval = 10
    
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
            results = model.predict(img, save_txt=False, save=False, classes=[0], imgsz=640, conf=0.5)
            bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
            scores = results[0].boxes.conf.cpu().numpy().astype(np.float32)
            
            cluster_bboxes, cluster_dic, cluster_num = dbscan_clustering(bboxes)
            # interval = detection_rate_adjuster(cluster_num)
            # print(f'Updated interval to {interval}')
            
            if cluster_dic:
                hard_regions = [compute_union(cluster_bboxes[x], (H, W)) for x in cluster_bboxes]
                
                packed_img, packed_rect = get_merge_info(hard_regions)
            
                bboxes = check_boundary(bboxes, hard_regions)
                
            bboxes = tlbr2tlwh(bboxes)
                    
            t1 = time.time()
            for bbox in bboxes:
                x1, y1, w, h = bbox
                tracker = cv2.legacy.TrackerMedianFlow_create()
                tracker.init(img, (x1, y1, w, h))
                trackers.append(tracker)
            t2 = time.time()
            # print('Init time: {} ms\n'.format((t2-t1)*1000))
            
        else:
            tracker_bboxes = []
            detector_bboxes = []
        
            if cluster_dic:
                merged_img = get_merge_img(img, packed_img, packed_rect)
                results = model.predict(merged_img, save_txt=False, save=True, classes=[0], imgsz=(merged_img.shape[0], merged_img.shape[1]), conf=0.5)
                detector_bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
                
                detector_bboxes = revert_bboxes(detector_bboxes, packed_rect)
            
            for j, tracker in enumerate(trackers):
                ok, out = tracker.update(img)
                tracker_bboxes.append(out)
                
            tracker_bboxes = np.array(tracker_bboxes)
            
    
            bboxes = merge_bboxes(detector_bboxes, tracker_bboxes)
        
        bboxes = transform_bbox(bboxes, H0, W0, H, W)
        
        if i % interval == 0: 
            color = (0, 0, 255)                # red for detector
        else:
            color = (255, 0, 0)                # blue for light tracker
        # Draw bounding boxes
        for bbox in bboxes.astype(np.int32):
            x1, y1, w, h = bbox 
            cv2.rectangle(img0_copy, (x1, y1), (x1 + w, y1 + h), color, 2)
        
        bboxes = tlwh2xywhn(bboxes, H0, W0)
        bboxes = np.insert(bboxes, 0, 0, axis=1)
        
        cv2.imwrite(osp.join(save_path, img_filename.replace('png', 'jpg')), img0_copy)
        
        # np.savetxt(osp.join(save_path, img_filename.replace('png', 'txt')), bboxes)