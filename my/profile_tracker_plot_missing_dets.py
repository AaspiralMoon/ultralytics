import os
import os.path as osp
import cv2
import time
import numpy as np
from ultralytics import YOLO
from test_cluster import dbscan_clustering
from myMedianTracker import mkdir_if_missing
from utils import STrack, tlwh2tlbr, tlbr2tlwh, tlwh2xywhn, bbox_to_blocks, plot_cluster, plot_grid, plot_bbox
from test_merge import get_merge_info, get_merge_img

# def cal_hit_rate(hard_regions, bboxes):
#     def bbox_intersects_with_hard_regions(bbox, hard_regions):
#         for hr in hard_regions:
#             if not (bbox[2] < hr[0] or bbox[0] > hr[2] or bbox[3] < hr[1] or bbox[1] > hr[3]):
#                 return True
#         return False
    
#     total_bboxes = len(bboxes)
#     if total_bboxes == 0:
#         return 0.0
    
#     intersecting_bboxes = sum(1 for bbox in bboxes if bbox_intersects_with_hard_regions(bbox, hard_regions))
    
#     return intersecting_bboxes / total_bboxes

# def cal_hit_rate(list_A, list_B):
#     if not list_A:
#         return -1 if not list_B else 0
#     if not list_B:
#         return 0
#     return sum(1 for elem in list_A if elem in list_B) / len(list_A)

if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/MOT17_Det_YOLO/datasets_separated/MOT17-04-SDP/images'
    save_root = '/home/wiser-renjie/projects/yolov8/my/runs/my'
    exp_id = 'profile_tracker_MOT17-04-SDP_30'
    save_path = mkdir_if_missing(osp.join(save_root, exp_id))
    
    interval = 30
    
    model = YOLO('/home/wiser-renjie/projects/yolov8/my/weights/yolov8x_MOT17.pt')
    tracker = cv2.legacy.MultiTracker_create()
    trackers = []
    
    for i, img_filename in enumerate(sorted(os.listdir(img_root))):
        print('\n ----------------- Frame : {} ------------------- \n'.format(img_filename))
        if i == 3500:
            break
        
        img_path = osp.join(img_root, img_filename)
        
        img0 = cv2.imread(img_path)
        
        H, W = 1152, 1920
        img = cv2.resize(img0, (W, H))
        img_copy = img.copy()
        
        img_copy = plot_grid(img_copy, block_size=128)
        
        if i % interval == 0:
            trackers.clear()
            results = model.predict(img, save_txt=False, save=False, classes=[0], imgsz=(img.shape[0], img.shape[1]), conf=0.3)
            bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
            confs = results[0].boxes.conf.cpu().numpy().astype(np.float32)
            clses = results[0].boxes.cls.cpu().numpy().astype(np.int32)

            bboxes = tlbr2tlwh(bboxes)
                    
            for bbox in bboxes:
                x1, y1, w, h = bbox
                tracker = cv2.legacy.TrackerMedianFlow_create()
                tracker.init(img, (x1, y1, w, h))
                trackers.append(tracker)
            
            img_copy = plot_bbox(img_copy, tlwh2tlbr(bboxes), color=(0, 0, 255))
            prev_tracker_bboxes = bboxes

        else:
            results = model.predict(img, save_txt=False, save=False, classes=[0], imgsz=(img.shape[0], img.shape[1]), conf=0.3)
            bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
            confs = results[0].boxes.conf.cpu().numpy().astype(np.float32)
            clses = results[0].boxes.cls.cpu().numpy().astype(np.int32)


            tracker_bboxes = []
            missing_bboxes = []
            tracked_bboxes = []
            for j, tracker in enumerate(trackers):
                ok, out = tracker.update(img)
                if not ok:
                    missing_bboxes.append(prev_tracker_bboxes[j])
                    tracker_bboxes.append(prev_tracker_bboxes[j])
                else:
                    tracked_bboxes.append(out)
                    tracker_bboxes.append(out)
                    
            tracker_bboxes = np.array(tracker_bboxes)
            
            prev_tracker_bboxes = tracker_bboxes
            
            if missing_bboxes:
                missing_bboxes = tlwh2tlbr(missing_bboxes)

                hard_blocks = np.array([bbox_to_blocks(b, block_size=128) for b in missing_bboxes], dtype=np.int32)
                 
                img_copy = plot_cluster(img_copy, hard_blocks)
            
            img_copy = plot_bbox(img_copy, missing_bboxes, color=(0, 255, 0))
            img_copy = plot_bbox(img_copy, tlwh2tlbr(tracked_bboxes), color=(255, 0, 0))
    
        
        cv2.imwrite(osp.join(save_path, img_filename), img_copy)
        

            
        