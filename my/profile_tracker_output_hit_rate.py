import os
import os.path as osp
import cv2
import time
import numpy as np
from ultralytics import YOLO
from test_cluster import dbscan_clustering
from myMedianTracker import mkdir_if_missing, transform_bbox
from utils import STrack, tlwh2tlbr, tlbr2tlwh, tlwh2xywhn, detection_rate_adjuster, compute_union, bbox_to_blocks, merge_bboxes, revert_bboxes, check_boundary, plot_cluster, plot_grid
from test_merge import get_merge_info, get_merge_img

def cal_hit_rate(hard_regions, bboxes):
    def bbox_intersects_with_hard_regions(bbox, hard_regions):
        for hr in hard_regions:
            if not (bbox[2] < hr[0] or bbox[0] > hr[2] or bbox[3] < hr[1] or bbox[1] > hr[3]):
                return True
        return False
    
    total_bboxes = len(bboxes)
    if total_bboxes == 0:
        return 0.0
    
    intersecting_bboxes = sum(1 for bbox in bboxes if bbox_intersects_with_hard_regions(bbox, hard_regions))
    
    return intersecting_bboxes / total_bboxes

# def cal_hit_rate(list_A, list_B):
#     if not list_A:
#         return -1 if not list_B else 0
#     if not list_B:
#         return 0
#     return sum(1 for elem in list_A if elem in list_B) / len(list_A)

if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/wildtrack/decoded_images/cam7'
    save_root = '/home/wiser-renjie/projects/yolov8/my/runs/my'
    cam_id = 'wildtrack_cam7_my_1152_1920_0.3_i10_TOP3000'
    save_path = mkdir_if_missing(osp.join(save_root, cam_id))
    
    interval = 3
    
    model = YOLO('/home/wiser-renjie/projects/yolov8/my/weights/yolov8x_MOT17.pt')
    tracker = cv2.legacy.MultiTracker_create()
    trackers = []
    
    hit_rate_list = []
    for i, img_filename in enumerate(sorted(os.listdir(img_root))):
        print('\n ----------------- Frame : {} ------------------- \n'.format(img_filename))
        if i == 3000:
            break
        
        img_path = osp.join(img_root, img_filename)
        
        img0 = cv2.imread(img_path)
        
        H, W = 1152, 1920
        img = cv2.resize(img0, (W, H))
        img_copy = img.copy()
        
        if i % interval == 0:
            trackers.clear()
            results = model.predict(img, save_txt=False, save=False, classes=[0], imgsz=(img.shape[0], img.shape[1]), conf=0.3)
            bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)

            bboxes = tlbr2tlwh(bboxes)
                    
            for bbox in bboxes:
                x1, y1, w, h = bbox
                tracker = cv2.legacy.TrackerMedianFlow_create()
                tracker.init(img, (x1, y1, w, h))
                trackers.append(tracker)

            prev_tracker_bboxes = bboxes
        else:
            tracker_bboxes = []
            missing_bboxes = []
            
            results = model.predict(img, save_txt=False, save=False, classes=[0], imgsz=(img.shape[0], img.shape[1]), conf=0.3)
            bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)

            cluster_bboxes, cluster_dic, cluster_num = dbscan_clustering(bboxes)
            
            if cluster_dic:
                hard_regions = [compute_union(cluster_bboxes[x], (H, W)) for x in cluster_bboxes]
                hard_blocks = np.array([bbox_to_blocks(y, 128) for y in hard_regions], dtype=np.int32)
                
            for j, tracker in enumerate(trackers):
                ok, out = tracker.update(img)
                x1, y1, w, h = out
                if not ok:
                    missing_bboxes.append(prev_tracker_bboxes[j])
                    tracker_bboxes.append(prev_tracker_bboxes[j])
                else:
                    tracker_bboxes.append(out)

            tracker_bboxes = np.array(tracker_bboxes)

            if len(missing_bboxes) !=0 and cluster_dic:
                hit_rate = cal_hit_rate(hard_blocks, tlwh2tlbr(missing_bboxes))
                hit_rate_list.append(hit_rate)
    
    print(f'Average hit rate: {np.array(hit_rate_list).mean()}')
            
        