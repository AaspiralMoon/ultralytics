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
        
        H, W = 1152, 1920
        img = cv2.resize(img0, (W, H))
        img_copy = img.copy()
        
        if i % interval == 0:
            trackers.clear()
            results = model.predict(img, save_txt=False, save=False, classes=[0], imgsz=(img.shape[0], img.shape[1]), conf=0.2)
            bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
            scores = results[0].boxes.conf.cpu().numpy().astype(np.float32)
            clses = results[0].boxes.cls.cpu().numpy()
            
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox.astype(np.int32) 
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
            cluster_bboxes, cluster_dic, cluster_num = dbscan_clustering(bboxes)
            # interval = detection_rate_adjuster(cluster_num)
            # print(f'Updated interval to {interval}')
            
            if cluster_dic:
                hard_regions = [compute_union(cluster_bboxes[x], (H, W)) for x in cluster_bboxes]
                hard_blocks = np.array([bbox_to_blocks(y, 128) for y in hard_regions], dtype=np.int32)
                
                packed_img, packed_rect = get_merge_info(hard_blocks)
            
                bboxes = check_boundary(bboxes, hard_blocks)
                
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
                results = model.predict(merged_img, save_txt=False, save=True, classes=[0], imgsz=(merged_img.shape[0], merged_img.shape[1]), conf=0.2)
                detector_bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
                
                detector_bboxes = revert_bboxes(detector_bboxes, packed_rect)
                detector_bboxes = tlbr2tlwh(detector_bboxes)
                
            for j, tracker in enumerate(trackers):
                ok, out = tracker.update(img)
                tracker_bboxes.append(out)

            tracker_bboxes = np.array(tracker_bboxes)
              
            bboxes = merge_bboxes(detector_bboxes, tracker_bboxes)

            # cluster_bboxes, cluster_dic, cluster_num = dbscan_clustering(bboxes)
        
            # if cluster_dic:
            #     hard_regions = [compute_union(cluster_bboxes[x], (H, W)) for x in cluster_bboxes]
            #     hard_blocks = np.array([bbox_to_blocks(y, 128) for y in hard_regions], dtype=np.int32)             
            #     packed_img, packed_rect = get_merge_info(hard_blocks)
                
            for tracker_bbox in tracker_bboxes:
                x1, y1, w, h = tracker_bbox.astype(np.int32) 
                cv2.rectangle(img_copy, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            
            for detector_bbox in detector_bboxes:
                x1, y1, w, h = detector_bbox.astype(np.int32) 
                cv2.rectangle(img_copy, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
        
        if cluster_dic:
            img_copy = plot_cluster(img_copy, hard_blocks)
            
        img_copy = plot_grid(img_copy, block_size=128)
        # bboxes = tlwh2xywhn(bboxes, H0, W0)
        # bboxes = np.insert(bboxes, 0, 0, axis=1)
        
        cv2.imwrite(osp.join(save_path, img_filename.replace('png', 'jpg')), img_copy)
        
        # np.savetxt(osp.join(save_path, img_filename.replace('png', 'txt')), bboxes)