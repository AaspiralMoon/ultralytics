import os
import os.path as osp
import cv2
import time
import threading
import queue
import numpy as np
from ultralytics import YOLO
from test_cluster import dbscan_clustering
from myMedianTracker import mkdir_if_missing, scale_bbox
from utils import STrack, tlwh2tlbr, tlbr2tlwh, tlwh2xywhn, detection_rate_adjuster, compute_union, bbox_to_blocks, merge_bboxes, revert_bboxes, check_boundary, plot_cluster, plot_grid, plot_bbox, run_detector, handle_boundary_conflicts, find_bbox_in_hard_region, clip_bbox
from test_merge import get_merge_info, get_merge_img


if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/wildtrack/decoded_images/cam7'
    save_root = '/home/wiser-renjie/projects/yolov8/my/runs/my'
    exp_id = 'test_cam7'
    save_path = mkdir_if_missing(osp.join(save_root, exp_id))
    
    interval = 10
    light_tracker_frame = 3
    
    model = YOLO('/home/wiser-renjie/projects/yolov8/my/weights/yolov8x.pt')
    tracker = cv2.legacy.MultiTracker_create()
    trackers = []
    
    H, W = 1152, 1920
    Ht, Wt = 144, 240  # resolution for tracking
    
    e2e_time_list = []
    
    for i, img_filename in enumerate(sorted(os.listdir(img_root))):
        print('\n ----------------- Frame : {} ------------------- \n'.format(img_filename))
        if i == 3000:
            break
        # if img_filename == '001038.jpg':
        #     break
        img_path = osp.join(img_root, img_filename)
        
        img0 = cv2.imread(img_path)
        img = cv2.resize(img0, (W, H))
        img_copy = img.copy()
        
        if i % interval == 0:
            trackers.clear()
            results = model.predict(img, save_txt=False, save=False, classes=[0], imgsz=(img.shape[0], img.shape[1]), conf=0.3)
            bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
            confs = results[0].boxes.conf.cpu().numpy().astype(np.float32)
            clses = results[0].boxes.cls.cpu().numpy().astype(np.int32)
            detector_time = results[0].speed['preprocess'] + results[0].speed['inference'] + results[0].speed['postprocess']

            bboxes = np.hstack((clses[:, None], bboxes, confs[:, None]))

            img_copy = plot_bbox(img_copy, bboxes[:, 1:5], color=(0, 0, 255))
            
            cluster_start = time.time()
            _, cluster_bboxes, cluster_dic, cluster_num = dbscan_clustering(bboxes)
            # interval = detection_rate_adjuster(cluster_num)
            # print(f'Updated interval to {interval}')
            
            if cluster_dic:
                hard_regions = [compute_union(cluster_bboxes[x], (H, W)) for x in cluster_bboxes]
                hard_blocks = np.array([bbox_to_blocks(y, 128) for y in hard_regions], dtype=np.int32)
                hard_bboxes = find_bbox_in_hard_region(bboxes, hard_blocks)
                packed_img, packed_rect = get_merge_info(hard_blocks)
            
                bboxes = check_boundary(bboxes, hard_blocks)
                
            bboxes = tlbr2tlwh(bboxes)
            confs = bboxes[..., 0]
            clses = bboxes[..., 5]
            stracks = np.array([STrack(tlwh) for tlwh in bboxes[..., 1:5]], dtype=object)
            cluster_end = time.time()
            cluster_time = (cluster_end - cluster_start)*1000
            
            # resize img for fast tracking
            resize_start = time.time()
            img_track = cv2.resize(img0, (Wt, Ht)) # small img for tracking
            resize_end = time.time()
            resize_time = (resize_end - resize_start)*1000
                
            # init light tracker
            tracker_init_start = time.time()
            for bbox in scale_bbox(bboxes, H, W, Ht, Wt):
                _, x1, y1, w, h, _ = bbox
                tracker = cv2.legacy.TrackerMedianFlow_create()
                tracker.init(img_track, (x1, y1, w, h))
                trackers.append(tracker)
            tracker_init_end = time.time()
            tracker_init_time = (tracker_init_end - tracker_init_start)*1000
            
            e2e_time = detector_time + cluster_time + resize_time + tracker_init_time

        else:
            tracker_bboxes = []
            detector_bboxes = []
            
            q = queue.Queue()
            
            detector_tracker_start = time.time()
            if cluster_dic:
                merged_img = get_merge_img(img, packed_img, packed_rect)
                detector_thread = threading.Thread(target=run_detector, args=(merged_img, model, merged_img.shape[0], merged_img.shape[1], q))
                detector_thread.start()
            
            # use light tracker to tracker N frames and init 
            if i % interval <= light_tracker_frame:                 
                # do light tracking
                tracker_start = time.time()
                for j, tracker in enumerate(trackers):
                    ok, out = tracker.update(img_track)
                    x1, y1, w, h = out
                    cls = clses[j]
                    conf = confs[j]
                    tracker_bboxes.append([cls, x1, y1, w, h, conf])
                tracker_bboxes = scale_bbox(tracker_bboxes, Ht, Wt, H, W)
                tracker_end = time.time()
                tracker_time = (tracker_end - tracker_start)*1000
                
            else:
                # kalman filter for remaining frames
                kalman_start = time.time()
                for strack in stracks:
                    strack.predict()
                    bbox = strack.bbox
                    tracker_bboxes.append(bbox)
                tracker_bboxes = np.array(tracker_bboxes)
                tracker_bboxes = np.hstack((clses[:, None], tracker_bboxes, confs[:, None]))
                kalman_end = time.time()
                kalman_time = (kalman_end - kalman_start)*1000
                
            if cluster_dic:    
                detector_thread.join()
                detector_bboxes, detector_time = q.get()
            # detector bboxes post processing
            if cluster_dic:
                detector_bboxes = revert_bboxes(detector_bboxes, packed_rect)
                detector_bboxes = clip_bbox(detector_bboxes, H, W)
                t1 = time.time()
                detector_bboxes = handle_boundary_conflicts(hard_bboxes, detector_bboxes, dist_thresh=50, type='dist')
                t2 = time.time()
                print(f'Boundary time: {(t2-t1)*1000} ms')
                detector_bboxes = tlbr2tlwh(detector_bboxes)
            
            bboxes = merge_bboxes(detector_bboxes, tracker_bboxes)

            detector_tracker_end = time.time()
            detector_tracker_time = (detector_tracker_end - detector_tracker_start)*1000
            
            e2e_time = detector_tracker_time
            
            tracker_bbox_color = (255, 0, 0) if i % interval <= light_tracker_frame else (0, 255, 0)
            img_copy = plot_bbox(img_copy, tlwh2tlbr(tracker_bboxes[..., 1:5]), color=tracker_bbox_color)
            img_copy = plot_bbox(img_copy, tlwh2tlbr(detector_bboxes[..., 1:5]), color=(0, 0, 255))

        if cluster_dic:
            img_copy = plot_cluster(img_copy, hard_blocks)
            
        img_copy = plot_grid(img_copy, block_size=128)
        
        cv2.imwrite(osp.join(save_path, img_filename.replace('png', 'jpg')), img_copy)
        
        e2e_time_list.append(e2e_time)
        # np.savetxt(osp.join(save_path, img_filename.replace('jpg', 'txt')), tlwh2xywhn(bboxes, H, W), fmt='%.6f')
    print(e2e_time_list)
    print(np.array(e2e_time_list).mean())