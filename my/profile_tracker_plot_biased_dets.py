import os
import os.path as osp
import cv2
import time
import numpy as np
from ultralytics import YOLO
from test_cluster import dbscan_clustering
from myMedianTracker import mkdir_if_missing
from utils import STrack, tlwh2tlbr, tlbr2tlwh, tlwh2xywhn, bbox_to_blocks, plot_cluster, plot_grid, plot_bbox, plot_bbox_with_labels, find_FPs, get_best_iou
from test_merge import get_merge_info, get_merge_img

def load_gt(path, H, W):
    gt = np.loadtxt(path, dtype=np.float32)
    y = np.empty_like(gt)
    
    if y.size == 0:
        return y
    
    if y.shape[1] == 5:  # [cls, xcn, ycn, wn, hn]
        cls = gt[:, 0]
        xc = gt[:, 1] * W
        yc = gt[:, 2] * H
        w = gt[:, 3] * W
        h = gt[:, 4] * H

        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2

        y[..., 0] = cls
        y[..., 1] = x1
        y[..., 2] = y1
        y[..., 3] = x2
        y[..., 4] = y2
    else:
        raise NotImplementedError("Input shape not supported")
    
    return y

if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/MOT17_Det_YOLO/datasets_separated/MOT17-09-SDP/images'
    save_root = '/home/wiser-renjie/projects/yolov8/my/runs/my'
    exp_id = 'profile_tracker_MOT17-09-SDP_i10_iou50_with_iou'
    save_path = mkdir_if_missing(osp.join(save_root, exp_id))
    gt_root = '/home/wiser-renjie/remote_datasets/MOT17_Det_YOLO/datasets_separated/MOT17-09-SDP/labels'
    
    interval = 10
    
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
        
        gt = load_gt(osp.join(gt_root, img_filename.replace('jpg', 'txt').replace('png', 'txt')), H, W)[..., 1:]
        
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
            # results = model.predict(img, save_txt=False, save=False, classes=[0], imgsz=(img.shape[0], img.shape[1]), conf=0.3)
            # bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
            # confs = results[0].boxes.conf.cpu().numpy().astype(np.float32)
            # clses = results[0].boxes.cls.cpu().numpy().astype(np.int32)
            
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
            
            missing_bboxes = tlwh2tlbr(missing_bboxes)
            tracked_bboxes = tlwh2tlbr(tracked_bboxes)
            ious_tracked_bboxes = get_best_iou(tracked_bboxes, gt)
            img_copy = plot_bbox(img_copy, missing_bboxes, color=(0, 255, 0))
            img_copy = plot_bbox_with_labels(img_copy, tracked_bboxes, ious_tracked_bboxes, thresh=0.5, color=(255, 0, 0))
            
        # img_copy = plot_bbox(img_copy, gt, color=(0, 165, 255))
        cv2.imwrite(osp.join(save_path, img_filename), img_copy)
        

            
        