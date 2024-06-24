import cv2
import os
import numpy as np
import os.path as osp
from ultralytics import YOLO
import time
from test_cluster import dbscan_clustering
from myMedianTracker import mkdir_if_missing, transform_bbox
from utils import STrack, tlwh2tlbr, tlbr2tlwh, tlwh2xywhn, detection_rate_adjuster, compute_union, bbox_to_blocks, merge_bboxes, revert_bboxes, check_boundary
from test_merge import get_merge_info, get_merge_img

def plot_cluster(img, hard_blocks):
    overlay = img.copy()
    alpha = 0.5
    
    for x1, y1, x2, y2 in hard_blocks:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
    
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    return img

def plot_grid(img, block_size=128):
    h, w, _ = img.shape
    
    for y in range(0, h, block_size):
        cv2.line(img, (0, y), (w, y), (255, 0, 0), 1)
    for x in range(0, w, block_size):
        cv2.line(img, (x, 0), (x, h), (255, 0, 0), 1)
    
    return img

def plot_bbox(img, bboxes, color=(0, 255, 0), thickness=2):
    
    for bbox in bboxes.astype(np.int32):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
    return img
    
if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/wildtrack/decoded_images/cam7'
    save_root = '/home/wiser-renjie/projects/yolov8/my/runs/my'
    cam_id = 'wildtrack_cam7_cluster_1152_1920_200_2_0.1_TOP3000'
    save_path = mkdir_if_missing(osp.join(save_root, cam_id))
    
    model = YOLO('yolov8x.pt')

    H = 1152
    W = 1920
    block_size = 128
    
    for i, img_filename in enumerate(sorted(os.listdir(img_root))):
        print('\n ----------------- Frame : {} ------------------- \n'.format(img_filename))
        if i == 3000:
            break
        img_path = osp.join(img_root, img_filename)
        img0 = cv2.imread(img_path)
        
        img = cv2.resize(img0, (W, H))
           
        results = model.predict(img, save_txt=False, save=False, classes=[0], imgsz=(H, W), conf=0.3)

        bboxes = results[0].boxes.xyxy.cpu().numpy()
        
        t1 = time.time()
        cluster_bboxes, cluster_dic, cluster_num = dbscan_clustering(bboxes)
        t2 = time.time()
        print(f'cluster time: {(t2-t1)*1000} ms')
        
        img = plot_grid(img, block_size=128)
        img = plot_bbox(img, bboxes)
        
        if cluster_dic:
            hard_regions = [compute_union(cluster_bboxes[x], (H, W)) for x in cluster_bboxes]
            
            hard_blocks = np.array([bbox_to_blocks(y, 128) for y in hard_regions], dtype=np.int32)

            packed_img, packed_rect = get_merge_info(hard_blocks)
            merged_img = get_merge_img(img, packed_img, packed_rect)
            
            img = plot_cluster(img, hard_blocks)
            # cv2.imwrite(osp.join(save_path, img_filename.replace('png', 'jpg')), merged_img)
            
        cv2.imwrite(osp.join(save_path, img_filename.replace('png', 'jpg')), img)
        