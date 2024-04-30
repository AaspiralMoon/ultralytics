import cv2
import os
import re
import random
import os.path as osp
import numpy as np
from ultralytics import YOLO
from test_crop import get_idx, load_image

def mask_image(img, bboxes, min_patch=1, scale_factor=0.6):
    h, w, _ = img.shape
    
    num_object = len(bboxes)
    if num_object == 0:
        return img  # No boxes, return unmodified image

    min_patch = num_object // 2
    num_patch = random.randint(min_patch, max(min_patch, num_object))
    
    # Ensure bboxes is a list for random sampling
    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.tolist()
    
    # Shuffle bboxes to randomize selection and avoid duplicates
    shuffled_bboxes = random.sample(bboxes, len(bboxes))
    
    for i in range(num_patch):
        bbox = shuffled_bboxes[i]
        obj_w = bbox[2] - bbox[0]
        obj_h = bbox[3] - bbox[1]

        # Determine size of the patch
        patch_w = int(obj_w * scale_factor)
        patch_h = int(obj_h * scale_factor)

        min_x = max(0, bbox[0] - patch_w)
        min_y = max(0, bbox[1] - patch_w)
        
        # Choose a random position for the patch within the object's area
        x1 = random.randint(min_x, bbox[2] - 1)
        y1 = random.randint(min_y, bbox[3] - 1)
        x2 = min(w, x1 + patch_w)
        y2 = min(h, y1 + patch_w)

        # Apply the patch
        img[y1:y2, x1:x2] = np.full((patch_w, patch_w, 3), 127.5, dtype=np.uint8)  # Gray fill

    return img

if __name__ == '__main__':
    result_root = '/home/wiser-renjie/projects/yolov8/my/runs/detect/'
    save_path = osp.join(result_root, 'predict'+get_idx(result_root))
    img_path = '/home/wiser-renjie/remote_datasets/cityscapes/leftImg8bit_sequence/train/jena/jena_000066_000002_leftImg8bit.png'
    
    img = load_image(img_path, (2048, 1024))
    
    Yolox = YOLO('yolov8x.pt')
    
    results = Yolox.predict(img, save=False, imgsz=(img.shape[0], img.shape[1]), conf=0.5)
    bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
    
    img_masked = mask_image(img, bboxes)
    
    results_masked = Yolox.predict(img_masked, save=True, imgsz=(img.shape[0], img.shape[1]), conf=0.5)