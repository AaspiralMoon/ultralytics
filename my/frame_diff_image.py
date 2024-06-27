import os
import os.path as osp
import cv2
import time
import numpy as np
from utils import mkdir_if_missing

def cal_diff_frame(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray1, gray2)
    
    return diff

if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/MOT17_Det_YOLO/datasets_separated/MOT17-04-SDP/images'
    save_root = '/home/wiser-renjie/projects/yolov8/my/runs/my'
    exp_id = 'MOT17-04-SDP_diff'
    save_path = mkdir_if_missing(osp.join(save_root, exp_id))

    prev_img = None

    for i, img_filename in enumerate(sorted(os.listdir(img_root))):
        print('\n ----------------- Frame : {} ------------------- \n'.format(img_filename))
        img_path = osp.join(img_root, img_filename)
        
        img0 = cv2.imread(img_path)
        
        H, W = 1152, 1920
        img = cv2.resize(img0, (W, H))
        
        if prev_img is not None:
            t1 = time.time()
            diff_img = cal_diff_frame(prev_img, img)
            t2 = time.time()
            print(f'Frame diff time: {(t2-t1)*1000} ms')
            diff_img_filename = 'diff_' + img_filename
            # cv2.imwrite(osp.join(save_path, diff_img_filename), diff_img)
        
        prev_img = img
