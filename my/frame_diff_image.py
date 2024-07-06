import os
import os.path as osp
import cv2
import time
import numpy as np
from utils import mkdir_if_missing

def cal_diff_frame(img1, img2):
    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # diff = cv2.absdiff(gray1, gray2)
    diff = cv2.absdiff(img1, img2)
    return diff

def draw_blocks(img, diff, block_size, diff_thresh):
    H, W = img.shape[:2]

    for y in range(0, H, block_size):
        for x in range(0, W, block_size):
            block = diff[y:y + block_size, x:x + block_size]
            block_mean_diff = np.mean(block)

            # 画网格线
            cv2.rectangle(img, (x, y), (x + block_size, y + block_size), (255, 255, 255), 1)

            # 在网格中心显示差异值
            text_x = x + block_size // 2
            text_y = y + block_size // 2
            cv2.putText(img, f'{block_mean_diff:.1f}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if block_mean_diff > diff_thresh else (255, 0, 0), 1, cv2.LINE_AA)

    return img

if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/wildtrack/decoded_images/cam7'
    save_root = '/home/wiser-renjie/projects/yolov8/my/runs/my'
    exp_id = 'cam7_diff_RGB_video'
    save_path = mkdir_if_missing(osp.join(save_root, exp_id))

    prev_img = None
    block_size = 128
    diff_thresh = 2

    for i, img_filename in enumerate(sorted(os.listdir(img_root))):
        print('\n ----------------- Frame : {} ------------------- \n'.format(img_filename))
        if i == 1000:
            break
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
            
            # 绘制网格线并显示差异值
            img_with_blocks = draw_blocks(img.copy(), diff_img, block_size, diff_thresh)
            
            # 保存标记了差异块的原图
            diff_img_with_blocks_filename = 'blocks_' + img_filename
            # cv2.imwrite(osp.join(save_path, diff_img_with_blocks_filename), img_with_blocks)
            cv2.imwrite(osp.join(save_path, diff_img_filename), diff_img)
        prev_img = img
