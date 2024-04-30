import cv2
import os
import os.path as osp
import numpy as np
from test_crop import load_image

def get_blocks_and_bboxes(image, bboxes):
    H, W = image.shape[:2]
    block_H, block_W = 128, 128
    rows, cols = H // block_H, W // block_W

    bboxes_ext = []
    for bbox in bboxes:
        cls, xcn, ycn, wn, hn = bbox
        xc = xcn * W
        yc = ycn * H
        w = wn * W
        h = hn * H
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        bboxes_ext.append([cls, xc, yc, w, h, x1, y1, x2, y2])

    output_blocks = []
    output_bboxes = []

    for i in range(rows):
        for j in range(cols):
            x_start = j * block_W
            y_start = i * block_H
            x_end = x_start + block_W
            y_end = y_start + block_H

            block_bboxes = []
            for bbox in bboxes_ext:
                cls, xc, yc, w, h, x1, y1, x2, y2 = bbox
                
                if not (x2 < x_start or x1 >= x_end or y2 < y_start or y1 >= y_end):
                    new_xc = (xc - x_start) / block_W
                    new_yc = (yc - y_start) / block_H
                    new_w = w / block_W
                    new_h = h / block_H

                    if xc < x_start:
                        new_xc = new_xc - 1
                    if yc < y_start:
                        new_yc = new_yc - 1

                    block_bboxes.append([0, new_xc, new_yc, new_w, new_h])

            if block_bboxes:
                block = image[y_start:y_end, x_start:x_end]
                output_blocks.append(block)
                output_bboxes.append(block_bboxes)

    return output_blocks, output_bboxes

if __name__ == '__main__':
    block_save_path = '/home/wiser-renjie/datasets/test_partial/test/images'
    label_save_path = '/home/wiser-renjie/datasets/test_partial/test/labels'
    img_root = '/home/wiser-renjie/datasets/temp/test/images'
    label_root = '/home/wiser-renjie/datasets/temp/test/labels'

    idx = 1
    for filename in sorted(os.listdir(img_root)):
        img_path = osp.join(img_root, filename)
        label_path = img_path.replace('images', 'labels').replace('jpg', 'txt')
        
        print(f'Processing: {img_path}')
        
        img = load_image(img_path, (2048, 1024))
        labels = np.loadtxt(label_path)
        output_blocks, output_bboxes = get_blocks_and_bboxes(img, labels)
        
        for block, block_bboxes in zip(output_blocks, output_bboxes):
            block_filename = osp.join(block_save_path, f'{idx:08}.jpg')
            label_filename = osp.join(label_save_path, f'{idx:08}.txt')
            cv2.imwrite(block_filename, block)
            np.savetxt(label_filename, block_bboxes)
            idx += 1
