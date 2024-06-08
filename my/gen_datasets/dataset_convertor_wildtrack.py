import os
import shutil
import os.path as osp
import json
import numpy as np
from myMedianTracker import mkdir_if_missing

def copy_img(img_path, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    image_name = os.path.basename(img_path)
    target_image_path = os.path.join(dir, image_name)
    shutil.copy2(img_path, target_image_path)
    # print(f"Image {img_path} copied to {target_image_path}")
    
def xyxy2xywhn(x, w=640, h=640):
    x = np.asarray(x, dtype=np.float32)
    y = np.empty_like(x)
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / w  # x center
    y[..., 2] = ((x[..., 2] + x[..., 4]) / 2) / h  # y center
    y[..., 3] = (x[..., 3] - x[..., 1]) / w  # width
    y[..., 4] = (x[..., 4] - x[..., 2]) / h  # height
    return y.tolist()

json_root = '/home/wiser-renjie/remote_datasets/wildtrack/raw/Wildtrack_dataset/Wildtrack_dataset/annotations_positions'
img_root = '/home/wiser-renjie/remote_datasets/wildtrack/raw/Wildtrack_dataset/Wildtrack_dataset/Image_subsets'
save_root = '/home/wiser-renjie/remote_datasets/wildtrack/datasets_separated'

H = 1080
W = 1920

for filename in sorted(os.listdir(json_root)):                 # each frame
    annos_file = open(osp.join(json_root, filename))
    img_filename = filename.replace('json', 'png')
    label_filename = filename.replace('json', 'txt')
    
    
    annos = json.load(annos_file)
    
    objs = {
        '1': [],
        '2': [],
        '3': [],
        '4': [],
        '5': [],
        '6': [],
        '7': []
    }
    
    for anno in annos:                    # each anno is an object
        
        views = anno['views']
        
        for view in views:
            viewNum = view['viewNum']
            xmax = view['xmax']
            xmin = view['xmin']
            ymax = view['ymax']
            ymin = view['ymin']
            
            if xmax == -1 or xmin == -1 or ymax == -1 or ymin == -1:
                continue
            
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(W, xmax)
            ymax = min(H, ymax)
            
            objs[f'{viewNum + 1}'].append(xyxy2xywhn([0, xmin, ymin, xmax, ymax], w=1920, h=1080))


    # copy image and create labels
    for i in range(1, 8):
        img_file = osp.join(img_root, f'C{i}', img_filename)
        img_save_path = osp.join(save_root, f'C{i}', 'images')
        copy_img(img_file, img_save_path)
        label_save_path = img_save_path.replace('images', 'labels')
        label_file = osp.join(label_save_path, label_filename)
        np.savetxt(label_file, np.asarray(objs[str(i)]))