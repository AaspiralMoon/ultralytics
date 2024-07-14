import os
import os.path as osp
import numpy as np
import cv2
from ultralytics import YOLO

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)
    return d

def plot_bbox(img, bboxes, color=(0, 0, 255), thickness=2):
    bboxes = np.asarray(bboxes, dtype=np.int32)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/MOT17_Det_YOLO/datasets_separated_splitted/MOT17-04-SDP/test/images'
    save_root = '/home/wiser-renjie/projects/yolov8/my/runs/my'
    exp_id = 'MOT17-04-SDP_yolov8x'
    save_path = mkdir_if_missing(osp.join(save_root, exp_id))
    
    model = YOLO('/home/wiser-renjie/projects/yolov8/my/weights/yolov8x_MOT17.pt')
    
    H = 1152
    W = 1920
    
    for idx, img_filename in enumerate(sorted(os.listdir(img_root))):
        if idx == 3000:
            break
        img_path = osp.join(img_root, img_filename)
        img0 = cv2.imread(img_path)
        img = cv2.resize(img0, (W, H))

        preds = model.predict(img, save=False, save_txt=False, save_conf=False, classes=[0], imgsz=(H, W), conf=0.3)
        
        results = []
        bboxes = preds[0].boxes.xyxy.cpu().numpy()
        confs = preds[0].boxes.conf.cpu().numpy()
        clses = preds[0].boxes.cls.cpu().numpy()
        
        for bbox, conf, cls in zip(bboxes, confs, clses):
            x1, y1, x2, y2 = bbox
            xcn = np.clip((x1 + x2) / 2 / W, 0, 1)
            ycn = np.clip((y1 + y2) / 2 / H, 0, 1)
            wn = np.clip((x2 - x1) / W, 0, 1)
            hn = np.clip((y2 - y1) / H, 0, 1)
            result = [cls, xcn, ycn, wn, hn, conf]
            results.append(result)
        txt_path = osp.join(save_path, img_filename.replace('jpg', 'txt').replace('png', 'txt'))
        # np.savetxt(txt_path, np.array(results), fmt='%.6f')
        
        img = plot_bbox(img, bboxes)
        cv2.imwrite(osp.join(save_path, img_filename.replace('png', 'jpg')), img)