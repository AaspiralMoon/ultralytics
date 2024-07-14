import os
import cv2
import time
import numpy as np
import os.path as osp
from ultralytics import YOLO as yolo
from utils import mkdir_if_missing, tlbr2tlwh, tlwh2xywhn, scale_bbox

if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/wildtrack/decoded_images/cam7'
    result_root = '/home/wiser-renjie/projects/yolov8/my/runs/my'
    exp_id = 'wildtrack_cam7_medianflow_yolox_1152_1920_0.3_576_960_i10'
    result_path = mkdir_if_missing(osp.join(result_root, exp_id))
    
    interval = 10
    
    model = yolo('/home/wiser-renjie/projects/yolov8/my/weights/yolov8x.pt')
    tracker = cv2.legacy.MultiTracker_create()
    trackers = []
    
    H, W = 1152, 1920
    Ht, Wt = 576, 960  # resolution for tracking
    
    processing_times = []
    
    for i, filename in enumerate(sorted(os.listdir(img_root))):
        print('\n ----------------- Frame : {} ------------------- \n'.format(filename))
        if i == 3000:
            break
        img0 = cv2.imread(osp.join(img_root, filename))
        img = cv2.resize(img0, (W, H))
        img_track = cv2.resize(img0, (Wt, Ht))         # small img for tracking
        img_copy = img.copy()
        
        if i % interval == 0:
            trackers.clear()
            results = model.predict(img, save=False, imgsz=(H, W), classes=[0], conf=0.3)
            bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
            confs = results[0].boxes.conf.cpu().numpy().astype(np.float32)
            clses = results[0].boxes.cls.cpu().numpy().astype(np.int32)
            detector_time = results[0].speed['preprocess'] + results[0].speed['inference'] + results[0].speed['postprocess']
            
            bboxes = np.hstack((clses[:, None], bboxes, confs[:, None]))
            
            bboxes = tlbr2tlwh(bboxes)
            
            t1 = time.time()
            for bbox in scale_bbox(bboxes[..., 1:5], H, W, Ht, Wt):
                x1, y1, w, h= bbox
                tracker = cv2.legacy.TrackerMedianFlow_create()
                tracker.init(img_track, (x1, y1, w, h))
                trackers.append(tracker)
            t2 = time.time()
            init_time = (t2-t1)*1000
            print('Init time: {} ms\n'.format(init_time))

            processing_times.append(detector_time+init_time)
        else:
            bboxes = []
            t3 = time.time()
            for j, tracker in enumerate(trackers):
                ok, out = tracker.update(img_track)
                bboxes.append(out)
            
            bboxes = scale_bbox(bboxes, Ht, Wt, H, W)
            bboxes = np.hstack((clses[:, None], bboxes, confs[:, None]))
            t4 = time.time()
            track_time = (t4-t3)*1000
            print('Track time: {} ms\n'.format(track_time))
            processing_times.append(track_time)
        
        color = (0, 0, 255) if i % interval == 0 else (255, 0, 0)
        # Draw bounding boxes
        for bbox in bboxes.astype(np.int32):
            _, x1, y1, w, h, _ = bbox 
            cv2.rectangle(img_copy, (x1, y1), (x1 + w, y1 + h), color, 2)
        
        # cv2.imwrite(osp.join(result_path, filename), img_copy)
        
        np.savetxt(osp.join(result_path, filename.replace('jpg', 'txt').replace('png', 'txt')), tlwh2xywhn(bboxes, H, W), fmt='%.6f')
        
        print(f'Average processing time: {np.array(processing_times).mean()} ms')
        print(f'Average FPS: {round(1000/np.array(processing_times).mean(), 2)}')