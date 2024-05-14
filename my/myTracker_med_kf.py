import os
import cv2
import numpy as np
import os.path as osp
from ultralytics import YOLO as yolo

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)
    return d

class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    
    def predict(self):
        return self.kalman.predict()
    
    def correct(self, measurement):
        return self.kalman.correct(np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]]))

if __name__ == '__main__':
    img_root = '/home/wiser-renjie/remote_datasets/traffic/video1'
    result_root = '/home/wiser-renjie/projects/yolov8/my/runs/my'
    exp_id = 'OBDS2'
    result_path = mkdir_if_missing(osp.join(result_root, exp_id))

    model = yolo('yolov8x.pt')
    interval = 5
    trackers = []
    kf = KalmanFilter()

    for i, filename in enumerate(sorted(os.listdir(img_root))):
        img = cv2.imread(osp.join(img_root, filename))
        
        if i % interval == 0:  # Detect with YOLO and initialize trackers
            results = model.predict(img, save=False, classes=[2], conf=0.5)
            bboxes_prev = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
            trackers = [cv2.legacy.TrackerMedianFlow_create() for _ in range(len(bboxes_prev))]
            for tracker, bbox in zip(trackers, bboxes_prev):
                tracker.init(img, tuple(bbox[:4]))
            kf = KalmanFilter()  # Reset Kalman Filter for new detections
        
        else:  # Track with MedianFlow and update with Kalman filter
            new_bboxes = []
            for tracker, bbox_prev in zip(trackers, bboxes_prev):
                ok, bbox = tracker.update(img)
                if ok:
                    bbox = list(map(int, bbox))
                    predicted = kf.predict()
                    measured = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
                    corrected = kf.correct(measured)
                    bbox = [corrected[0] - bbox[2] / 2, corrected[1] - bbox[3] / 2, bbox[2], bbox[3]]
                    new_bboxes.append(bbox)
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255, 0, 0), 2)
    
            bboxes_prev = new_bboxes  # Update previous bounding boxes
        
        cv2.imwrite(osp.join(result_path, filename), img)
