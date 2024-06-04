import numpy as np
import cv2
from ultralytics import YOLO

def plot_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox.astype(np.int32)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

# Load a pretrained YOLOv8n model
model = YOLO('yolov8x.pt')

img = cv2.imread('/home/wiser-renjie/remote_datasets/wildtrack/datasets_combined/train/images/C7_00001300.png')

results = model.predict(img, save_txt=False, save=False, classes=[0], imgsz=(1080, 1920), conf=0.5)

print(results[0].tojson())
# bboxes = results[0].boxes.xyxy.cpu().numpy()
# scores = results[0].boxes.conf.cpu().numpy()

# for bbox in bboxes:
#     img = plot_bbox(img, bbox)

# cv2.imwrite('test2.jpg', img)