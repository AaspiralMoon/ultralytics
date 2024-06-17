import numpy as np
import cv2
from ultralytics import YOLO

def plot_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox.astype(np.int32)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

# Load a pretrained YOLOv8n model
model = YOLO('yolov8x.pt')

# img = cv2.imread('aachen_000000_000019_leftImg8bit.png')
# img = cv2.resize(img, (640, 640))

results = model.predict('/home/wiser-renjie/remote_datasets/wildtrack/decoded_images/cam7', save_txt=True, save_conf=True, save=False, classes=[0], imgsz=(1152, 1920), conf=0.3)

# print(results[0].tojson())
# bboxes = results[0].boxes.xyxy.cpu().numpy()
# scores = results[0].boxes.conf.cpu().numpy()

# for bbox in bboxes:
#     img = plot_bbox(img, bbox)

# cv2.imwrite('test2.jpg', img)
