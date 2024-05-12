import cv2
import numpy as np
import torch
from ultralytics import YOLO
from test_crop import plot_bbox

def pad_and_plot(img, bboxes, pad=0, color=(0, 255, 0), thickness=2):
    padded= cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    H, W = img.shape[:2]
    
    for bbox in bboxes:
        xcn, ycn, wn, hn = bbox
        
        xc, yc, w, h = xcn * W, ycn * H, wn * W, hn * H
        
        x1 = xc - w / 2 + pad
        y1 = yc - h / 2 + pad
        
        x2 = x1 + w
        y2 = y1 + h
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        cv2.rectangle(padded, (x1, y1), (x2, y2), color, thickness)
        
        if x2 - pad > W:
            print('x2 is outside the boundary: x2 - pad = {}, W = {}'.format(x2-pad, W))
        if y2 - pad > H:
            print('y2 is outside the boundary: y2 - pad = {}, H = {}'.format(y2-pad, H))
            
    return padded
    

def pad_image(img, pad=0):
    padded= cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(127.5, 127.5, 127.5))
    return padded

if __name__ == '__main__':
    model_path = '/home/wiser-renjie/projects/yolov8/my/runs/detect/train8/weights/best.pt'
    img_path = '/home/wiser-renjie/datasets/test_partial/train/images/00003881.jpg'
    label_path = img_path.replace('images', 'labels').replace('jpg', 'txt')
    
    model = YOLO(model_path)

    # img = cv2.imread(img_path)
    img = cv2.imread(img_path)
    labels = np.loadtxt(label_path)
    
    img_padded = pad_image(img, pad=0)
    
    results = model.predict(img_padded, save=True, imgsz=128, conf=0.5)

    bboxes = results[0].boxes.xyxy.cpu().numpy()
    print(bboxes)
    
    
    # print('Prediction: {}\nLabel: {}'.format(bboxes, labels))

    # out = pad_and_plot(img_padded, bboxes, pad=50)
            
    # cv2.imwrite('test2.jpg', out)