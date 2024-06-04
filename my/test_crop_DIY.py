import cv2
import os
import re
import os.path as osp
import numpy as np
from ultralytics import YOLO
from test_crop import get_idx, load_image

def split_image(img, block_size, block_dim):
    num_height, num_width = block_dim
    img_height, img_width = img.shape[:2]
    
    # Ensure the total required dimensions don't exceed the image dimensions
    if img_width > num_width * block_size or img_height > num_height * block_size:
        raise ValueError("Image dimensions are too small for the specified number of blocks and block size.")
    
    # Calculate the required overlap if the total dimension of the blocks is greater than the image dimension
    overlap_width = (num_width * block_size - img_width) // (num_width - 1) if (num_width * block_size > img_width and num_width > 1) else 0
    overlap_height = (num_height * block_size - img_height) // (num_height - 1) if (num_height * block_size > img_height and num_height > 1) else 0

    blocks = []
    coordinates = []
    for i in range(num_height):
        for j in range(num_width):
            # Calculate left and top coordinates considering overlap
            left = j * (block_size - overlap_width)
            top = i * (block_size - overlap_height)
            right = left + block_size
            bottom = top + block_size

            # Ensure the block does not exceed the image boundaries
            right = min(right, img_width)
            bottom = min(bottom, img_height)

            # Crop the image to create a block
            crop_img = img[top:bottom, left:right]
            blocks.append(crop_img)
            coordinates.append((left, top, right, bottom))
    
    return blocks, coordinates

def map_results(img, results, coordinates):
    # Define a list of colors as (B, G, R) tuples
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    color_index = 0

    for bboxes, coord in zip(results, coordinates):
        x1, y1, x2, y2 = coord
        for box in bboxes:
            # Map box coordinates back to original image
            box_x1, box_y1, box_x2, box_y2 = box.astype(np.int32)
            start_point = (box_x1 + x1, box_y1 + y1)
            end_point = (box_x2 + x1, box_y2 + y1)
            # Get the current color and draw the rectangle
            current_color = colors[color_index % len(colors)]
            cv2.rectangle(img, start_point, end_point, current_color, 2)
            # Increment the color index for the next box
            color_index += 1

    return img

def plot_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox.astype(np.int32)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

if __name__ == '__main__':
    result_root = '/home/wiser-renjie/projects/yolov8/my/runs/detect/'
    save_path = osp.join(result_root, 'predict'+get_idx(result_root))
    img_path = '/home/wiser-renjie/remote_datasets/traffic/video1/00000001.jpg'
    
    img = load_image(img_path, (2048, 1024))
    blocks, coordinates = split_image(img, block_size=256, block_dim=(8, 16))
    
    # Yolox = YOLO('yolov8x.pt')
    Yolox = YOLO('/home/wiser-renjie/projects/yolov8/my/runs/detect/train8/weights/best.pt')
    
    preds = Yolox.predict(blocks, save=False, classes=[0], imgsz=blocks[0].shape[:2], conf=0.5)

    results = []
    for pred in preds:
        bboxes = pred.boxes.xyxy.cpu().numpy()
        results.append(bboxes)
        
    img_combined = map_results(img, results, coordinates)
    cv2.imwrite('test_combined2.jpg', img_combined)