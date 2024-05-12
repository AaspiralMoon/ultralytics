import cv2
import os
import re
import os.path as osp
import numpy as np
from ultralytics import YOLO

def get_idx(dir):
    pattern = re.compile(r'predict(\d+)')
    items = os.listdir(dir)
    max_idx = max(
        (int(match.group(1)) for item in items if (match := pattern.match(item))),
        default=None
    )
    return str(max_idx + 1) if max_idx is not None else ''

def letterbox(img, height=1080, width=1920,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dh, dw

def load_image(img_path, img_size=(2048, 1024)):
    img0 = cv2.imread(img_path)  # BGR
    assert img0 is not None, 'Failed to load ' + img_path

    img = cv2.resize(img0, img_size) if img_size is not None else img0
    return img

def split_image(img, pad=5):
    blocks = []
    # Loop over the image and extract blocks
    for y in range(0, 1024, 128):  # 8 blocks vertically
        for x in range(0, 2048, 128):  # 16 blocks horizontally
            # Extract the block
            block = img[y:y+128, x:x+128]

            # Add padding around the block
            padded_block = cv2.copyMakeBorder(block, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(127.5, 127.5, 127.5))
            
            # Append the padded block to the list
            blocks.append(padded_block)

    return blocks

def combine_images(file_path):
    images = []  # List to hold all the images

    # Load the images
    for i in range(128):
        img_path = os.path.join(file_path, f'image{i}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image {img_path} not found.")
        images.append(img)

    # Assume that all images have the same shape
    img_height, img_width = images[0].shape[:2]

    # Determine the layout of the images
    num_rows = 8  # Because sqrt(128) = 8x16
    num_cols = 16

    # Create an empty array to hold the combined image
    combined_image = np.zeros((img_height * num_rows, img_width * num_cols, 3), dtype=images[0].dtype)

    # Combine the images
    for idx, img in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols
        combined_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width] = img

    return combined_image

def plot_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox.astype(np.int32)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

if __name__ == '__main__':
    result_root = '/home/wiser-renjie/projects/yolov8/my/runs/detect/'
    save_path = osp.join(result_root, 'predict'+get_idx(result_root))
    img_path = '/home/wiser-renjie/remote_datasets/traffic/video1/00000001.jpg'
    
    img = load_image(img_path, (2048, 1024))
    blocks = split_image(img, pad=0)
    
    # Yolox = YOLO('yolov8x.pt')
    Yolox = YOLO('/home/wiser-renjie/projects/yolov8/my/runs/detect/train8/weights/best.pt')
    
    results = Yolox.predict(blocks, save=True, imgsz=blocks[0].shape[0], conf=0.5)
    
    img_combined = combine_images(save_path)
    cv2.imwrite('test_combined.jpg', img_combined)