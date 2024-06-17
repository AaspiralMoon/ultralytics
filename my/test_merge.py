import cv2
import numpy as np
import time
from ultralytics import YOLO
from utils import bbox_to_blocks, compute_union
from test_cluster import dbscan_clustering

# def find_position(skyline, width, height):
#     best_x, best_y = None, None
#     min_height = float('inf')
#     for i in range(len(skyline) - width + 1):
#         max_height = max(skyline[i:i+width])
#         if max_height + height < min_height:
#             best_x = i
#             best_y = max_height
#             min_height = max_height + height
#     return best_x, best_y, min_height

# def update_skyline(skyline, x, width, height):
#     for i in range(width):
#         skyline[x + i] = height

# def rectangle_packing(bboxes):
#     total_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes)
#     side_length = int(np.ceil(np.sqrt(total_area)))
#     container_width = side_length
#     skyline = [0] * container_width

#     rectangles = [{'width': x2 - x1, 'height': y2 - y1, 'original_bbox': (x1, y1, x2, y2)} for x1, y1, x2, y2 in bboxes]
#     rectangles_sorted = sorted(rectangles, key=lambda r: -r['height'])

#     container_height = 0
#     for rect in rectangles_sorted:
#         x, y, new_height = find_position(skyline, rect['width'], rect['height'])
#         if x is not None:
#             rect['x'] = x
#             rect['y'] = y
#             update_skyline(skyline, x, rect['width'], new_height)
#             container_height = max(container_height, new_height)
#         else:
#             # If no position is found, expand container width and retry
#             container_width += rect['width']
#             skyline.extend([0] * rect['width'])
#             x, y, new_height = find_position(skyline, rect['width'], rect['height'])
#             rect['x'] = x
#             rect['y'] = y
#             update_skyline(skyline, x, rect['width'], new_height)
#             container_height = max(container_height, new_height)

#     container = np.zeros((container_height, container_width, 3), dtype=np.uint8)

#     return container, rectangles_sorted

def rectangle_packing(bboxes):
    packed_rect = []
    total_width = sum(x2 - x1 for x1, y1, x2, y2 in bboxes)
    max_height = max(y2 - y1 for x1, y1, x2, y2 in bboxes)
    
    current_x = 0

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        packed_rect.append({
            'original_bbox': (x1, y1, x2, y2),
            'x': current_x,
            'y': 0,
            'width': width,
            'height': height
        })

        current_x += width

    packed_img = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    return packed_img, packed_rect

def get_merge_info(bboxes):
    packed_img, packed_rect = rectangle_packing(bboxes)

    return packed_img, packed_rect

def get_merge_img(img, packed_img, packed_rect):
    merged_img = np.zeros_like(packed_img)

    for rect in packed_rect:
        x1, y1, x2, y2 = rect['original_bbox']
        sub_img = img[y1:y2, x1:x2]
        merged_img[rect['y']:rect['y']+rect['height'], rect['x']:rect['x']+rect['width']] = sub_img

    return merged_img
    
def save_image(container, output_path):
    cv2.imwrite(output_path, container)

if __name__ == '__main__':
    # 示例图像
    img = cv2.imread("/home/wiser-renjie/remote_datasets/wildtrack/decoded_images/cam7/00000190.jpg")
    H, W = 1152, 1920
    img = cv2.resize(img, (W, H))
    
    model = YOLO('yolov8x.pt')
    results = model.predict(img, save_txt=False, save=False, classes=[0], imgsz=(H, W), conf=0.2)
    
    bboxes = results[0].boxes.xyxy.cpu().numpy()

    t1 = time.time()
    cluster_bboxes, cluster_dic, cluster_num = dbscan_clustering(bboxes)
    t2 = time.time()
    print(f'cluster time: {(t2-t1)*1000} ms')
    
    if cluster_dic:
        hard_regions = [compute_union(cluster_bboxes[x], (H, W)) for x in cluster_bboxes]
        print(hard_regions)
           
        hard_blocks = np.array([bbox_to_blocks(y, 128) for y in hard_regions], dtype=np.int32)
        print(hard_blocks)
        
        packed_img, packed_rect = get_merge_info(hard_blocks)
        
        print(packed_rect)
        merged_img = get_merge_img(img, packed_img, packed_rect)
        
    output_path = "merged_image.jpg"
    save_image(merged_img, output_path)
