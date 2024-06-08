import cv2
import numpy as np

def find_position(skyline, width, height):
    best_x, best_y = None, None
    min_height = float('inf')
    for i in range(len(skyline) - width + 1):
        max_height = max(skyline[i:i+width])
        if max_height + height < min_height:
            best_x = i
            best_y = max_height
            min_height = max_height + height
    return best_x, best_y, min_height

def update_skyline(skyline, x, width, height):
    for i in range(width):
        skyline[x + i] = height

def rectangle_packing(bboxes):
    total_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes)
    side_length = int(np.ceil(np.sqrt(total_area)))
    container_width = side_length
    skyline = [0] * container_width

    rectangles = [{'width': x2 - x1, 'height': y2 - y1, 'original_bbox': (x1, y1, x2, y2)} for x1, y1, x2, y2 in bboxes]
    rectangles_sorted = sorted(rectangles, key=lambda r: -r['height'])

    container_height = 0
    for rect in rectangles_sorted:
        x, y, new_height = find_position(skyline, rect['width'], rect['height'])
        if x is not None:
            rect['x'] = x
            rect['y'] = y
            update_skyline(skyline, x, rect['width'], new_height)
            container_height = max(container_height, new_height)
        else:
            # If no position is found, expand container width and retry
            container_width += rect['width']
            skyline.extend([0] * rect['width'])
            x, y, new_height = find_position(skyline, rect['width'], rect['height'])
            rect['x'] = x
            rect['y'] = y
            update_skyline(skyline, x, rect['width'], new_height)
            container_height = max(container_height, new_height)

    container = np.zeros((container_height, container_width, 3), dtype=np.uint8)
    # for rect in rectangles_sorted:
    #     container[rect['y']:rect['y']+rect['height'], rect['x']:rect['x']+rect['width']] = (255, 255, 255)
    #     cv2.rectangle(container, (rect['x'], rect['y']), (rect['x'] + rect['width'], rect['y'] + rect['height']), (0, 0, 0), 1)

    return container, rectangles_sorted

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
    img = cv2.imread("/home/wiser-renjie/remote_datasets/wildtrack/datasets_combined/train/images/C7_00001300.png")

    # 示例目标框
    bboxes = np.array([
        [5, 5, 250, 25],
        [5, 5, 25, 250]
    ])

    import time
    t1 = time.time()
    merged_img = get_merged_img(img, bboxes)
    t2 = time.time()
    print((t2-t1)*1000)
    output_path = "merged_image.png"
    save_image(merged_img, output_path)
