import numpy as np
import cv2

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

    # 将最终的容器宽度和高度调整为128的倍数
    container_width = ((container_width + 127) // 128) * 128
    container_height = ((container_height + 127) // 128) * 128

    container = np.zeros((container_height, container_width, 3), dtype=np.uint8)

    # 放置矩形
    for rect in rectangles_sorted:
        x1, y1 = rect['x'], rect['y']
        x2, y2 = x1 + rect['width'], y1 + rect['height']
        container[y1:y2, x1:x2] = (255, 255, 255)  # 用白色填充

    return container, rectangles_sorted

def get_merge_info(hard_blocks):
    packed_img, packed_rect = rectangle_packing(hard_blocks)
    return packed_img, packed_rect

# 示例硬区域
hard_blocks = np.array([
    [0, 0, 256, 256],
    [0, 0, 128, 256],
    [0, 0, 256, 128],
    [0, 0, 128, 128]
])

packed_img, packed_rect = get_merge_info(hard_blocks)

# 保存结果图像
cv2.imwrite("packed_image.png", packed_img)

print("Packed Rectangles:")
for rect in packed_rect:
    print(rect)
