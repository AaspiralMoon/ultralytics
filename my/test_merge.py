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

def rectangle_packing(rectangles):
    total_area = sum(rect['width'] * rect['height'] for rect in rectangles)
    side_length = int(np.ceil(np.sqrt(total_area)))
    container_width = side_length
    skyline = [0] * container_width

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
    for rect in rectangles_sorted:
        container[rect['y']:rect['y']+rect['height'], rect['x']:rect['x']+rect['width']] = (255, 255, 255)
        cv2.rectangle(container, (rect['x'], rect['y']), (rect['x'] + rect['width'], rect['y'] + rect['height']), (0, 0, 0), 1)

    return container

def save_image(container, output_path):
    cv2.imwrite(output_path, container)

def crop_to_fit(container, rectangles):
    max_x = max(rect['x'] + rect['width'] for rect in rectangles)
    max_y = max(rect['y'] + rect['height'] for rect in rectangles)
    return container[:max_y, :max_x]

# 定义矩形框，包含宽度和高度
rectangles = [
    {'width': 160, 'height': 20},
    {'width': 140, 'height': 240},
]

import time
t1 = time.time()
container = rectangle_packing(rectangles)
container = crop_to_fit(container, rectangles)
t2 = time.time()
print((t2-t1)*1000)
output_path = "packed_image_fast.png"
save_image(container, output_path)
