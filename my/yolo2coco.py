import os
import json
from pycocotools.coco import COCO

def yolo_to_coco(yolo_results_dir, output_file, img_size, is_gt=False):
    if is_gt:
        coco_format = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": "person"}]
        }
    
        H, W = img_size
        annotation_id = 0
        image_id = 0
        
        for txt_file in sorted(os.listdir(yolo_results_dir)):
            image_id += 1
            if txt_file.endswith('.txt'):
                # Extract the image ID from the filename
                with open(os.path.join(yolo_results_dir, txt_file), 'r') as f:
                    for line in f:
                        parts = list(map(float, line.strip().split()))
                        class_id, x_center, y_center, width, height = parts[:5]

                        x_center, y_center, width, height = x_center * W, y_center * H, width * W, height * H    

                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(class_id),
                            "bbox": [
                                x_center - width / 2,
                                y_center - height / 2,
                                width,
                                height
                            ],
                            "area": width * height,
                            "iscrowd": 0
                        }

                        coco_format["annotations"].append(annotation)
                        annotation_id += 1

                coco_format["images"].append({
                    "id": image_id,
                    "file_name": f"{image_id}.jpg"
                })

        with open(output_file, 'w') as f:
            json.dump(coco_format, f, indent=4)
    else:
        detections = []
        H, W = img_size
        annotation_id = 0
        image_id = 0

        for txt_file in sorted(os.listdir(yolo_results_dir)):
            image_id += 1
            if txt_file.endswith('.txt'):
                # Extract the image ID from the filename
                with open(os.path.join(yolo_results_dir, txt_file), 'r') as f:
                    for line in f:
                        parts = list(map(float, line.strip().split()))
                        class_id, x_center, y_center, width, height, confidence = parts[:6]

                        x_center, y_center, width, height = x_center * W, y_center * H, width * W, height * H    

                        detection = {
                            "image_id": image_id,
                            "category_id": int(class_id),
                            "bbox": [
                                x_center - width / 2,
                                y_center - height / 2,
                                width,
                                height
                            ],
                            "score": confidence
                        }

                        detections.append(detection)

        with open(output_file, 'w') as f:
            json.dump(detections, f, indent=4)

# 生成包含 GT 的 COCO 格式 JSON
yolo_to_coco(
    '/home/wiser-renjie/projects/yolov8/my/runs/my/wildtrack_cam7_yolov8x_1152_1920_0.3_TOP3000', 
    '/home/wiser-renjie/projects/yolov8/my/runs/my/jsons/gt_wildtrack_cam7_yolov8x_1152_1920_0.3_TOP3000.json',
    img_size=(1152, 1920),
    is_gt=True
)

# 生成包含检测结果的 JSON
yolo_to_coco(
    '/home/wiser-renjie/projects/yolov8/my/runs/my/wildtrack_cam7_my_1152_1920_0.3_i10_TOP3000', 
    '/home/wiser-renjie/projects/yolov8/my/runs/my/jsons/wildtrack_cam7_my_1152_1920_0.3_i10_TOP3000.json',
    img_size=(1152, 1920),
    is_gt=False
)