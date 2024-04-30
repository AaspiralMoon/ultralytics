import os
import shutil

def move_files(start_idx, end_idx, source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for i in range(start_idx, end_idx + 1):
        filename = f"{i:08}.txt"
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)

        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)
            print(f"Copying: {filename}")
        else:
            print(f"File not found: {filename}")


source_dir = '/home/wiser-renjie/projects/yolov8/my/runs/detect/predict3/labels'
target_dir = '/home/wiser-renjie/datasets/temp/val/labels'
move_files(1001, 1100, source_dir, target_dir)
