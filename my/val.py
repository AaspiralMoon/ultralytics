from ultralytics import YOLO

if __name__ == '__main__':
    model_path = '/home/wiser-renjie/projects/yolov8/my/runs/detect/train8/weights/best.pt'
    model = YOLO(model_path)  # load a custom model

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category