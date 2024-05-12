from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')

# Train the model with 2 GPUs
results = model.train(data='/home/wiser-renjie/datasets/test_partial/test_partial.yaml', mosaic=0, epochs=200, batch=128, imgsz=128)