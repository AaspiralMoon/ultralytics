import multiprocessing
import time
import cv2
from ultralytics import YOLO

def process_images_in_process(image, model):
    _ = model.predict(image, classes=[0], save=False)

def process_images_sequentially(images, model):
    for image in images:
        _ = model.predict(image, classes=[0], save=False)

# Load the models
model1 = YOLO("yolov8n.pt")
model2 = YOLO("yolov8n.pt")

# Define the image files for the processors
image_file1 = "/home/wiser-renjie/remote_datasets/wildtrack/datasets_combined/train/images/C1_00000000.png"
image_file2 = "/home/wiser-renjie/remote_datasets/wildtrack/datasets_combined/train/images/C1_00000005.png"

# Read the images before processing
img1 = cv2.imread(image_file1)
img2 = cv2.imread(image_file2)

# # Measure end-to-end time for multiprocessing
# start_time = time.time()

# # Create the processor processes
# processor_process1 = multiprocessing.Process(target=process_images_in_process, args=(img1, model1))
# processor_process2 = multiprocessing.Process(target=process_images_in_process, args=(img2, model2))

# # Start the processor processes
# processor_process1.start()
# processor_process2.start()

# # Wait for the processor processes to finish
# processor_process1.join()
# processor_process2.join()

# end_time = time.time()
# total_multiprocessing_time = end_time - start_time
# print(f"Total multiprocessing end-to-end time: {total_multiprocessing_time:.5f} seconds")

# Measure end-to-end time for sequential processing
start_time = time.time()
process_images_sequentially([img1, img2], model1)
end_time = time.time()
total_sequential_time = end_time - start_time
print(f"Total sequential end-to-end time: {total_sequential_time:.5f} seconds")
