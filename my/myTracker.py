import cv2
import numpy as np
from ultralytics import YOLO as yolo


def main(video_path, processing_interval, processing_duration):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    # Load a pretrained YOLOv8n model
    model = yolo('yolov8x.pt')

    # Initialize MultiTracker
    trackers = cv2.legacy.MultiTracker_create()

    frame_number = 0
    processing_time_frames = int(fps * processing_duration)

    try:
        while cap.isOpened() and frame_number < processing_time_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_number % processing_interval == 0:
                trackers = cv2.legacy.MultiTracker_create()
                results = model(frame)
                bboxes = results[0].boxes.xyxy.cpu().numpy()
                
                for bbox in bboxes:
                    tracker = cv2.legacy.TrackerMedianFlow_create()
                    trackers.add(tracker, frame, tuple(bbox))

            # Update trackers and get new bounding boxes
            success, bboxes = trackers.update(frame)
            
            # Draw bounding boxes
            for box in bboxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Write the frame to the output video
            out.write(frame)
            frame_number += 1

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = '/home/wiser-renjie/remote_datasets/traffic/video1.mp4'
    processing_interval = 10  # Process every 30 frames
    processing_duration = 120  # Process the first 10 seconds of the video
    
    # trackers = {
	# "csrt": cv2.legacy.TrackerCSRT_create,
	# "kcf": cv2.legacy.TrackerKCF_create,
	# "boosting": cv2.legacy.TrackerBoosting_create,
	# "mil": cv2.legacy.TrackerMIL_create,
	# "tld": cv2.legacy.TrackerTLD_create,
	# "medianflow": cv2.legacy.TrackerMedianFlow_create,
	# "mosse": cv2.legacy.TrackerMOSSE_create
    # }
    
    main(video_path, processing_interval, processing_duration)
