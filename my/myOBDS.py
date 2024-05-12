import cv2
import numpy as np
from ultralytics import YOLO as yolo
from OBDS import OBDS_single

def main(video_path, processing_interval, processing_duration):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    # Load a pretrained YOLOv8n model
    model = yolo('yolov8x.pt')

    frame_number = 0
    processing_time_frames = int(fps * processing_duration)

    try:
        while cap.isOpened() and frame_number < processing_time_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            bboxes = []
            if frame_number % processing_interval == 0:
                results = model.predict(frame, save=False, classes=[2], conf=0.5)
                refs = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
                bboxes_prev = refs
                
            else:
                for bbox_prev, ref in zip(bboxes_prev, refs):
                    print(ref)
                    target = frame[ref[1]:ref[3], ref[0]:ref[2]]
                    print(target)
                    box = OBDS_single(frame, target, bbox_prev)
                    bboxes.append(box)            
                bboxes_prev = bboxes
                
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
    processing_duration = 5  # Process the first 10 seconds of the video
    
    main(video_path, processing_interval, processing_duration)
