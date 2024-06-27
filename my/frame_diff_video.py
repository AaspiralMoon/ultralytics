import cv2
import numpy as np

def process_video(input_video_path, output_video_path, process_duration=None):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to create output video

    # Prepare output video writer
    out = cv2.VideoWriter(output_video_path, codec, fps, (width, height))
    
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read the video")
        cap.release()
        out.release()
        return

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0
    total_frames = int(min(process_duration * fps, cap.get(cv2.CAP_PROP_FRAME_COUNT))) if process_duration else float('inf')

    # Process the video
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate the frame difference
        frame_diff = cv2.absdiff(current_frame_gray, prev_frame_gray)

        # Update the previous frame
        prev_frame_gray = current_frame_gray

        # Convert the single channel image back to BGR
        frame_diff_bgr = cv2.cvtColor(frame_diff, cv2.COLOR_GRAY2BGR)

        # Write the frame into the file 'output_video.mp4'
        out.write(frame_diff_bgr)

        frame_count += 1

    # Release everything when done
    cap.release()
    out.release()
    print("Processing complete. Output saved to", output_video_path)


# Example usage
process_video('/home/wiser-renjie/remote_datasets/wildtrack/raw/cam1.mp4', 'cam1.mp4', process_duration=20)  # Process first 10 seconds
