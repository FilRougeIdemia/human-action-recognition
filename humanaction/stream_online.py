import cv2
import os
import numpy as np
import torch
from keypoint_detection import detect_stream_keypoints
from predict_new_data import predict_on_stream

# Does it make sense to do everything on the video as it happens ?
# There isn't anyone looking at the video at all time
# So no, I don't think so.
# There should be an alert when a window with a specific action was detected.
# We must wait for a window to finish to start detecting keypoints on it, etc.
# Operations happening on windows must be performed in parallel.
# There is no need to wait for one window to finish in order to go the second one.
# At the end of a window process if there is an interesting action
# The window should be saved in a folder. 
# Information about this window should be displayed on the current video feed. 

# In the output stream folder, we must have a folder for each stream input
# in a stream output folder there are the window videos containing actions
# those video files are named after the stream 
# + indication about start and end of the window 
# + action performed

# Hopefully with a long enough stride and short enough window the process won't
# accumulate to much work to do simultanously

# improvement ? not detecting keypoint twice when windows overlap (but at the same time less parallel)


# constants 
# mainly paths 
original_video_dir = 'data/input/video'
keypoint_npy_dir = 'data/output/keypoint_npy'
keypoint_video_dir = 'data/output/keypoint_video'
keypoint_json_dir = 'data/output/keypoint_json'
prediction_dir = 'data/output/prediction' # contains both sliding window and whole
prediction_video_dir = 'data/output/prediction_keypoint_video'
file_name = 'video_JM_10s_HD.mp4'

base_file_name = file_name[:-4]
stream_dir = os.path.join('data/output/stream/online', base_file_name)

is_sliding_window = True
window_size = 45
stride = 30
threshold = 0.95

# Initialize the VideoCapture object
cap = cv2.VideoCapture(os.path.join(original_video_dir, file_name))
# length of the video
nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Check if the video source was opened successfully
if not cap.isOpened():
    print("Error opening video source")

# Read the first frame from the video source
ret, frame = cap.read()

# Store window frames
window = []

# remaining frames for next window prediction
remaining_frames = window_size

# total frame count
frame_count = 0

# Prepare to save the output video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

# Loop over the frames of the video source
while ret:
    # Add frame to the current window
    window.append(frame)
    # Remove frame to count
    remaining_frames -= 1
    # Add to total frame count
    frame_count += 1

    # Execute the following in parallel
    if remaining_frames == 0 or frame_count == nb_frames:
        # Infer keypoints on the frame
        pose_det_results_list, video_results_vis, video_results_export, pose_lift_model, pose_lift_dataset, pose_lift_dataset_info = detect_stream_keypoints(window, (frame.shape[:2]))

        # Get npy skeletons
        # TODO do better than going through list by outputing relevant info from detect_stream_keypoints 
        skeletons = np.array([frame_result['detections'][0]['keypoints'] for frame_result in video_results_export])[:,:,:2]

        # Once enough data is collected to make a window frame 
        # predict on window frame
        # Also we must wait for a stride to pass in order to make new predictions
        probs = predict_on_stream(skeletons, is_sliding_window=False)

        # Make a decision
        # subsequently save window in which the action happened
        action_prob = probs.max()
        i,j = np.unravel_index(probs.argmax(), probs.shape)
        if action_prob > threshold : # TODO change that with a real strategy
            action = j
            # save the action video file
            if not os.path.exists(stream_dir):
                os.makedirs(stream_dir)
            window_file_name = f'{base_file_name}_{action}_{frame_count}.mp4'
            writer = cv2.VideoWriter(os.path.join(stream_dir, window_file_name), fourcc, fps, (frame.shape[1], frame.shape[0]))
            # write all the frames now, which is probably not efficient
            for frame in window:
                writer.write(frame)
            writer.release()

        # Keep only the frames in the next window
        window = window[-(window_size-stride):]
        # Reset the number of frames to accumulate
        remaining_frames = stride


    # Read the next frame
    ret, frame = cap.read()

# Release the VideoCapture object
cap.release()
