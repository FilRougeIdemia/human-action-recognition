# Pipeline

# What should the pipeline be

# After collecting the videos
# Extract keypoints
# Generate videos with keypoints (while extraction or after)
# Model (Declare, train)
# Generate or modify videos by adding predictions

import keypoint_detection
import keypoint_json_to_npy
import predict_new_data

def detect_actions(original_video_dir, keypoint_video_dir, keypoint_json_dir, keypoint_npy_dir, prediction_dir, prediction_video_dir, file_names, is_sliding_window):
    keypoint_detection.detect_keypoints(original_video_dir, keypoint_video_dir, keypoint_json_dir, file_names)
    keypoint_json_to_npy.convert_json_to_npy(keypoint_json_dir, keypoint_npy_dir, file_names)
    predict_new_data.predict_on_videos(keypoint_npy_dir, keypoint_video_dir, prediction_dir, prediction_video_dir, is_sliding_window)
    return None


if __name__ == "__main__":
    original_video_dir = 'data/input/video'
    keypoint_npy_dir = 'data/output/keypoint_npy'
    keypoint_video_dir = 'data/output/keypoint_video'
    keypoint_json_dir = 'data/output/keypoint_json'
    prediction_dir = 'data/output/prediction' # contains both sliding window and whole
    prediction_video_dir = 'data/output/prediction_keypoint_video'
    file_names = None
    is_sliding_window = True
    detect_actions(original_video_dir, keypoint_video_dir, keypoint_json_dir, keypoint_npy_dir, prediction_dir, prediction_video_dir, file_names, is_sliding_window)