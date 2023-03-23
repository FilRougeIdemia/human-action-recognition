# Video annotated with Via should be cut into chunks
# Each of these chunks is annotated with a single label
# A json accompanies the chunk

# expects the video directory to contain a subdirectory for each person behind a camera
# each subdirectory has the videos
# annotation is a csv file that should contain the video name, the name of the json file, etc.
import pandas as pd
import json
import cv2
from os import path as osp
import os

def cut_jsons_with_annotations(video_dir, annotation_file, keypoint_dir, json_output_dir):
    for video_file in os.listdir(video_dir):
        cut_both_video_json(video_file, video_dir, annotation_file, keypoint_dir, None, json_output_dir, json_only=True)

def cut_both_video_json(video_file, video_dir, annotation_file, keypoint_dir, video_output_dir, json_output_dir, json_only=False):
    anno = pd.read_csv(annotation_file)
    anno_video = anno[anno["Source Video Name"] == video_file]

    # go over each row 
    # to find the label, the keypoint file associated
    # the frames involved
    for _, row in anno_video.iterrows():
        label = row['Label']
        action = row['Action']
        mode = row['Mode']
        # precise the directories to look for the data because of the messy "mode" thing we did
        if mode == 1:
            mode_kpt_dir = "mmpose-skeleton-predictions"
            mode_video_dir = None
            video_to_cut_file = video_file
        else :
            mode_kpt_dir = "mode2"
            mode_video_dir = "../track_selections"
            video_to_cut_file = row["Video Name"].replace(".h264", "") # You may need to match the h264, I don't have it in my data
        keypoint_file = row['File Name']
        start_frame = row['temporal_segment_start']  # Start frame number (inclusive)
        end_frame = row['temporal_segment_end']    # End frame number (inclusive)

        # prepare the base of outputs name
        output_name = f"{osp.splitext(video_to_cut_file)[0]}_segment_{start_frame}_{end_frame}_action_{action}_label_{label}"
        # declare the json output, video output only if json_only False
        output_json = osp.join(json_output_dir, output_name + '.json')

        # Open the json
        with open(osp.join(keypoint_dir, mode_kpt_dir, keypoint_file)) as f:    
            skeleton_file = json.load(f)
            json_frames = []

        if json_only:
            # only cut the json not the video
            json_frames = skeleton_file['frames'][start_frame:end_frame]
        else :
            output_video = osp.join(video_output_dir, output_name + '.mp4')

            # Go get the right input video to cut        
            # Open the input video
            input_video =  osp.join(video_dir, video_to_cut_file) if mode_video_dir is None else osp.join(video_dir, mode_video_dir, video_to_cut_file)
            cap = cv2.VideoCapture(input_video)

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create the VideoWriter for the output video
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

            # Set the start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Extract a smaller video corresponding only to the chunk
            frame_number = start_frame
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break

                # If the current frame is in the selected range, write it to the output video
                if start_frame <= frame_number <= end_frame:
                    out.write(frame)
                    # also store the corresponding json frame
                    json_frames.append(skeleton_file['frames'][frame_number])

                frame_number += 1

                # Stop processing when the end frame is reached
                if frame_number > end_frame:
                    break

            # Release the input and output video objects
            cap.release()
            out.release()

        # write the json for the cut video
        with open(output_json, 'w') as output_file:
            sub_json = {
                "sequence_name": skeleton_file['sequence_name'],
                "num_frames": len(json_frames),
                "frames": json_frames
            }
            json.dump(sub_json, output_file, indent=2)

if __name__ == '__main__':
    annotation_file = '/home/users/jma-21/BGDIA706 - Fil Rouge/human-action-recognition/data/input/annotation/Annotations.csv'
    video_file = '20230112_095742.mp4'
    video_dir = '/home/users/jma-21/BGDIA706 - Fil Rouge/acquisition_sacs/'
    keypoint_dir = '/home/users/jma-21/BGDIA706 - Fil Rouge/kpts'
    video_output_dir = "/home/users/jma-21/BGDIA706 - Fil Rouge/acquisition_sacs_decoupees/videos"
    json_output_dir = "/home/users/jma-21/BGDIA706 - Fil Rouge/acquisition_sacs_decoupees/jsons"
    json_only = False
    cut_both_video_json(video_file, video_dir+'vd', annotation_file, keypoint_dir, video_output_dir, json_output_dir, json_only)
    #for specific_dir in ['jm', 'ph', 'pp', 'sg', 'vd', 'wd']:
    #    cut_jsons_with_annotations(video_dir+specific_dir, annotation_file, keypoint_dir, json_output_dir)