# Goal here is to generate a video similar to keypoint_detection.py
# But without infering keypoint !
# We assume we already have them

import os
import numpy as np
import cv2
from mmpose.apis import init_pose_model, vis_3d_pose_result
from mmpose.datasets import DatasetInfo
import matplotlib.pyplot as plt


# path to skeleton 2d npy folder
data2D_dir = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\data\\mmpose_ntu\\"
# path to skeleton 3d npy folder
data3D_dir = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\data\\mmpose_ntu_3d\\"
data_files = "S001C001P001R001A006.npy"
# path to video folder
video_path = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\NTU_samples_and_inference_code\\videos"
video_filename = "S001C001P001R001A006_rgb.avi"
# path to output folder
out_path = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\NTU_samples_and_inference_code\\out"
out_video_filename = "S001C001P001R001A006_rgb_skeleton.avi"


def main():
    mmpose_config_file_3D = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\mmpose\\configs\\body\\3d_kpt_sview_rgb_vid\\video_pose_lift\\h36m\\videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py"
    # mmpose_checkpoint_file_3D https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth
    mmpose_checkpoint_file_3D = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\humanaction\\mm_checkpoint\\videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth"
    device = "cuda:0"
    radius = 8 # Keypoint radius for visualization
    thickness = 2 # Link thickness for visualization
    show = False # whether to show visualizations.

    # read npy file
    # trace lines on video
    skeleton = np.load(os.path.join(data2D_dir, data_files))
    skeleton_3d = np.load(os.path.join(data3D_dir, data_files))
    
    # open each frame 
    # apply the prediction on it
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(os.path.join(video_path, video_filename))
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    # Video writer is None for now, we wait for the image size 
    writer = None

    # Only "TopDown"
    pose_lift_model = init_pose_model(mmpose_config_file_3D, mmpose_checkpoint_file_3D, device=device)
    # Only "PoseLifter" model is supported for the 2nd stage
    pose_lift_dataset = pose_lift_model.cfg.data['test']['type']
    # get dataset info
    pose_lift_dataset_info = pose_lift_model.cfg.data['test'].get('dataset_info', None)
    pose_lift_dataset_info = DatasetInfo(pose_lift_dataset_info)
    
    # Read until video is completed
    frame_count = 0
    while(cap.isOpened() & (frame_count < len(skeleton)) & (frame_count < len(skeleton_3d))):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # add skeleton to frame
            # pose_det_dataset : TopDownPoseTrack18VideoDataset
            # pose_lift_dataset : Body3DH36MDataset
            '''
            0: 'root (pelvis)',
            1: 'right_hip',
            2: 'right_knee',
            3: 'right_foot',
            4: 'left_hip',
            5: 'left_knee',
            6: 'left_foot',
            7: 'spine',
            8: 'thorax',
            9: 'neck_base',
            10: 'head',
            11: 'left_shoulder',
            12: 'left_elbow',
            13: 'left_wrist',
            14: 'right_shoulder',
            15: 'right_elbow',
            16: 'right_wrist'
            '''

            # kpt_score only to fill the skeleton data
            kpt_score = np.ones((skeleton.shape[0], skeleton.shape[1], 1))
            skeleton = np.concatenate([skeleton, kpt_score], axis = 2)
            pose_lift_results_vis = [{'keypoints':skeleton[frame_count, ...], 'keypoints_3d':skeleton_3d[frame_count, ...]}]

            num_instances = len(pose_lift_results_vis)

            img_vis = vis_3d_pose_result(
                pose_lift_model,
                result=pose_lift_results_vis,
                img=frame,
                dataset=pose_lift_dataset,
                dataset_info=pose_lift_dataset_info,
                out_file=None,
                radius=radius,
                thickness=thickness,
                num_instances=num_instances,
                show=show)

            # Define the codec and create VideoWriter object.
            # Define the fps to be equal to 24. Also frame size is passed.
            if writer is None:
                writer = cv2.VideoWriter(os.path.join(out_path, out_video_filename), cv2.VideoWriter_fourcc(*'mp4v'), 24, (img_vis.shape[1], img_vis.shape[0]))

            # Write the frame into the file
            writer.write(img_vis)
            frame_count += 1
        else:
            break

    # When everything done, release the video capture object and writer object
    cap.release()
    writer.release()

if __name__ == '__main__':
    main()