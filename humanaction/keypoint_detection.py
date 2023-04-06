import mmcv
from mmdet.apis import init_detector, inference_detector
from mmpose.datasets import DatasetInfo
from mmpose.apis import (init_pose_model, process_mmdet_results, collect_multi_frames, 
                         inference_top_down_pose_model, get_track_id, extract_pose_sequence,
                         inference_pose_lifter_model, vis_3d_pose_result)
import copy
import cv2
import numpy as np
import os.path as osp
import json
import os

def convert_keypoint_definition(keypoints, pose_det_dataset, pose_lift_dataset):
    """Convert pose det dataset keypoints definition to pose lifter dataset
    keypoints definition, so that they are compatible with the definitions
    required for 3D pose lifting.

    Args:
        keypoints (ndarray[K, 2 or 3]): 2D keypoints to be transformed.
        pose_det_dataset, (str): Name of the dataset for 2D pose detector.
        pose_lift_dataset (str): Name of the dataset for pose lifter model.

    Returns:
        ndarray[K, 2 or 3]: the transformed 2D keypoints.
    """
    assert pose_lift_dataset in [
        'Body3DH36MDataset', 'Body3DMpiInf3dhpDataset'
        ], '`pose_lift_dataset` should be `Body3DH36MDataset` ' \
        f'or `Body3DMpiInf3dhpDataset`, but got {pose_lift_dataset}.'

    coco_style_datasets = [
        'TopDownCocoDataset', 'TopDownPoseTrack18Dataset',
        'TopDownPoseTrack18VideoDataset'
    ]
    keypoints_new = np.zeros((17, keypoints.shape[1]), dtype=keypoints.dtype)
    if pose_lift_dataset == 'Body3DH36MDataset':
        if pose_det_dataset in ['TopDownH36MDataset']:
            keypoints_new = keypoints
        elif pose_det_dataset in coco_style_datasets:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
            # in COCO, head is in the middle of l_eye and r_eye
            # in PoseTrack18, head is in the middle of head_bottom and head_top
            keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
            # rearrange other keypoints
            keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
                keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
        elif pose_det_dataset in ['TopDownAicDataset']:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[0] = (keypoints[9] + keypoints[6]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[8] = (keypoints[3] + keypoints[0]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
            # neck base (top end of neck) is 1/4 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[9] = (3 * keypoints[13] + keypoints[12]) / 4
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[10] = (5 * keypoints[13] + 7 * keypoints[12]) / 12

            keypoints_new[[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
                keypoints[[6, 7, 8, 9, 10, 11, 3, 4, 5, 0, 1, 2]]
        elif pose_det_dataset in ['TopDownCrowdPoseDataset']:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[0] = (keypoints[6] + keypoints[7]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[8] = (keypoints[0] + keypoints[1]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
            # neck base (top end of neck) is 1/4 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[9] = (3 * keypoints[13] + keypoints[12]) / 4
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[10] = (5 * keypoints[13] + 7 * keypoints[12]) / 12

            keypoints_new[[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
                keypoints[[7, 9, 11, 6, 8, 10, 0, 2, 4, 1, 3, 5]]
        else:
            raise NotImplementedError(
                f'unsupported conversion between {pose_lift_dataset} and '
                f'{pose_det_dataset}')

    elif pose_lift_dataset == 'Body3DMpiInf3dhpDataset':
        if pose_det_dataset in coco_style_datasets:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[14] = (keypoints[11] + keypoints[12]) / 2
            # neck (bottom end of neck) is in the middle of
            # l_shoulder and r_shoulder
            keypoints_new[1] = (keypoints[5] + keypoints[6]) / 2
            # spine (centre of torso) is in the middle of neck and root
            keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2

            # in COCO, head is in the middle of l_eye and r_eye
            # in PoseTrack18, head is in the middle of head_bottom and head_top
            keypoints_new[16] = (keypoints[1] + keypoints[2]) / 2

            if 'PoseTrack18' in pose_det_dataset:
                keypoints_new[0] = keypoints[1]
                # don't extrapolate the head top confidence score
                keypoints_new[16, 2] = keypoints_new[0, 2]
            else:
                # head top is extrapolated from neck and head
                keypoints_new[0] = (4 * keypoints_new[16] -
                                    keypoints_new[1]) / 3
                # don't extrapolate the head top confidence score
                keypoints_new[0, 2] = keypoints_new[16, 2]
            # arms and legs
            keypoints_new[2:14] = keypoints[[
                6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15
            ]]
        elif pose_det_dataset in ['TopDownAicDataset']:
            # head top is head top
            keypoints_new[0] = keypoints[12]
            # neck (bottom end of neck) is neck
            keypoints_new[1] = keypoints[13]
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[14] = (keypoints[9] + keypoints[6]) / 2
            # spine (centre of torso) is in the middle of neck and root
            keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[16] = (5 * keypoints[13] + 7 * keypoints[12]) / 12
            # arms and legs
            keypoints_new[2:14] = keypoints[0:12]
        elif pose_det_dataset in ['TopDownCrowdPoseDataset']:
            # head top is top_head
            keypoints_new[0] = keypoints[12]
            # neck (bottom end of neck) is in the middle of
            # l_shoulder and r_shoulder
            keypoints_new[1] = (keypoints[0] + keypoints[1]) / 2
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[14] = (keypoints[7] + keypoints[6]) / 2
            # spine (centre of torso) is in the middle of neck and root
            keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[16] = (5 * keypoints[13] + 7 * keypoints[12]) / 12
            # arms and legs
            keypoints_new[2:14] = keypoints[[
                1, 3, 5, 0, 2, 4, 7, 9, 11, 6, 8, 10
            ]]

        else:
            raise NotImplementedError(
                f'unsupported conversion between {pose_lift_dataset} and '
                f'{pose_det_dataset}')

    return keypoints_new


def detect_stream_keypoints(video, # an iterable containing frames of the video stream
                            resolution, # resolution of the video
                           mmdet_config_file = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\mmpose\\demo\\mmdetection_cfg\\faster_rcnn_r50_fpn_coco.py",
                           mmdet_checkpoint_file = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\humanaction\\mm_checkpoint\\faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
                           mmpose_config_file_2D = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\mmpose\\configs\\body\\2d_kpt_sview_rgb_vid\\posewarper\\posetrack18\\hrnet_w48_posetrack18_384x288_posewarper_stage2.py",
                           mmpose_checkpoint_file_2D = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\humanaction\\mm_checkpoint\\hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth",
                           mmpose_config_file_3D = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\mmpose\\configs\\body\\3d_kpt_sview_rgb_vid\\video_pose_lift\\h36m\\videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py",
                           mmpose_checkpoint_file_3D = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\humanaction\\mm_checkpoint\\videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth",
                           use_multi_frames = True, # use multiple frames for inference on current frame
                           online = True, # to use only past frames when using multiple frames for inference on current frame
                           device = "cuda:0",
                           bbox_thr = 0.9, # Bounding box score threshold
                           use_oks_tracking = True, # Using OKS tracking
                           tracking_thr = 0.3, # Tracking threshold 
                           det_cat_id = 1, # Category id for bounding box detection model
                           num_instances = -1, # The number of 3D poses to be visualized in every frame. If less than 0, it will be set to the number of pose results in the first frame.
                           rebase_keypoint_height = True, # Rebase the predicted 3D pose so its lowest keypoint has a height of 0 (landing on the ground). This is useful for visualization when the model do not predict the global position of the 3D pose.
                           norm_pose_2d = True, # Scale the bbox (along with the 2D pose) to the average bbox scale of the dataset, and move the bbox (along with the 2D pose) to the average bbox center of the dataset. This is useful when bbox is small, especially in multi-person scenarios.
                           radius = 8, # Keypoint radius for visualization
                           thickness = 2, # Link thickness for visualization
                           show = False, # whether to show visualizations.
                           ):
    ####################################################
    #--------- First stage: 2D pose detection ---------#
    ####################################################
    print('Stage 1: 2D pose detection.')
    print('Initializing model...')
    person_det_model = init_detector(mmdet_config_file, mmdet_checkpoint_file, device=device)
    # Only "TopDown"
    pose_det_model = init_pose_model(mmpose_config_file_2D, mmpose_checkpoint_file_2D, device=device)
    # frame index offsets for inference, used in multi-frame inference setting
    indices = pose_det_model.cfg.data.test.data_cfg['frame_indices_test']
    # type de dataset ?
    pose_det_dataset = pose_det_model.cfg.data['test']['type']
    print(f"pose_det_dataset {pose_det_dataset}")
    # get dataset info
    dataset_info = pose_det_model.cfg.data['test'].get('dataset_info', None)
    print(f"dataset_info {dataset_info}")
    dataset_info = DatasetInfo(dataset_info)

    pose_det_results_list = []
    next_id = 0
    pose_det_results = []


    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    print('Running 2D pose detection inference...')
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        pose_det_results_last = pose_det_results

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(person_det_model, cur_frame)

        # keep the "person class" bounding boxes.
        person_det_results = process_mmdet_results(mmdet_results, det_cat_id)

        frames = collect_multi_frames(video, frame_id, indices, online)

        # make person results for current image
        pose_det_results, _ = inference_top_down_pose_model(
            pose_det_model,
            frames if use_multi_frames else cur_frame,
            person_det_results,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=pose_det_dataset,
            dataset_info=dataset_info,
            outputs=output_layer_names)

        # get track id for each person instance
        pose_det_results, next_id = get_track_id(
            pose_det_results,
            pose_det_results_last,
            next_id,
            use_oks=use_oks_tracking,
            tracking_thr=tracking_thr)

        pose_det_results_list.append(copy.deepcopy(pose_det_results))
        
    
    ####################################################
    #----------- Second stage: Pose lifting -----------#
    ####################################################
    print('Stage 2: 2D-to-3D pose lifting.')

    print('Initializing model...')
    pose_lift_model = init_pose_model(mmpose_config_file_3D, mmpose_checkpoint_file_3D, device=device)
    # Only "PoseLifter" model is supported for the 2nd stage
    pose_lift_dataset = pose_lift_model.cfg.data['test']['type']

    # convert keypoint definition
    for pose_det_results in pose_det_results_list:
        for res in pose_det_results:
            keypoints = res['keypoints']
            res['keypoints'] = convert_keypoint_definition(keypoints, pose_det_dataset, pose_lift_dataset)

    # load temporal padding config from model.data_cfg
    data_cfg = pose_lift_model.cfg.test_data_cfg

    pose_lift_dataset_info = pose_lift_model.cfg.data['test'].get('dataset_info', None)
    pose_lift_dataset_info = DatasetInfo(pose_lift_dataset_info)

    print('Running 2D-to-3D pose lifting inference...')
    video_results_vis = [] # storage for video 3D keypoints visualization
    video_results_export = [] # storage for video 3D keypoints json exports
    for i, pose_det_results in enumerate(mmcv.track_iter_progress(pose_det_results_list)):
        # extract and pad input pose2d sequence
        pose_results_2d = extract_pose_sequence(
            pose_det_results_list,
            frame_idx=i,
            causal=data_cfg.causal,
            seq_len=data_cfg.seq_len,
            step=data_cfg.seq_frame_interval)

        # 2D-to-3D pose lifting
        pose_lift_results = inference_pose_lifter_model(
            pose_lift_model,
            pose_results_2d=pose_results_2d,
            dataset=pose_lift_dataset,
            dataset_info=pose_lift_dataset_info,
            with_track_id=True,
            image_size=resolution,
            norm_pose_2d=norm_pose_2d)

        # Pose processing
        pose_lift_results_vis = []
        pose_lift_results_export = []
        for idx, res in enumerate(pose_lift_results):
            keypoints_3d = res['keypoints_3d']
            # exchange y,z-axis, and then reverse the direction of x,z-axis
            keypoints_3d = keypoints_3d[..., [0, 2, 1]]
            keypoints_3d[..., 0] = -keypoints_3d[..., 0]
            keypoints_3d[..., 2] = -keypoints_3d[..., 2]
            # rebase height (z-axis)
            if rebase_keypoint_height:
                keypoints_3d[..., 2] -= np.min(keypoints_3d[..., 2], axis=-1, keepdims=True)
            res['keypoints_3d'] = keypoints_3d
            # add title
            det_res = pose_det_results[idx]
            instance_id = det_res['track_id']
            res['title'] = f'Prediction ({instance_id})'
            # only visualize the target frame
            res['keypoints'] = det_res['keypoints']
            res['bbox'] = det_res['bbox']
            res['track_id'] = instance_id
            pose_lift_results_vis.append(res)
            # adjust for export
            res_export = copy.deepcopy(pose_lift_results[idx])
            res_export['keypoints_3d'] = keypoints_3d.tolist()
            res_export['keypoints'] = det_res['keypoints'].tolist()
            res_export['bbox'] = det_res['bbox'].tolist()
            pose_lift_results_export.append(res_export)

        # aggregating detections of a frame
        frame_results_export = {
            "id" : i,
            "num_pedestrians" : len(pose_lift_results_export),
            "detections" : copy.deepcopy((pose_lift_results_export))
        }
        # keep at video level
        video_results_vis.append(pose_lift_results_vis)
        video_results_export.append(frame_results_export)

    return pose_det_results_list, video_results_vis, video_results_export, pose_lift_model, pose_lift_dataset, pose_lift_dataset_info


def detect_video_keypoints(original_video_dir, 
                           keypoint_video_dir, 
                           keypoint_json_dir, 
                           file_name,
                           mmdet_config_file = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\mmpose\\demo\\mmdetection_cfg\\faster_rcnn_r50_fpn_coco.py",
                           mmdet_checkpoint_file = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\humanaction\\mm_checkpoint\\faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
                           mmpose_config_file_2D = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\mmpose\\configs\\body\\2d_kpt_sview_rgb_vid\\posewarper\\posetrack18\\hrnet_w48_posetrack18_384x288_posewarper_stage2.py",
                           mmpose_checkpoint_file_2D = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\humanaction\\mm_checkpoint\\hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth",
                           mmpose_config_file_3D = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\mmpose\\configs\\body\\3d_kpt_sview_rgb_vid\\video_pose_lift\\h36m\\videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py",
                           mmpose_checkpoint_file_3D = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\humanaction\\mm_checkpoint\\videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth",
                           use_multi_frames = True, # use multiple frames for inference on current frame
                           online = True, # to use only past frames when using multiple frames for inference on current frame
                           device = "cuda:0",
                           bbox_thr = 0.9, # Bounding box score threshold
                           use_oks_tracking = True, # Using OKS tracking
                           tracking_thr = 0.3, # Tracking threshold 
                           det_cat_id = 1, # Category id for bounding box detection model
                           save_out_video = True,
                           num_instances = -1, # The number of 3D poses to be visualized in every frame. If less than 0, it will be set to the number of pose results in the first frame.
                           rebase_keypoint_height = True, # Rebase the predicted 3D pose so its lowest keypoint has a height of 0 (landing on the ground). This is useful for visualization when the model do not predict the global position of the 3D pose.
                           norm_pose_2d = True, # Scale the bbox (along with the 2D pose) to the average bbox scale of the dataset, and move the bbox (along with the 2D pose) to the average bbox center of the dataset. This is useful when bbox is small, especially in multi-person scenarios.
                           radius = 8, # Keypoint radius for visualization
                           thickness = 2, # Link thickness for visualization
                           show = False # whether to show visualizations.
                           ):
    # Here to change the video path ! 
    video_path = os.path.join(original_video_dir, file_name)
    out_video_root = keypoint_video_dir
    out_json_root = keypoint_json_dir
    # video_path = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\NTU_samples_and_inference_code\\videos\\S001C001P001R001A006_rgb.avi"
    # video_path = "C:\\\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\data\\input_test\\video_JM_20s.mp4"
    # out_video_root = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\NTU_samples_and_inference_code\\out"
    # out_json_root = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\NTU_samples_and_inference_code\\jsons"
    # mmdet_checkpoint_file https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    # mmpose_checkpoint_file_2D https://download.openmmlab.com/mmpose/top_down/posewarper/hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth
    # mmpose_checkpoint_file_3D https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth
    video = mmcv.VideoReader(video_path)

    pose_det_results_list, video_results_vis, video_results_export, pose_lift_model, pose_lift_dataset, pose_lift_dataset_info  = detect_stream_keypoints(video, video.resolution) # TODO make something to pass all the config mess (dict or class)

    # Prepare to save the output video
    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video.fps
        writer = None

    for i, pose_det_results in enumerate(mmcv.track_iter_progress(pose_det_results_list)):
        pose_lift_results_vis = video_results_vis[i]['detections']
        # Visualization
        if num_instances < 0:
            num_instances = len(pose_lift_results_vis)
        img_vis = vis_3d_pose_result(
            pose_lift_model,
            result=pose_lift_results_vis,
            img=video[i],
            dataset=pose_lift_dataset,
            dataset_info=pose_lift_dataset_info,
            out_file=None,
            radius=radius,
            thickness=thickness,
            num_instances=num_instances,
            show=show)

        if save_out_video:
            if writer is None:
                writer = cv2.VideoWriter(
                    osp.join(out_video_root,
                             file_name), fourcc,
                    fps, (img_vis.shape[1], img_vis.shape[0]))
            writer.write(img_vis)

    if save_out_video:
        writer.release()

    ####################################################
    #------- Additional step : export keypoints -------#
    ####################################################

    sequence_name = osp.splitext(osp.basename(video_path))[0]
    # aggregating detections of all frames in a video sequence
    result_json = {
            "sequence_name" : sequence_name,
            "num_frames" : len(video),
            "frames" : video_results_export
        }
    # sort the result alphabetically by keys:
    json_string = json.dumps(result_json, indent=4, sort_keys=False)
    # Write json file
    with open(osp.join(out_json_root, f'{sequence_name}.json'), 'w') as outfile:
        outfile.write(json_string)


def detect_keypoints(original_video_dir, keypoint_video_dir, keypoint_json_dir, file_names=None):
    if file_names is None:
        file_names = os.listdir(original_video_dir)
    for file_name in file_names:
        detect_video_keypoints(original_video_dir, keypoint_video_dir, keypoint_json_dir, file_name)


if __name__ == '__main__':
    original_video_dir = 'data/input/video'
    keypoint_video_dir = 'data/output/keypoint_video'
    keypoint_json_dir = 'data/output/keypoint_json'
    detect_keypoints(original_video_dir, keypoint_video_dir, keypoint_json_dir)