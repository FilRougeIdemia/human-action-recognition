# Goal is https://github.com/open-mmlab/mmaction2 
# Skeleton-based Spatio-Temporal Action Detection and Action Recognition Results on Kinetics-400


import mmcv
from mmdet.apis import init_detector, inference_detector
from mmpose.datasets import DatasetInfo
from mmpose.apis import (init_pose_model, process_mmdet_results, collect_multi_frames, 
                         inference_top_down_pose_model, get_track_id, extract_pose_sequence,
                         inference_pose_lifter_model)
import copy
import cv2
import numpy as np
import os.path as osp
import os
import copy as cp
from keypoint_detection import convert_keypoint_definition
from predict_new_data import predict_on_stream, actions, classes


FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


plate_blue = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4-03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
plate_blue = plate_blue.split('-')
plate_blue = [hex2color(h) for h in plate_blue]
plate_green = '004b23-006400-007200-008000-38b000-70e000-004b23-006400-007200-008000-38b000-70e000'
plate_green = plate_green.split('-')
plate_green = [hex2color(h) for h in plate_green]


def visualize(frames, annotations, plate=plate_blue, max_num=9):
    """Visualize frames with predicted annotations.
    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted results.
        plate (str): The plate used for visualization. Default: plate_blue.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.
    Returns:
        list[np.ndarray]: Visualized frames.
    """

    assert max_num + 1 <= len(plate)
    plate = [x[::-1] for x in plate]
    frames_ = cp.deepcopy(frames)
    nf, na = len(frames), len(annotations)
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    for i in range(na):
        anno = annotations[i]
        if anno is None:
            continue
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_[ind]
            for ann in anno:
                box = ann[0].astype(np.int64)
                label = ann[1]
                if not len(label):
                    continue
                score = np.round(ann[2], 3)
                st, ed = tuple(box[:2]), tuple(box[2:])
                cv2.rectangle(frame, st, ed, plate[0], 2)
                for k, lb in enumerate(label):
                    if k >= max_num:
                        break
                    text = abbrev(lb)
                    text = ': '.join([text, str(score[k])])
                    location = (0 + st[0], st[1] - k * 18) #18 + k * 18 + st[1])
                    textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                               THICKNESS)[0]
                    textwidth = textsize[0]
                    diag0 = (location[0] + textwidth, location[1] - 14)
                    diag1 = (location[0], location[1] + 2)
                    cv2.rectangle(frame, diag0, diag1, plate[k + 1], -1)
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)

    return frames_


def abbrev(name):
    """Get the abbreviation of label name:
    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name


def init_mm_models(mmdet_config_file = "../mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
                    mmdet_checkpoint_file = "data/config/mm_checkpoint/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
                    mmpose_config_file_2D = "../mmpose/configs/body/2d_kpt_sview_rgb_vid/posewarper/posetrack18/hrnet_w48_posetrack18_384x288_posewarper_stage2.py",
                    mmpose_checkpoint_file_2D = "data/config/mm_checkpoint/hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth",
                    mmpose_config_file_3D = "../mmpose/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py",
                    mmpose_checkpoint_file_3D = "data/config/mm_checkpoint/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth",
                    device = "cuda:0",
                    ):
    print('Stage 1: 2D pose detection.')
    print('Initializing model...')
    person_det_model = init_detector(mmdet_config_file, mmdet_checkpoint_file, device=device)
    # Only "TopDown"
    pose_det_model = init_pose_model(mmpose_config_file_2D, mmpose_checkpoint_file_2D, device=device)

    print('Stage 2: 2D-to-3D pose lifting.')
    print('Initializing model...')
    pose_lift_model = init_pose_model(mmpose_config_file_3D, mmpose_checkpoint_file_3D, device=device)
    return person_det_model, pose_det_model, pose_lift_model


def process_stream_offline(video, # an iterable containing frames of the video stream
                            output_path,
                            person_det_model,
                            pose_det_model,
                            pose_lift_model,
                           device = "cuda:0",
                           online=True,
                           buffer_size=5,
                           window_size=45,
                           stride=30,
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
                           model_type = "LSTM", # Either LSTM or 2S-AGCN
                           ):
    ####################################################
    #--------- Prepare the two stages ---------#
    ####################################################
    # Prepare stage 1 : 2D pose detection
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
    
    # Prepare stage 2 : 
    # Only "PoseLifter" model is supported for the 2nd stage
    pose_lift_dataset = pose_lift_model.cfg.data['test']['type']
    # load temporal padding config from model.data_cfg
    data_cfg = pose_lift_model.cfg.test_data_cfg

    pose_lift_dataset_info = pose_lift_model.cfg.data['test'].get('dataset_info', None)
    pose_lift_dataset_info = DatasetInfo(pose_lift_dataset_info)
    skeletons = []
    
    # Prepare visualization
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), video.fps, video.resolution)
    labels = None

    # Loop over incoming frames
    for frame_id, cur_frame in enumerate(video):
        ####################################################
        #--------- First stage: 2D pose detection ---------#
        ####################################################
        pose_det_results_last = pose_det_results

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(person_det_model, cur_frame)

        # keep the "person class" bounding boxes.
        person_det_results = process_mmdet_results(mmdet_results, det_cat_id)

        frames = collect_multi_frames(video, frame_id, indices, online)

        # make person results for current image
        pose_det_results, _ = inference_top_down_pose_model(
            pose_det_model,
            frames,
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

        # store n pose_det_results, with n=buffer_size
        # as pose2d sequence needs, well, a sequence
        pose_det_results_list.append(copy.deepcopy(pose_det_results))
        # In case there is someone on the frame
        if np.all([len(pdr) != 0 for pdr in pose_det_results_list]):
            # check if there is a full buffer
            if len(pose_det_results_list) >= buffer_size :
                ####################################################
                #----------- Second stage: Pose lifting -----------#
                ####################################################
                # extract and pad input pose2d sequence
                pose_results_2d = extract_pose_sequence(
                    pose_det_results_list,
                    frame_idx=buffer_size-1, #frame_id,#int(1+(buffer_size/2)),
                    causal=True,#data_cfg.causal,
                    seq_len=data_cfg.seq_len, #buffer_size, 
                    step=data_cfg.seq_frame_interval)

                # 2D-to-3D pose lifting
                pose_lift_results = inference_pose_lifter_model(
                    pose_lift_model,
                    pose_results_2d=pose_results_2d,
                    dataset=pose_lift_dataset,
                    dataset_info=pose_lift_dataset_info,
                    with_track_id=True,
                    image_size=video.resolution,
                    norm_pose_2d=norm_pose_2d)
                
                if len(pose_det_results)>1 or len(pose_lift_results)>1:
                    print("two people ?")
                # Format results
                for res in pose_det_results:
                    keypoints = res['keypoints']
                    res['keypoints'] = convert_keypoint_definition(keypoints, pose_det_dataset, pose_lift_dataset)
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

                if model_type == "LSTM":
                    skeletons.append(res['keypoints']) # 2D case
                elif model_type == "2S-AGCN":
                    skeletons.append(res['keypoints_3d']) # 3D case
                if len(skeletons) >= window_size:
                    # skeletons are added one-by-one, as soon as there are enough skeletons a prediction is made.
                    skel_stream = np.array(skeletons)
                    if model_type == "LSTM":
                        skel_stream = skel_stream[:,:,:2]
                    elif model_type == "2S-AGCN":
                        skel_stream = skel_stream[:,:,:3]
                    probs = predict_on_stream(skel_stream, is_sliding_window=False, model_type=model_type)
                    # keep only top 3 probs
                    top_indices = np.argsort(probs[-1,...])[probs.shape[-1]-3:]
                    labels = [actions[c-1] for c in classes]
                    #skeletons.pop(0) => equivalent to stride=1
                    skeletons = skeletons[stride:] # does not work with stride longer than the window_size with this code
                if labels is not None:
                    # Performing visualization
                    # with the latest available annotations
                    annotations = (res['bbox'][:4], [labels[i] for i in top_indices], probs[-1,...][top_indices])
                    vis_frames = visualize(frames=[cur_frame], annotations=[[annotations]])
                    cur_frame = vis_frames[0]
                # Pop to keep only buffer    
                pose_det_results_list.pop(0)
        else:
            # Empty list to avoid frames without people
            pose_det_results_list = []
        
        # At the end of the process for a frame, we write it
        # if there is a visualization it has been added to the current frame 
        writer.write(cur_frame)

    writer.release()
        


if __name__ == '__main__':
    # constants 
    # mainly paths 
    original_video_dir = 'data/input/video'
    keypoint_npy_dir = 'data/output/keypoint_npy'
    keypoint_video_dir = 'data/output/keypoint_video'
    keypoint_json_dir = 'data/output/keypoint_json'
    prediction_dir = 'data/output/prediction' # contains both sliding window and whole
    prediction_video_dir = 'data/output/prediction_keypoint_video'
    #file_name = 'video_JM_10s_HD.mp4'
    file_name='VID20230112100603.mp4'
    
    base_file_name = file_name[:-4]
    stream_dir = os.path.join('data/output/stream/offline')

    is_sliding_window = True
    window_size = 45
    stride = 30
    model_type = "2S-AGCN"

    video = mmcv.VideoReader(os.path.join(original_video_dir, file_name))

    person_det_model, pose_det_model, pose_lift_model = init_mm_models()

    process_stream_offline(video, osp.join(stream_dir, file_name), person_det_model, pose_det_model, pose_lift_model, window_size=window_size, stride=stride, model_type=model_type)