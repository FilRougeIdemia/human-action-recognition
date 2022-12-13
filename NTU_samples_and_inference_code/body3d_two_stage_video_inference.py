# Copyright (c) OpenMMLab. All rights reserved.
import copy
from pathlib import Path
import warnings
from argparse import ArgumentParser
from tqdm import tqdm
import json
import mmcv
import numpy as np
import re
from mmpose.apis import (collect_multi_frames, extract_pose_sequence,
                         get_track_id, inference_pose_lifter_model,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_3d_pose_result)
from mmpose.datasets import DatasetInfo
from mmpose.models import PoseLifter, TopDown
from body3d_two_stage_video_demo import convert_keypoint_definition

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def init_models_and_configs(args):
    assert args.show or (args.output_path != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    # First stage: 2D pose detection
    print('Stage 1: 2D pose detection.')

    print('Initializing model...')
    person_det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    pose_det_model = init_pose_model(
        args.pose_detector_config,
        args.pose_detector_checkpoint,
        device=args.device.lower())

    assert isinstance(pose_det_model, TopDown), 'Only "TopDown"' \
                                                'model is supported for the 1st stage (2D pose detection)'

    # frame index offsets for inference, used in multi-frame inference setting
    if args.use_multi_frames:
        assert 'frame_indices_test' in pose_det_model.cfg.data.test.data_cfg
        indices = pose_det_model.cfg.data.test.data_cfg['frame_indices_test']

    pose_det_dataset = pose_det_model.cfg.data['test']['type']
    # get datasetinfo
    dataset_info = pose_det_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    print('Initializing model...')
    pose_lift_model = init_pose_model(
        args.pose_lifter_config,
        args.pose_lifter_checkpoint,
        device=args.device.lower())

    assert isinstance(pose_lift_model, PoseLifter), \
        'Only "PoseLifter" model is supported for the 2nd stage ' \
        '(2D-to-3D lifting)'
    pose_lift_dataset = pose_lift_model.cfg.data['test']['type']

    # load temporal padding config from model.data_cfg
    if hasattr(pose_lift_model.cfg, 'test_data_cfg'):
        data_cfg = pose_lift_model.cfg.test_data_cfg
    else:
        data_cfg = pose_lift_model.cfg.data_cfg

    pose_lift_dataset_info = pose_lift_model.cfg.data['test'].get(
        'dataset_info', None)
    if pose_lift_dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        pose_lift_dataset_info = DatasetInfo(pose_lift_dataset_info)

    return person_det_model, pose_det_model, pose_det_dataset,dataset_info,indices,pose_lift_model,pose_lift_dataset,pose_lift_dataset_info,data_cfg


def main():
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument(
        'pose_detector_config',
        type=str,
        default=None,
        help='Config file for the 1st stage 2D pose detector')
    parser.add_argument(
        'pose_detector_checkpoint',
        type=str,
        default=None,
        help='Checkpoint file for the 1st stage 2D pose detector')
    parser.add_argument(
        'pose_lifter_config',
        help='Config file for the 2nd stage pose lifter model')
    parser.add_argument(
        'pose_lifter_checkpoint',
        help='Checkpoint file for the 2nd stage pose lifter model')
    parser.add_argument(
        '--input_path', type=str, default='', help='Video path')
    parser.add_argument(
        '--rebase-keypoint-height',
        action='store_true',
        help='Rebase the predicted 3D pose so its lowest keypoint has a '
        'height of 0 (landing on the ground). This is useful for '
        'visualization when the model do not predict the global position '
        'of the 3D pose.')
    parser.add_argument(
        '--norm-pose-2d',
        action='store_true',
        help='Scale the bbox (along with the 2D pose) to the average bbox '
        'scale of the dataset, and move the bbox (along with the 2D pose) to '
        'the average bbox center of the dataset. This is useful when bbox '
        'is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=-1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument(
        '--output_path',
        type=str,
        default='vis_results',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.9,
        help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=8,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Link thickness for visualization')
    parser.add_argument(
        '--smooth',
        action='store_true',
        help='Apply a temporal filter to smooth the 2D pose estimation '
        'results. See also --smooth-filter-cfg.')
    parser.add_argument(
        '--smooth-filter-cfg',
        type=str,
        default='configs/_base_/filters/one_euro.py',
        help='Config file of the filter to smooth the pose estimation '
        'results. See also --smooth.')
    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the 2D pose'
        'detection stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the 2D pose'
        'detection stage. Default: False.')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    person_det_model, pose_det_model, pose_det_dataset,dataset_info,indices,pose_lift_model,pose_lift_dataset,pose_lift_dataset_info,data_cfg = init_models_and_configs(args)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    subset_of_actions = [x for x in input_path.iterdir() if int(re.findall("A[0-9]+", str(x))[0][1:]) in [6,7,8,9,15,25,31,33,43]] #only keeping actions used before

    print(f"keeping only a subset of {len(subset_of_actions)} out of {len([x for x in input_path.iterdir()])} videos")

    for video_path in tqdm(subset_of_actions,total=len(subset_of_actions)):
        json_out_path = output_path / Path(str(video_path.stem) + ".json")
        if Path.is_file(json_out_path):
            print(f"skipping {video_path} as {json_out_path} already exists")
            continue
        video = mmcv.VideoReader(str(video_path))
        assert video.opened, f'Failed to load video file {args.video_path}'

        pose_det_results_list = []
        next_id = 0
        pose_det_results = []

        result_json = {
            "sequence_name" : str(video_path.stem),
            "num_frames" : len(video),
            "frames" : []
        }

        print('Running 2D pose detection inference...')
        for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
            if frame_id == 5:
                break
            pose_det_results_last = pose_det_results

            # test a single image, the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(person_det_model, cur_frame)

            # keep the person class bounding boxes.
            person_det_results = process_mmdet_results(mmdet_results,
                                                       args.det_cat_id)

            if args.use_multi_frames:
                frames = collect_multi_frames(video, frame_id, indices,
                                              args.online)

            # make person results for current image
            pose_det_results, _ = inference_top_down_pose_model(
                pose_det_model,
                frames if args.use_multi_frames else cur_frame,
                person_det_results,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=pose_det_dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=None)

            # get track id for each person instance
            pose_det_results, next_id = get_track_id(
                pose_det_results,
                pose_det_results_last,
                next_id,
                use_oks=args.use_oks_tracking,
                tracking_thr=args.tracking_thr)

            pose_det_results_list.append(copy.deepcopy(pose_det_results))

        # Second stage: Pose lifting
        print('Stage 2: 2D-to-3D pose lifting.')

        # convert keypoint definition
        for pose_det_results in pose_det_results_list:
            for res in pose_det_results:
                keypoints = res['keypoints']
                res['keypoints'] = convert_keypoint_definition(
                    keypoints, pose_det_dataset, pose_lift_dataset)

        print('Running 2D-to-3D pose lifting inference...')
        pose_lift_results_list = []
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
                image_size=video.resolution,
                norm_pose_2d=args.norm_pose_2d)

            # Pose processing
            pose_lift_results_vis_per_frame = []
            for idx, res in enumerate(pose_lift_results):
                keypoints_3d = res['keypoints_3d']
                # exchange y,z-axis, and then reverse the direction of x,z-axis
                keypoints_3d = keypoints_3d[..., [0, 2, 1]]
                keypoints_3d[..., 0] = -keypoints_3d[..., 0]
                keypoints_3d[..., 2] = -keypoints_3d[..., 2]
                # rebase height (z-axis)
                if args.rebase_keypoint_height:
                    keypoints_3d[..., 2] -= np.min(
                        keypoints_3d[..., 2], axis=-1, keepdims=True)
                res['keypoints_3d'] = keypoints_3d.tolist()
                # add title
                det_res = pose_det_results[idx]
                instance_id = det_res['track_id']
                # only visualize the target frame
                res['keypoints'] = det_res['keypoints'].tolist()
                res['bbox'] = det_res['bbox'].tolist()
                res['pedestrian_id'] = instance_id
                del res['track_id']
                pose_lift_results_vis_per_frame.append(res)

            # aggregating detections of a frame
            frame = {
                "id" : i,
                "num_pedestrians" : len(pose_lift_results_vis_per_frame),
                "detections" : copy.deepcopy((pose_lift_results_vis_per_frame))
            }
            pose_lift_results_list.append(frame)

        # aggregating detections of all frames in a video sequence
        result_json["frames"] = pose_lift_results_list
        # sort the result alphabetically by keys:
        json_string = json.dumps(result_json, indent=4, sort_keys=False)
        # Write json file
        with open(json_out_path, 'w') as outfile:
            outfile.write(json_string)

    # with open(out_path, 'r') as j:
    #     contents = json.loads(j.read())


if __name__ == '__main__':
    main()
