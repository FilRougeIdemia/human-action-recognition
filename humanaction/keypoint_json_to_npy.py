# Convert keypoints json to npy
# takes a folder

import os
import numpy as np
import json
import warnings

with open("data/actions.txt", 'r') as f:
    actions = [line.strip() for line in f.readlines()]

def rename_in_ntu_style(file_name, file_name_idx):
    for idx, action in enumerate(actions):
        if action.lower() in file_name.lower():
            new_file_name = f"I{str(file_name_idx).zfill(3)}XXXXXXXXXXXXA{str(idx + 1).zfill(3)}"
            return new_file_name

def convert_json_to_npy(keypoint_json_dir, keypoint_npy_dir, file_names=None, ntu_style=False):
    # load data
    #data_dir = os.path.join('data', 'input_test')
    #data_files = os.listdir(data_dir)[:10]
    #output_dir = os.path.join('data', 'output_test')
    data_dir = keypoint_json_dir
    data_files = file_names
    if file_names is None:
        data_files = os.listdir(data_dir)
    output_dir_2D = os.path.join(keypoint_npy_dir, '2D')
    output_dir_3D = os.path.join(keypoint_npy_dir, '3D')


    incomplete_files_2d = list()
    incomplete_files_3d = list()
    for file_name_idx, file_name in enumerate(data_files):
        with open(os.path.join(data_dir, file_name)) as f:
            skeleton_file = json.load(f)
            num_frames = skeleton_file['num_frames']

            # Taking info from the last frame which is less likely to be empty 
            #nb_joints = len(skeleton_file['frames'][-1]['detections'][0]['keypoints_3d'])
            #nb_persons = skeleton_file['frames'][-1]['num_pedestrians']
            for frame in skeleton_file['frames']:
                if frame['detections']:
                    nb_joints = len(frame['detections'][0]['keypoints_3d'])
                    nb_persons = frame['num_pedestrians']
                    break
            
            for person in range(nb_persons):
                if person>1:
                    warnings.warn("More than 1 person in this video")
                data2D = np.zeros((num_frames, nb_joints, 2))
                data3D = np.zeros((num_frames, nb_joints, 3))
                if skeleton_file['num_frames'] >= num_frames:
                    for i_frame, frame in enumerate(skeleton_file['frames']):
                        try:
                            data2D[i_frame] = np.array(frame['detections'][person]['keypoints'])[:, 0:2]
                            data3D[i_frame] = np.array(frame['detections'][person]['keypoints_3d'])[:, 0:3]
                        except IndexError:
                            incomplete_files_2d.append(file_name)
                            incomplete_files_3d.append(file_name)
                            continue
                    
                    file_name = rename_in_ntu_style(file_name, file_name_idx) if ntu_style else file_name
                    os.makedirs(output_dir_2D) if not os.path.exists(output_dir_2D) else None
                    os.makedirs(output_dir_3D) if not os.path.exists(output_dir_3D) else None
                    np.save(os.path.join(output_dir_2D, file_name+'.npy'), data2D)
                    np.save(os.path.join(output_dir_3D, file_name+'.npy'), data3D)
                    break
    print("List of all 'incomplete' files :")
    print(incomplete_files_2d)
    print(incomplete_files_3d)

if __name__ == '__main__':
    keypoint_json_dir = 'data/output/keypoint_json'
    keypoint_npy_dir = 'data/output/keypoint_npy'
    convert_json_to_npy(keypoint_json_dir, keypoint_npy_dir)
    #keypoint_json_dir = '/home/users/jma-21/BGDIA706 - Fil Rouge/acquisition_sacs_decoupees/jsons'
    #keypoint_npy_dir = 'data/input/acquisition_sacs'
    #convert_json_to_npy(keypoint_json_dir, keypoint_npy_dir, ntu_style=True)