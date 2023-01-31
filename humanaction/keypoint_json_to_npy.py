# Convert keypoints json to npy
# takes a folder

import os
import numpy as np
import json
import warnings

def main():
    # load data
    data_dir = os.path.join('data', 'input_test')
    data_files = os.listdir(data_dir)[:10]
    output_dir = os.path.join('data', 'output_test')

    incomplete_files_2d = set()
    incomplete_files_3d = set()
    for file_name in data_files:
        with open(os.path.join(data_dir, file_name)) as f:
            skeleton_file = json.load(f)
            num_frames = skeleton_file['num_frames']
            
            nb_joints = len(skeleton_file['frames'][0]['detections'][0]['keypoints_3d'])
            nb_persons = skeleton_file['frames'][0]['num_pedestrians']
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
                            incomplete_files_2d.update(file_name)
                            incomplete_files_3d.update(file_name)
                            continue
                    

                    np.save(os.path.join(output_dir, file_name[:-9]+'_2D.npy'), data2D)
                    np.save(os.path.join(output_dir, file_name[:-9]+'_3D.npy'), data3D)
                    break
    print("List of all 'incomplete' files :")
    print(incomplete_files_2d)
    print(incomplete_files_3d)

if __name__ == '__main__':
    main()