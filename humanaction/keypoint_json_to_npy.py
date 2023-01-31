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

    incomplete_files_3d = []
    for file_name in data_files:
        with open(os.path.join(data_dir, file_name)) as f:
            skeleton_file = json.load(f)
            try:
                num_frames = skeleton_file['num_frames']
                
                nb_joints = len(skeleton_file['frames'][0]['detections'][0]['keypoints_3d'])
                nb_persons = skeleton_file['frames'][0]['num_pedestrians']
                for person in range(nb_persons):
                    if person>1:
                        warnings.warn("More than 1 person in this video")
                    data = np.zeros((num_frames, nb_joints, 3))
                    if skeleton_file['num_frames'] >= num_frames:
                        for i_frame, frame in enumerate(skeleton_file['frames']):
                                if i_frame < num_frames:
                                    data[i_frame] = np.array(frame['detections'][person]['keypoints_3d'])[:, 0:3]
                                else:
                                    break
                        np.save(os.path.join(output_dir, file_name[:-9]+'.npy'), data)
                        break
            except IndexError:
                incomplete_files_3d.append(file_name)
                continue
    print("List of all 'incomplete' files :")
    print(incomplete_files_3d)

if __name__ == '__main__':
    main()