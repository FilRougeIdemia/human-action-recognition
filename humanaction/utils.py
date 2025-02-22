# Utils
import numpy as np
import mmcv
import math

# For 2S-AGCN
def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) # np.clip(a, a_min, a_max); np.arccos gives an angle between [0, pi]


def rotation_matrix(axis, theta): # detil in calculation?
    """Return the rotation matrix associated with counterclockwise rotation
    about the given axis by theta radians."""
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def pre_normalization(data, zaxis=[0, 7], xaxis=[14, 11]): 
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N C T V M -> N M T V C

    #print('pad the null frames with the previous frames')
    for i_s, skeleton in enumerate(s):
        #if skeleton.sum() == 0:
            #print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0: # if the content of the first frame of this person is 0
                index = (person.sum(-1).sum(-1) != 0) # if the last joint in the last frame exists, index = 1, if not 0
                tmp = person[index].copy() 
                person *= 0 # clear out the content
                person[:len(tmp)] = tmp # paste the same content

            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate(
                            [person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break

    #print('sub the center joint #7 (spine joint in ntu and '
          #'neck joint in kinetics)')
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, 7:8, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    #print('parallel the bone between hip(jpt 0) and '
          #'spine(jpt 7) of the first person to the z axis')
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        '''
        The cross product is a binary operation that takes two vectors as input and returns a third vector 
        that is orthogonal to the input vectors and has a magnitude equal to the area of the parallelogram 
        determined by the input vectors.
        '''
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    #print('parallel the bone between right shoulder(jpt 14) and '
          #'left shoulder(jpt 11) of the first person to the x axis')
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data

def get_nonzero_std(s):  # T V C
    index = s.sum(-1).sum(-1) != 0
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :,
                                                    2].std()  # three channels
    else:
        s = 0
    return s

def preprocess_skeleton_sequence_for_inference(data):
    # takes a sequence of skeletons as input 
    # returns input for inference with 2S-AGCN
    data = data.transpose(3, 1, 2, 0)
    data = pre_normalization(data[np.newaxis, ...])
    data = data.transpose(0, 4, 2, 3, 1)
    anno = dict()
    anno['total_frames'] = data.shape[2]
    anno['keypoint'] = data[0]
    anno['frame_dir'] = "NotImportant"
    anno['img_shape'] = (1080, 1920)
    anno['original_shape'] = (1080, 1920)
    anno['label'] = [] 
    return anno