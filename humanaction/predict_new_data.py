# Model
# Declare a LSTM and train it on NTU skeleton action data
import os
import os.path as osp
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import copy
from model import HumanActionDataset, ActionLSTM, PadSequence
import cv2
from mmaction.apis import init_recognizer, inference_recognizer
from utils import pre_normalization, preprocess_skeleton_sequence_for_inference

# setting the device as the GPU if available, else the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: {}".format(device))


with open("data/actions.txt", 'r') as actions_file:
    actions = [line.replace('\n', '') for line in actions_file.readlines()]
    actions_file.close()
classes = [6, 7, 8, 9, 15, 25, 31, 33, 43, 121, 122, 123, 89, 87, 88, 90]
for i,elem in enumerate(classes):
    print("class {} : {}".format(i, actions[elem-1]))

# Fill-in the zeros in the skeleton npy
def fill_zeros(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                if arr[i][j][k] == 0:
                    if i > 0:
                        arr[i][j][k] = arr[i-1][j][k]
                else:
                    prev_value = arr[i][j][k]
                    break
    return arr

# Define the sliding window function
def sliding_window(data, window_size, stride):
    window_size = int(window_size)
    stride = int(stride)
    nb_frames = data.shape[1]
    num_windows = int(np.ceil((nb_frames - window_size + 1) / stride)) + 1
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        result = data[:,start:end,:]
        yield start, result.shape[1], result  


def duplicate_array(arr, stride):
    result = []
    rg = int(arr[0][1])
    for i in range(len(arr)):
        if i>0:
            rg = stride
        for j in range(rg):
                result.append(arr[i][2:])
    return np.array(result)


def predict_on_stream(stream, is_sliding_window=False, model_type="LSTM"):
    # Complete missing values in stream
    filled = fill_zeros(stream)
    
    # LSTM
    if model_type == "LSTM":
        # Load model
        model = ActionLSTM(nb_classes=len(classes), input_size=2*17, hidden_size_lstm=256, hidden_size_classifier=128, num_layers=1, device=device)
        model.to(device)
        model.load_state_dict(torch.load("models_saved/action_lstm_2D_luggage_0410.pt", map_location=torch.device('cuda:0')))
        model.eval()
        # Transform in tensor and reshape
        tensor = torch.Tensor(filled)
        tensor = tensor.reshape((1, tensor.shape[0], 2*17))/1000 # todo manage this with Dataloader ?

        if is_sliding_window:
            input = tensor.to(device)
            h_n, c_n = None, None
            output, h_n, c_n = model(input, h_n, c_n)
            h_n, c_n = copy.copy(h_n).to(device), copy.copy(c_n).to(device)
            sm = nn.Softmax(dim=1).to(device)    
            probs_ = sm(output).reshape(len(classes)).detach().cpu().numpy()
        else:
            stream_probs = []
            for frame_count in range(len(stream)):
                # Let's go through each frame of the window
                x = tensor[:, :frame_count+1, ...].to(device) # first element, data (and no label)
                h_n, c_n = None, None
                output, h_n, c_n = model(x, h_n, c_n)
                h_n, c_n = copy.copy(h_n).to(device), copy.copy(c_n).to(device)
                sm = nn.Softmax(dim=1).to(device)
                frames_probs = sm(output).reshape(len(classes)).detach().cpu().numpy()
                stream_probs.append(np.expand_dims(frames_probs, axis=0))
            stream_probs =  np.concatenate(stream_probs)
    # 2S-AGCN
    elif model_type == "2S-AGCN":
        # Load model
        config_file = '../mmaction2/configs/skeleton/2s-agcn/2sagcn_mmpose_keypoint_3d.py'  # Replace with the path to your model's configuration file
        checkpoint_file = '../mmaction2/configs/skeleton/2s-agcn/best_top1_acc_epoch_8.pth'  # Replace with the path to your model's checkpoint file
        model = init_recognizer(config_file, checkpoint_file, device='cuda:0')
        # Reshape (no Tensor here contrary to LSTM)
        filled = filled.reshape((1, filled.shape[0], 3*17))

        if is_sliding_window:
            data = filled.reshape(1, filled.shape[1], 17, 3)
            anno = preprocess_skeleton_sequence_for_inference(data)
            probs_ = inference_recognizer(model, anno)
            stream_probs = np.tile(probs_, (len(stream), 1))
        else:
            stream_probs = []
            for frame_count in range(len(stream)):
                # Let's go through each frame of the window
                data = filled.reshape(1, filled.shape[1], 17, 3)[:, :frame_count+1, ...]
                anno = preprocess_skeleton_sequence_for_inference(data)
                frames_probs_ = inference_recognizer(model, anno)
                # 2S-AGCN needs transformation of the results
                frames_probs = torch.zeros(len(classes)) # number of classes
                # Fill in the tensor with the values from frames_probs
                for frame, prob in frames_probs_:
                    frames_probs[frame] = float(prob)
                stream_probs.append(np.expand_dims(frames_probs, axis=0))
            stream_probs =  np.concatenate(stream_probs)
    return stream_probs


def predict_on_videos(keypoint_npy_dir, keypoint_video_dir, prediction_dir, prediction_video_dir, file_names=None, is_sliding_window=False):
    # constant that can become arguments
    data2D_dir = keypoint_npy_dir
    if file_names is None:
        data2D_files = os.listdir(data2D_dir)
    else:
        data2D_files = file_names

    #output_dir = os.path.join('data', 'output_test')
    # path to video folder
    #out_path = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\NTU_samples_and_inference_code\\out"
    #video_filename = "vis_video_JM_20s.mp4"
    #out_video_filename = "vis_video_JM_20s_sliding_window_predict.mp4"

    # Instanciate dataset
    HAD2D = HumanActionDataset('2D', data2D_dir, data2D_files, classes, is_train=False)
    dataloader = DataLoader(dataset=HAD2D, batch_size=1)
    
    # TODO remove
    # Load model
    model = ActionLSTM(nb_classes=len(classes), input_size=2*17, hidden_size_lstm=256, hidden_size_classifier=128, num_layers=1, device=device)
    model.to(device)
    model.load_state_dict(torch.load("./models_saved/action_lstm_2D.pt"))
    model.eval()

    # Predict
    for data_count, data in enumerate(dataloader):
        video_filename = data2D_files[data_count][:-7]+'.mp4'

        if is_sliding_window:
            windows_probs = []
            window_size = 45
            stride = 15
            data_filled = fill_zeros(data[0])
            windows = sliding_window(data_filled, window_size, stride)
            
            for start, end, window in windows:
                input = window.to(device)
                h_n, c_n = None, None
                output, h_n, c_n = model(input, h_n, c_n)
                h_n, c_n = copy.copy(h_n).to(device), copy.copy(c_n).to(device)
                sm = nn.Softmax(dim=1).to(device)    
                probs_ = sm(output).reshape(len(classes)).detach().cpu().numpy()
                window_probs = np.concatenate([start*np.ones((1,)), end*np.ones((1,)), probs_]).reshape(1,11)
                windows_probs.append(window_probs)
            vid_probs = np.concatenate(windows_probs)
            np.save(os.path.join(prediction_dir, video_filename[:-4]+'.npy'), vid_probs)
            vid_probs_dup = duplicate_array(vid_probs, stride)

        
        # open each frame 
        # apply the prediction on it
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        cap = cv2.VideoCapture(os.path.join(keypoint_video_dir, video_filename))
        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Define the codec and create VideoWriter object.
        # Define the fps to be equal to 24. Also frame size is passed.
        # cv2.VideoWriter_fourcc('M','J','P','G')
        if is_sliding_window:
            suffix = '_sw'
        else:
            suffix = ''
        writer = cv2.VideoWriter(os.path.join(prediction_video_dir, video_filename+suffix), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 24, (frame_width, frame_height))
        

        # Read until video is completed
        frame_count = 1
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:

                # predict
                if is_sliding_window:
                    probs_frame = vid_probs_dup[frame_count-1]
                if not is_sliding_window:
                    x = data[0][:, :frame_count, ...].to(device) # first element, data (and no label)
                    h_n, c_n = None, None
                    output, h_n, c_n = model(x, h_n, c_n)
                    h_n, c_n = copy.copy(h_n).to(device), copy.copy(c_n).to(device)
                    sm = nn.Softmax(dim=1).to(device)
                    probs_frame = sm(output).reshape(len(classes)).detach().cpu().numpy()
                
                # else keep on going and repeat the previous results

                # Display the resulting frame
                
                frame = cv2.putText(
                    img=frame,
                    text=f"true label : Unknown",
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.5,
                    org=(10,25), color=(0,0,0),
                    thickness=1)

                for i in range(len(classes)):
                    p = float(probs_frame[i].round(2))
                    color = (0,0,255)
                    cell_height = 12
                    cells_start = 20
                    frame = cv2.rectangle(img=frame, pt1=(frame_width-5,cells_start+i*cell_height), pt2=(frame_width-int(5+p*245), cells_start+cell_height+i*cell_height), color=color, thickness=-1)
                    frame = cv2.rectangle(img=frame, pt1=(frame_width-5,cells_start+i*cell_height), pt2=(frame_width-250,cells_start+cell_height+i*cell_height), color=(0,0,0), thickness=2)

                    frame = cv2.putText(
                        img=frame,
                        text="{}".format(actions[classes[i]]),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 0.25,
                        org=(frame_width-245, cells_start+int(cell_height*0.75)+i*cell_height), color=(0,0,0),
                        thickness=1)
                
                # Write the frame into the file
                writer.write(frame)
                frame_count += 1
            else:
                break

        # When everything done, release the video capture object and writer object
        cap.release()
        writer.release()

if __name__ == "__main__":
    keypoint_npy_dir = 'data/output/keypoint_npy'
    keypoint_video_dir = 'data/output/keypoint_video'
    prediction_dir = 'data/output/prediction' # contains both sliding window and whole
    prediction_video_dir = 'data/output/prediction_keypoint_video'
    is_sliding_window = True
    predict_on_videos(keypoint_npy_dir, keypoint_video_dir, prediction_dir, prediction_video_dir, is_sliding_window=is_sliding_window)