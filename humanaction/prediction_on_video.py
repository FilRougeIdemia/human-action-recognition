from model import ActionLSTM, classes, device, HumanActionDataset, PadSequence, actions
import torch
import copy
import torch.nn as nn
import cv2
import os
import matplotlib.pyplot as plt
import ffmpeg

# path to skeleton npy folder
data2D_dir = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\data\\input_test\\"
data2D_files = ["S001C001P001R001A006.npy"]
# path to video folder
out_path = "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\NTU_samples_and_inference_code\\out"
video_filename = "vis_video_JM_20s.mp4"
out_video_filename = "vis_video_JM_20s_predict.mp4"

def main():
    # take a skeleton file name and predict for each frame.
    # look up the corresponding video
    # add prediction information
    # output new video

    # for labeled data
    HAD2D = HumanActionDataset('2D', data2D_dir, data2D_files, classes)
    predict_dataloader2D = torch.utils.data.DataLoader(dataset=HAD2D)

    # load model
    model = ActionLSTM(nb_classes=len(classes), input_size=2*17, hidden_size_lstm=256, hidden_size_classifier=128, num_layers=1, device=device)
    model.to(device)
    model.load_state_dict(torch.load("C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\models_saved\\action_lstm_2D.pt"))
    model.eval()

    # open each frame 
    # apply the prediction on it
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(os.path.join(out_path, video_filename))
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Define the codec and create VideoWriter object.
    # Define the fps to be equal to 24. Also frame size is passed.
    # cv2.VideoWriter_fourcc('M','J','P','G')
    writer = cv2.VideoWriter(os.path.join(out_path, out_video_filename), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 24, (frame_width, frame_height))
    

    # Read until video is completed
    frame_count = 1
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # predict
            for data in predict_dataloader2D: 
                x = data[0][:, :frame_count, ...].to(device) # first element, data (and no label)
                label_2D = data[1]
            if x.nelement() > 0: # if there are keypoints on this frame
                h_n, c_n = None, None
                output, h_n, c_n = model(x, h_n, c_n)
                h_n, c_n = copy.copy(h_n).to(device), copy.copy(c_n).to(device)
                sm = nn.Softmax(dim=1).to(device)
                probs = sm(output).reshape(len(classes)).detach().cpu().numpy()
            # else keep on going and repeat the previous results

            # Display the resulting frame
            
            frame = cv2.putText(
                img=frame,
                text=f"true label : {actions[classes[label_2D]]}",
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.5,
                org=(10,25), color=(0,0,0),
                thickness=1)

            for i in range(len(classes)):
                p = float(probs[i].round(2))
                color = (0,0,255)
                if i == label_2D:
                    color = (0,255,0)
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
    main()