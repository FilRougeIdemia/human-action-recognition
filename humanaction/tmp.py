import torch
from mmaction.models import build_model
from mmaction.apis import init_recognizer
from mmaction.datasets import build_dataset
from mmaction.datasets.pipelines import Compose
from mmcv import Config
import os


def main():
    print(os.getcwd())
    config_file = '/home/infres/jma-21/FilRouge/mmaction2/configs/skeleton/2s-agcn/2sagcn_mmpose_keypoint_3d.py'  # Replace with the path to your model's configuration file
    checkpoint_file = '/home/infres/jma-21/FilRouge/mmaction2/configs/skeleton/2s-agcn/best_top1_acc_epoch_8.pth'  # Replace with the path to your model's checkpoint file

    # Load the config
    cfg = Config.fromfile(config_file)

    # Initialize the model using the config
    model = init_recognizer(config_file, checkpoint_file, device='cuda:0')
    print("coucou")

if __name__ == '__main__':
    main()