# Evaluate models

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from model import HumanActionDataset, stratified_split, ActionLSTM, PadSequence
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score
import numpy as np
from evaluate_training import compute_metrics 
from model import calculate_sample_weights, WeightedRandomSampler
import matplotlib.pyplot as plt
import os

def evaluate(model, dataloader, device):
    model.eval()

    # Evaluate the model on the validation or test dataset
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            inputs = data[0].to(device)
            labels = data[1]
            h_n, c_n = None, None
            outputs, _, _ = model(inputs, h_n, c_n)
            _, preds = torch.max(outputs, 1)

            all_outputs.extend(outputs.tolist())
            all_labels.extend(data[-1].tolist())

    cm, f1, precision, recall, accuracy, mean_results, cm_fig = compute_metrics(all_outputs, all_labels)


if __name__ == "__main__":
    # need to get the same dataloaders than during training
    # so we don't have leaks
    # setting the device as the GPU if available, else the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: {}".format(device))

    # Instanciate dataset
    HAD = HumanActionDataset('2D', is_train=True)
    train_dataset, val_dataset = stratified_split(dataset=HAD, test_size=0.2)
    
    # Calculate sample weights
    train_sample_weights = calculate_sample_weights(train_dataset)
    val_sample_weights = calculate_sample_weights(val_dataset)

    # Create WeightedRandomSamplers
    train_sampler = WeightedRandomSampler(train_sample_weights, len(train_dataset))
    val_sampler = WeightedRandomSampler(val_sample_weights, len(val_dataset))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=24, collate_fn=PadSequence(), sampler=train_sampler)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=24, collate_fn=PadSequence(), sampler=val_sampler)

    # Instanciate model
    model = ActionLSTM(nb_classes=len(HAD.classes), input_size=HAD.data_dim*17, hidden_size_lstm=256, hidden_size_classifier=128, num_layers=1, device=device)
    model.to(device) 
    model.load_state_dict(torch.load("models_saved/action_lstm_2D_luggage_0410.pt", map_location=torch.device('cuda:0')))
    model.eval()

    evaluate(model, val_dataloader, device)