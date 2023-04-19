# Evaluate models

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from model import HumanActionDataset, stratified_split, ActionLSTM, PadSequence
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import MulticlassF1Score
import numpy as np
import time

def evaluate(model, dataloader, device):
    model.eval()

    # Evaluate the model on the validation or test dataset
    true_labels = []
    predicted_labels = []
    
    start_time = time.time()

    with torch.no_grad():
        for data in dataloader:
            inputs = data[0].to(device)
            labels = data[1]
            h_n, c_n = None, None
            outputs, _, _ = model(inputs, h_n, c_n)
            _, preds = torch.max(outputs, 1)
            
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())
            
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Compute the confusion matrix
    conf_mat = confusion_matrix(true_labels, predicted_labels)
    print("----------------------------------------------------------------------------------")
    print("Confusion matrix :")
    print(conf_mat)
    
    # Per class accuracy
    # extracted from the confusion matrix
    print("----------------------------------------------------------------------------------")
    print("Per class accuracy :")
    print(np.diag(conf_mat)/conf_mat.sum(1))
    
    # Use preds, target tensors for torch metrics
    # F1 related, micro and macro
    preds = torch.tensor(predicted_labels)
    target = torch.tensor(true_labels)
    mcf1s_micro = MulticlassF1Score(num_classes=model.classifier[2].out_features, average='micro')
    print("----------------------------------------------------------------------------------")
    print("F1-micro :")
    print(mcf1s_micro(preds, target))
    print("----------------------------------------------------------------------------------")
    print("F1-macro :")
    mcf1s_macro = MulticlassF1Score(num_classes=model.classifier[2].out_features, average='macro')
    print(mcf1s_macro(preds, target))
    print("----------------------------------------------------------------------------------")
    print("Per class F1 :")
    mcf1s_pc = MulticlassF1Score(num_classes=model.classifier[2].out_features, average=None)
    print(mcf1s_pc(preds, target))
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))

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

    #evaluate(model, train_dataloader, device)
    evaluate(model, val_dataloader, device)
    
    