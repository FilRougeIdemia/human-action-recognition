# Evaluate models

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from model import HumanActionDataset, stratified_split, ActionLSTM, PadSequence
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import MulticlassF1Score
import numpy as np

def evaluate(model, dataloader, device):
    model.eval()

    # Evaluate the model on the validation or test dataset
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data in dataloader:
            inputs = data[0].to(device)
            labels = data[1]
            h_n, c_n = None, None
            outputs, _, _ = model(inputs, h_n, c_n)
            _, preds = torch.max(outputs, 1)
            
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

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

if __name__ == "__main__":
    # need to get the same dataloaders than during training
    # so we don't have leaks
    # setting the device as the GPU if available, else the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: {}".format(device))

    # Instanciate dataset
    HAD2D = HumanActionDataset('2D', is_train=True)
    train_dataset2D, val_dataset2D = stratified_split(dataset=HAD2D, test_size=0.2)
    train_dataloader2D = DataLoader(dataset=train_dataset2D)
    val_dataloader2D = DataLoader(dataset=val_dataset2D)
    # Instanciate model
    model = ActionLSTM(nb_classes=len(HAD2D.classes), input_size=2*17, hidden_size_lstm=256, hidden_size_classifier=128, num_layers=1, device=device)
    model.to(device) 
    model.load_state_dict(torch.load("models_saved/action_lstm_2D_luggage.pt"))
    model.eval()

    evaluate(model, train_dataloader2D, device)