# Model
# Declare a LSTM and train it on NTU skeleton action data
import os
import torch
from torch.utils.data import Dataset, DataLoader
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class HumanActionDataset(Dataset):

    """
    inputs:
    - (str) data_type: type of the dataset to build either '2D' or '3D'.
    - (str) data_dir: path to the data folder of the data (either 2D or 3D) to consider.
    - (list) data_files: name of the files in the folder to consider.
    - (list) classes: id of the classes to consider.
    """

    # constant that can become arguments
    data2D_dir = "data/input/mmpose_ntu/"
    data2D_files = os.listdir(data2D_dir)
    with open("C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\data\\actions.txt", 'r') as actions_file:
        actions = [line.replace('\n', '') for line in actions_file.readlines()]
        actions_file.close()
    classes = [5, 6, 7, 8, 14, 24, 30, 32, 42]
    for i,elem in enumerate(classes):
        print("class {} : {}".format(i, actions[elem]))

    def __init__(self, data_type:str='2D', data_dir:str=data2D_dir, data_files:list=data2D_files, classes:list=classes, is_train:bool=False):
        self.data_type = data_type
        self.data_dir = data_dir
        self.is_train = is_train
        if self.is_train:
            self.data_files = [data_file for data_file in data_files if int(data_file[17:-4])-1 in classes]
        else:
            self.data_files = os.listdir(data_dir)
        self.classes = classes
        

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        tensor = torch.Tensor(np.load(os.path.join(self.data_dir, self.data_files[idx])))
        tensor = tensor.reshape((tensor.shape[0], 2*17))/1000
        if self.is_train:
            label = self.classes.index(int(self.data_files[idx][17:-4])-1)
        else:
            label = 999 # TODO change that
        return (tensor, label)


class PadSequence():

    def __call__(self, batch):

        # let's assume that each element in "batch" is a tuple (data, label).
        # the following line of code sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        
        # then we take each sequence of the batch and pad it
        sequences = [x[0] for x in sorted_batch]
        sequences_padded_end = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

        lengths = torch.LongTensor([len(x) for x in sequences])

        # here we adjust the padding because we want zeros at the beginning
        # (we had poor results with zeros at the end)
        sequences_padded_begin = torch.stack(
            [torch.cat([
                sequences_padded_end[i][lengths[i]:],
                sequences_padded_end[i][:lengths[i]]]
            ) for i in range(len(sequences_padded_end))]
        )

        # don't forget to grab the labels of the *sorted* batch
        labels = torch.LongTensor([x[1] for x in sorted_batch])
        return sequences_padded_begin, lengths, labels


def train_model(model, criterion, optimizer, nb_epochs, epoch_print_frequence, train_dataset, val_dataset, train_dataloader, val_dataloader):

    s = time.time()
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    sm = nn.Softmax(dim=1).to(device)

    for epoch in range(nb_epochs):
        
        running_loss_train, running_loss_val, running_acc_train, running_acc_val = 0, 0, 0, 0

        for train in [True, False]:

            if train:
                dataloader = train_dataloader
                model.train()
            
            else:
                dataloader = val_dataloader
                model.eval()

            for data in dataloader:
                
                inputs = data[0].to(device)
                labels = torch.zeros((inputs.shape[0], len(classes)))
                for i in range(inputs.shape[0]):
                    labels[i][int(data[-1][i])] = 1
                labels = labels.to(device)

                if train:
                    optimizer.zero_grad()

                outputs_for_loss,_,_ = model(inputs)
                
                outputs = sm(outputs_for_loss).to(device)
                loss = criterion(outputs_for_loss, labels)

                if train:
                    loss.backward()
                    optimizer.step()
                    running_loss_train += loss.item()
                    running_acc_train += int(torch.sum(outputs.argmax(dim=1) == labels.argmax(dim=1)))
                else:
                    running_loss_val += loss.item()
                    running_acc_val += int(torch.sum(outputs.argmax(dim=1) == labels.argmax(dim=1)))

        running_loss_train /= len(train_dataloader)
        running_loss_val /= len(val_dataloader)
        running_acc_train /= len(train_dataset)
        running_acc_val /= len(val_dataset)

        train_losses.append(running_loss_train)
        val_losses.append(running_loss_val)
        train_accs.append(running_acc_train)
        val_accs.append(running_acc_val)

        if (epoch == 0) or ((epoch+1) % epoch_print_frequence == 0):
            print("epochs {} ({} s) | train loss : {} | val loss : {} | train acc : {} | val acc : {}".format(
                epoch+1,
                int(time.time()-s),
                running_loss_train,
                running_loss_val,
                running_acc_train,
                running_acc_val
            ))
    
    return train_losses, val_losses, train_accs, val_accs


class ActionLSTM(nn.Module):

    def __init__(self, nb_classes, input_size, hidden_size_lstm, hidden_size_classifier, num_layers, device):

        super(ActionLSTM, self).__init__()

        self.num_classes = nb_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size_lstm = hidden_size_lstm
        self.device = device

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_lstm,
            num_layers=num_layers,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size_lstm, hidden_size_classifier),
            nn.ReLU(),
            nn.Linear(hidden_size_classifier, nb_classes)
        )

    def forward(self,x,h_0=None,c_0=None):
        if h_0 is None:
            h_0 = torch.rand(self.num_layers, x.size(0), self.hidden_size_lstm).to(self.device) #   hidden state (short memory)
            c_0 = torch.rand(self.num_layers, x.size(0), self.hidden_size_lstm).to(self.device) # internal state (long memory)
        _, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_n_reshape = h_n[-1].reshape(1, h_n.shape[1], h_n.shape[2]).view(-1, self.hidden_size_lstm)
        results = self.classifier(h_n_reshape) # reshaping the data for clasifier
        return results, h_n, c_n


def main():
    # setting the device as the GPU if available, else the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: {}".format(device))

    # Instanciate dataset
    HAD2D = HumanActionDataset('2D', data2D_dir, data2D_files, classes)
    train_dataset2D, val_dataset2D = torch.utils.data.random_split(HAD2D, [int(0.85*len(HAD2D)), len(HAD2D)-int(0.85*len(HAD2D))])
    train_dataloader2D = torch.utils.data.DataLoader(dataset=train_dataset2D, batch_size=32, collate_fn=PadSequence(), shuffle=True)
    val_dataloader2D = torch.utils.data.DataLoader(dataset=val_dataset2D, batch_size=32, collate_fn=PadSequence(), shuffle=True)
    # Instanciate model
    model = ActionLSTM(nb_classes=len(classes), input_size=2*17, hidden_size_lstm=256, hidden_size_classifier=128, num_layers=1, device=device)
    model.to(device)
    # Declare training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    nb_epochs = 200
    epoch_print_frequence = 20
    # Train
    losses_accs_LSTM03D = train_model(model, criterion, optimizer, nb_epochs, epoch_print_frequence, train_dataset2D, val_dataset2D, train_dataloader2D, val_dataloader2D)
    torch.save(model.state_dict(), "C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\models_saved\\action_lstm_2D.pt")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

    ax[0].set(title="Action LSTM (2D) - Loss evolution")
    ax[0].plot(losses_accs_LSTM03D[0], label="train")
    ax[0].plot(losses_accs_LSTM03D[1], label="test")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")

    ax[1].set(title="Action LSTM (2D) - Accuracy evolution")
    ax[1].plot(losses_accs_LSTM03D[2], label="train")
    ax[1].plot(losses_accs_LSTM03D[3], label="test")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")

    plt.legend()
    plt.savefig("C:\\Users\\Shadow\\Documents\\Projets\\MastereIA\\Idemia\\human-action-recognition\\models_saved\\action_lstm_2D_loss_acc.png")

if __name__ == "__main__":
    main()