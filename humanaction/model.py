# Model
# Declare a LSTM and train it on NTU skeleton action data
import os
import os.path as osp
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.ops import sigmoid_focal_loss
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
# local import
from focal_loss import FocalLoss

writer = SummaryWriter()

class HumanActionDataset(Dataset):

    """
    inputs:
    - (str) data_type: type of the dataset to build either '2D' or '3D'.
    - (str) data_dir: path to the data folder of the data (either 2D or 3D) to consider.
    - (list) data_files: name of the files in the folder to consider.
    - (list) classes: id of the classes to consider.
    """

    def __init__(self, data_type:str='2D', data_dirs:list=None, data_dirs_files:list=None, _data_files_path:list=None, classes:list=None, actions:list=None, is_train:bool=False):
        self.data_type = data_type
        self.data_dim = int(self.data_type[0])
        self.data_dirs = [osp.join("data/input/mmpose_ntu", self.data_type), osp.join("data/input/acquisition_sacs", self.data_type)] if data_dirs is None else data_dirs
        self.is_train = is_train
        self.classes = [5, 6, 7, 8, 14, 24, 30, 32, 42, 61, 62, 63, 64, 65, 66, 67] if classes is None else classes
        if self.is_train:
            self.data_dirs_files = [[data_file for data_file in os.listdir(data_dir) if int(data_file[-7:-4])-1 in self.classes] for data_dir in self.data_dirs] if data_dirs_files is None else data_dirs_files
        else:
            self.data_dirs_files = [os.listdir(dir) for dir in self.data_dirs]
        self._data_files_path = [os.path.join(data_dir, data_dir_file) for data_dir, data_dir_files in zip(self.data_dirs, self.data_dirs_files) for data_dir_file in data_dir_files] if _data_files_path is None else _data_files_path
        if actions is None:
            with open("data/actions.txt", 'r') as actions_file:
                actions = [line.replace('\n', '') for line in actions_file.readlines()]
                actions_file.close()
        self.actions = actions
        
        for i,elem in enumerate(self.classes):
            print("class {} : {}".format(i, self.actions[elem - 1]))

    def __len__(self):
        return len(self._data_files_path)

    def __getitem__(self, idx):
        tensor = torch.Tensor(np.load(self._data_files_path[idx]))
        normalization = 1000 if self.data_dim==2 else 1
        tensor = tensor.reshape((tensor.shape[0], self.data_dim*17))/normalization
        if self.is_train:
            label = self.classes.index(int(os.path.basename(self._data_files_path[idx])[-7:-4])-1)
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


def train_model(model, criterion, optimizer, nb_epochs, epoch_print_frequence, train_dataset, val_dataset, train_dataloader, val_dataloader, classes, device):
    s = time.time()
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    sm = nn.Softmax(dim=1).to(device)
    step = 0

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
                step += 1

        running_loss_train /= len(train_dataloader)
        running_loss_val /= len(val_dataloader)
        running_acc_train /= len(train_dataset)
        running_acc_val /= len(val_dataset)

        train_losses.append(running_loss_train)
        val_losses.append(running_loss_val)
        train_accs.append(running_acc_train)
        val_accs.append(running_acc_val)

        # Log to TensorBoard
        writer.add_scalar("Loss train", running_loss_train, global_step=step)
        writer.add_scalar("Loss val", running_loss_val, global_step=step)
        writer.add_scalar("Accuracy train", running_acc_train, global_step=step)
        writer.add_scalar("Accuracy val", running_acc_val, global_step=step)


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


def stratified_split(dataset: HumanActionDataset, test_size: float):
    # Extract the labels from the dataset
    targets = [label for _, label in dataset]

    # Perform stratified splitting using the targets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(sss.split(dataset._data_files_path, targets))

    # Create train and test datasets using the obtained indices
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    return train_dataset, test_dataset


def calculate_sample_weights(dataset):
    labels = [label for _, label in dataset]
    counter = Counter(labels)
    total_samples = len(dataset)
    class_weights = {cls: total_samples / count for cls, count in counter.items()}
    sample_weights = [class_weights[label] for label in labels]
    return sample_weights


def plot_class_distribution(dataset, title, writer, tag):
    labels = [label for _, label in dataset]
    counter = Counter(labels)

    plt.figure()
    plt.bar(list(counter.keys()), list(counter.values()))
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.savefig("models_saved/{}_class_distribution.png".format(tag))
    plt.close()

    # Log class distribution to TensorBoard
    for i, count in counter.items():
        writer.add_scalars("Classes distribution", {str(key): value for key, value in dict(counter).items()})


def main():
    # setting the device as the GPU if available, else the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: {}".format(device))

    # Instanciate dataset
    HAD = HumanActionDataset('3D', is_train=True)
    train_dataset, val_dataset = stratified_split(dataset=HAD, test_size=0.2)

    # Distribution of classes within train and val datasets
    plot_class_distribution(train_dataset, "Train Dataset Class Distribution", writer, "Train_Dataset")
    plot_class_distribution(val_dataset, "Val Dataset Class Distribution", writer, "Val_Dataset")

    # Calculate sample weights
    train_sample_weights = calculate_sample_weights(train_dataset)
    val_sample_weights = calculate_sample_weights(val_dataset)

    # Create WeightedRandomSamplers
    train_sampler = WeightedRandomSampler(train_sample_weights, len(train_dataset))
    val_sampler = WeightedRandomSampler(val_sample_weights, len(val_dataset))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, collate_fn=PadSequence(), sampler=train_sampler)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, collate_fn=PadSequence(), sampler=val_sampler)

    # Instanciate model
    model = ActionLSTM(nb_classes=len(HAD.classes), input_size=HAD.data_dim*17, hidden_size_lstm=256, hidden_size_classifier=128, num_layers=1, device=device)
    model.to(device)
    # Declare training parameters
    criterion = FocalLoss()
    # alternatively : #criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    nb_epochs = 200
    epoch_print_frequence = 20
    # Train
    losses_accs_LSTM03D = train_model(model, criterion, optimizer, nb_epochs, epoch_print_frequence, train_dataset, val_dataset, train_dataloader, val_dataloader, HAD.classes, device)
    torch.save(model.state_dict(), f"models_saved/action_lstm_{HAD.data_type}_luggage_0329.pt")

    # Graphiques de train
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

    ax[0].set(title="Action LSTM - Loss evolution")
    ax[0].plot(losses_accs_LSTM03D[0], label="train")
    ax[0].plot(losses_accs_LSTM03D[1], label="test")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")

    ax[1].set(title="Action LSTM - Accuracy evolution")
    ax[1].plot(losses_accs_LSTM03D[2], label="train")
    ax[1].plot(losses_accs_LSTM03D[3], label="test")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")

    plt.legend()
    plt.savefig("models_saved/action_lstm_{HAD.data_type}_luggage_0329_loss_acc.png")

if __name__ == "__main__":
    main()