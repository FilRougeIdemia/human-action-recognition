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
from evaluate_training import confusion_matrix, top_k_accuracy

device_index=1
if torch.cuda.is_available():
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    print(f"device cuda:{device_index}")


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
        # To respect the order grab bag, hold bag, leave bag,  put something into a bag, put on bag, take off bag, take something out of a bag
        # the classes are not in ascending order
        self.classes = [6, 7, 8, 9, 15, 25, 31, 33, 43, 121, 122, 123, 89, 87, 88, 90] if classes is None else classes
        if self.is_train:
            self.data_dirs_files = [[data_file for data_file in os.listdir(data_dir) if int(data_file[-7:-4]) in self.classes] for data_dir in self.data_dirs] if data_dirs_files is None else data_dirs_files
        else:
            self.data_dirs_files = [os.listdir(dir) for dir in self.data_dirs]
        self._data_files_path = [os.path.join(data_dir, data_dir_file) for data_dir, data_dir_files in zip(self.data_dirs, self.data_dirs_files) for data_dir_file in data_dir_files] if _data_files_path is None else _data_files_path
        if actions is None:
            with open("data/actions.txt", 'r') as actions_file:
                actions = [line.replace('\n', '') for line in actions_file.readlines()]
                actions_file.close()
        self.actions = actions
        
        self.classes_names = dict()
        for i,elem in enumerate(self.classes):
            self.classes_names.update({i : self.actions[elem-1]})
        print(self.classes_names)

    def __len__(self):
        return len(self._data_files_path)

    def __getitem__(self, idx):
        tensor = torch.Tensor(np.load(self._data_files_path[idx]))
        normalization = 1000 if self.data_dim==2 else 1
        tensor = tensor.reshape((tensor.shape[0], self.data_dim*17))/normalization
        if self.is_train:
            label = self.classes.index(int(os.path.basename(self._data_files_path[idx])[-7:-4]))
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


def train_model(model, criterion, optimizer, nb_epochs, epoch_log_frequence, step_log_frequence_train, train_dataset, val_dataset, train_dataloader, val_dataloader, classes, device, writer):
    sm = nn.Softmax(dim=1).to(device)
    step = 0

    # Initialize step interval metrics
    # later reinitialize after each step interval
    step_log_loss_train, step_log_acc_train = 0, 0
    # Initialize/Reinitialize at each epoch
    epoch_log_loss, epoch_log_acc, epoch_log_obs_cnt = 0, 0, 0 

    for epoch in range(nb_epochs):
        model.train()

        # Iterate over batches
        for data in train_dataloader:
            # get skeleton sequences data per batch
            inputs = data[0].to(device)
            # get labels per batch
            labels = torch.zeros((inputs.shape[0], len(classes)))
            for i in range(inputs.shape[0]):
                labels[i][int(data[-1][i])] = 1
            labels = labels.to(device)

            # reset gradients
            optimizer.zero_grad()
            
            # collect raw predictions before softmax (logits)
            outputs_for_loss,_,_ = model(inputs)
            
            # apply softmax, calculate loss
            outputs = sm(outputs_for_loss).to(device)
            loss = criterion(outputs_for_loss, labels)

            loss.backward()
            optimizer.step()
            step += 1
            # Metrics for log
            # Log to TensorBoard every step_log_frequence for metrics on the train set
            if(step%step_log_frequence_train==0):
                # loss for that batch
                step_log_loss_train = loss.item()
                # accuracy for that batch
                step_log_acc_train = int(torch.sum(outputs.argmax(dim=1) == labels.argmax(dim=1)))/len(inputs)
                writer.add_scalar(f"Loss [Train set][Step interval:{step_log_frequence_train}]", step_log_loss_train, global_step=step)
                writer.add_scalar(f"Accuracy [Train set][Step interval:{step_log_frequence_train}]", step_log_acc_train, global_step=step)
                # reset to 0 to compute over the next step interval
                step_log_loss_train, step_log_acc_train = 0, 0

        # Log to TensorBoard each epoch_log_frequence for metrics on the entire train and validation sets
        if(epoch%epoch_log_frequence==0):
            plot_class_distribution(train_dataset, "Train Dataset Class Distribution With Weighted Sampler", writer, "Train_Dataset_Weighted")
            model.eval()
            with torch.no_grad():
                # Go through the data with dataloaders to take advantage of samplers
                for dataloader in [train_dataloader, val_dataloader]:
                    dataset_type = 'Train' if dataloader==train_dataloader else 'Validation' 
                    all_preds = []
                    all_labels = []
                    all_outputs = []
                    for data in dataloader:
                        inputs = data[0].to(device)
                        labels = torch.zeros((inputs.shape[0], len(classes)))
                        for i in range(inputs.shape[0]):
                            labels[i][int(data[-1][i])] = 1
                        labels = labels.to(device)
                        epoch_log_obs_cnt += 1
                        outputs = model(inputs)[0]
                        preds = torch.argmax(outputs, dim=1)
                        loss = criterion(outputs, labels)
                        epoch_log_loss += loss.item()
                        epoch_log_acc += int(torch.sum(outputs.argmax(dim=1) == labels.argmax(dim=1)))
                        all_preds.extend(preds.tolist())
                        all_labels.extend(data[-1].tolist())
                        all_outputs.extend(outputs.tolist())
                    # Calculate loss 
                    epoch_log_loss /= len(dataloader)
                    writer.add_scalar(f"Loss [{dataset_type} set][Epoch interval:{epoch_log_frequence}]", epoch_log_loss, global_step=step)
                    # Calculate accuracy for the entire val_dataset
                    epoch_log_acc /= epoch_log_obs_cnt
                    writer.add_scalar(f"Accuracy overall [{dataset_type} set][Epoch interval:{epoch_log_frequence}]", epoch_log_acc, global_step=step)
                    # Calculate accuracy per class
                    acc_per_class = dict()
                    for act_idx, act in train_dataset.dataset.classes_names.items():
                        all_labels_bin = (np.array(all_labels) == act_idx)
                        all_preds_bin = (np.array(all_preds) == act_idx)
                        correct = np.sum(all_labels_bin == all_preds_bin)
                        acc_per_class.update({act:correct / len(all_labels_bin)})
                    writer.add_scalars(f"Accuracy per class [{dataset_type} set][Epoch interval:{epoch_log_frequence}]", acc_per_class, global_step=step)
                    # Calculate confusion matrix for the entire val_dataset
                    confusion_mat, f1, precision, recall = confusion_matrix(all_outputs, all_labels, normalize=None)
                    writer.add_scalar(f'F1 overall [{dataset_type} set][Epoch interval:{epoch_log_frequence}]', np.mean(f1), global_step=step)
                    writer.add_scalars(f'F1 per class [{dataset_type} set][Epoch interval:{epoch_log_frequence}]', {act:f1[act_idx] for act_idx, act in train_dataset.dataset.classes_names.items()}, global_step=step)
                    writer.add_scalar(f'Precision overall [{dataset_type} set][Epoch interval:{epoch_log_frequence}]', np.mean(precision), global_step=step)
                    writer.add_scalars(f'Precision per class [{dataset_type} set][Epoch interval:{epoch_log_frequence}]', {act:precision[act_idx] for act_idx, act in train_dataset.dataset.classes_names.items()}, global_step=step)
                    writer.add_scalar(f'Recall overall [{dataset_type} set][Epoch interval:{epoch_log_frequence}]', np.mean(recall), global_step=step)
                    writer.add_scalars(f'Recall per class [{dataset_type} set][Epoch interval:{epoch_log_frequence}]', {act:precision[act_idx] for act_idx, act in train_dataset.dataset.classes_names.items()}, global_step=step)
                    writer.add_figure(f'Confusion [{dataset_type} set][Epoch interval:{epoch_log_frequence}]', confusion_mat, global_step=step)

                    # reset to 0 to compute over the next (dataloader, epoch)
                    epoch_log_loss, epoch_log_acc, epoch_log_obs_cnt = 0, 0, 0


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
    bars = plt.bar(list(counter.keys()), list(counter.values()))

    # Add the exact number per bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')

    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.savefig("models_saved/{}_class_distribution.png".format(tag))
    plt.close()

    # Log class distribution to TensorBoard
    for i, count in counter.items():
        writer.add_scalars("Classes distribution", {str(key): value for key, value in dict(counter).items()}, global_step=0)


def main():
    # setting the device as the GPU if available, else the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: {}".format(device))

    writer = SummaryWriter()

    # Instanciate dataset
    HAD = HumanActionDataset('2D', is_train=True)
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

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=24, collate_fn=PadSequence(), sampler=train_sampler)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=24, collate_fn=PadSequence(), sampler=val_sampler)

    # Instanciate model
    model = ActionLSTM(nb_classes=len(HAD.classes), input_size=HAD.data_dim*17, hidden_size_lstm=256, hidden_size_classifier=128, num_layers=1, device=device)
    model.to(device)
    # Declare training parameters
    criterion = FocalLoss()
    # alternatively : #criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    nb_epochs = 500
    epoch_log_frequence = 1
    step_log_frequence_train = 10
    # Train
    losses_accs_LSTM03D = train_model(model, criterion, optimizer, nb_epochs, epoch_log_frequence, step_log_frequence_train, train_dataset, val_dataset, train_dataloader, val_dataloader, HAD.classes, device, writer)
    torch.save(model.state_dict(), f"models_saved/action_lstm_{HAD.data_type}_luggage_0410.pt")

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
    plt.savefig(f"models_saved/action_lstm_{HAD.data_type}_luggage_0410_loss_acc.png")

if __name__ == "__main__":
    main()