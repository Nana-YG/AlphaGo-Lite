import gc
import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import bisect
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np
import random

from Utils import (
    print_blue,
    print_green,
    print_red
)

# Define Dataset for HDF5
class DeviceDataLoader(DataLoader):
    def __init__(self, *args, device, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def __iter__(self):
        for batch in super().__iter__():
            # Move data to device
            yield tuple(item.to(self.device) for item in batch)

class GoDataset(Dataset):
    def __init__(self, h5_file, type):
        self.type = type
        self.h5_file = h5py.File(h5_file, 'r')
        self.group_names = list(self.h5_file.keys())

        # Load start indices for each group
        print(">>> Reading startingIndex from file...")
        self.start_indices = {}
        for group_name in self.group_names:
            group = self.h5_file[group_name]
            if "startingIndex" in group.keys():
                self.start_indices[group_name] = int(group["startingIndex"][()])
            else:
                raise ValueError(f"Group {group_name} does not have a startingIndex attribute.")

        # Sort groups by startingIndex
        print("Sorting group keys...")
        self.group_names = sorted(
            self.group_names, key=lambda name: self.start_indices[name]
        )

        # Preload data into memory if specified
        self.cache = {}
        print("Preloading data into memory... File name =", h5_file)
        idx = 0
        for group_name in self.group_names:
            if idx % 100 == 0:
                print('#', end='', flush=True)
            group = self.h5_file[group_name]
            liberty_inversed = {}
            for key in group:
                if key.startswith("liberty"):
                    # Convert data type
                    liberty_raw = group[key][:].astype(np.float32)
                    # Take inverse to non-zero terms
                    liberty_inversed[key] = np.where(liberty_raw != 0, 1.0 / liberty_raw, 0.0)

            self.cache[group_name] = {
                "boards": {key: group[key][:] for key in group if key.startswith("board")},
                "liberties": liberty_inversed,
                "labels": {key: group[key][:] for key in group if key.startswith("nextMove")},
            }
            idx += 1

    def __len__(self):

        length = 0
        for group in self.group_names:
            group_keys = self.h5_file[group].keys()  # get all keys from current group
            length += len(group_keys)

        print("Length =", length // 3)
        return length

    def __getitem__(self, idx):
        # Use binary search to find the correct group
        start_values = [self.start_indices[name] for name in self.group_names]
        group_idx = bisect.bisect_right(start_values, idx) - 1
        group_name = self.group_names[group_idx]

        # Calculate the local index within the group
        local_idx = idx - self.start_indices[group_name]

        try:
            board = self.cache[group_name]["boards"][f"board_{local_idx}"]
            liberty = self.cache[group_name]["liberties"][f"liberty_{local_idx}"]
            label = self.cache[group_name]["labels"][f"nextMove_{local_idx}"]
        except Exception as e:
            # Use default values in case of error
            board = np.zeros((19, 19), dtype=np.float32)
            liberty = np.zeros((19, 19), dtype=np.float32)
            label = np.array([3, 3], dtype=np.int64)

        if board.shape != (19, 19):
            raise ValueError(f"Invalid board shape: {board.shape}")
        if liberty.shape != (19, 19):
            raise ValueError(f"Invalid liberty shape: {liberty.shape}")

        # Combine board and liberty into a single NumPy array and then convert to Tensor
        ones = np.ones((19, 19), dtype=np.float32)
        zeros = np.zeros((19, 19), dtype=np.float32)

        # Enhance data
        board, liberty, label = random_augment(board, liberty, label, board_size=19)

        combined_array = np.stack(
            [board,
             liberty,
             ones,
             zeros],
            axis=0)  # Shape: (4, 19, 19)
        input_tensor = torch.from_numpy(combined_array).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return input_tensor, label_tensor


# Define the neural network
class GoNet(nn.Module):
    def __init__(self):
        super(GoNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.hidden_convs = nn.Sequential(
            *[nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1) for _ in range(20)]
        )
        self.output_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.conv1(x)
        for conv in self.hidden_convs:
            x = self.relu(conv(x))
        x = self.output_conv(x)
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 361]
        # x = self.softmax(x)  # Apply softmax to get probabilities
        return x

def initialize_weights(model, seed=42):
    torch.manual_seed(seed)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            init.ones_(layer.weight)
            init.zeros_(layer.bias)


def calculate_accuracy(model, test_data, batch_size):
    model.eval()
    correct = 0
    total = 0

    inputs, labels = test_data
    with torch.no_grad():
        for start_idx in range(0, inputs.size(0), batch_size):
            end_idx = min(start_idx + batch_size, inputs.size(0))
            input_batch = inputs[start_idx:end_idx]
            label_batch = labels[start_idx:end_idx]

            outputs = model(input_batch)
            outputs = F.log_softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            target_indices = label_batch[:, 0] * 19 + label_batch[:, 1]

            correct += (predicted == target_indices).sum().item()
            total += label_batch.size(0)

    accuracy = correct / total
    return accuracy


# Training function
def train_one_epoch(model,
                    test_data,
                    criterion,
                    optimizer,
                    epoch,
                    device,
                    plot_loss,
                    train_accuracy_history,
                    test_accuracy_history,
                    loss_history):
    global max_accuracy
    max_accuracy = 0.0

    folder_path = "Dataset/train/"
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

    accuracy = calculate_accuracy(model, test_data, batch_size=512)
    print_green(f"Initial accuracy: {accuracy:.10f}")
    test_accuracy_history.append(accuracy)
    train_accuracy_history.append(0)
    save_accuracy_plot(train_accuracy_history, test_accuracy_history)

    file_count = 0


    for h5_file in h5_files:
        print_blue(">>> Train: Loading from file: " + str(h5_file))
        train_file_path = "Dataset/train/" + h5_file
        train_dataset = GoDataset(train_file_path, 'train')
        train_loader = DeviceDataLoader(train_dataset,
                                        batch_size=128,
                                        shuffle=True,
                                        device=device,
                                        num_workers=6,
                                        pin_memory=True)

        print_blue(">>> Train: Data loaded, training on " + str(device) + "...")
        file_accuracy, file_losses = train_one_file(model, train_loader, criterion, optimizer)
        train_accuracy_history.append(file_accuracy)
        if plot_loss:
            loss_history.extend(file_losses)
            save_loss_plot(loss_history)

        print_blue(">>> Train: Evaluating on test set...")
        accuracy = calculate_accuracy(model, test_data, batch_size=512)
        print_green(f"Test Accuracy after file {file_count + 1}: {accuracy:.10f}")
        if accuracy > max_accuracy:
            save_checkpoint(model, optimizer, epoch)
            max_accuracy = accuracy

        test_accuracy_history.append(accuracy)
        save_accuracy_plot(train_accuracy_history, test_accuracy_history)
        print_blue(">>> Train: Deleting dataset...")
        del train_dataset
        del train_loader
        gc.collect()
        file_count += 1

    print_green(f"Test Accuracy after Epoch {epoch + 1}: {accuracy:.10f}")


def train_one_file(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    interval_loss = 0.0
    interval_losses = []

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        target_indices = labels[:, 0] * 19 + labels[:, 1]
        loss = criterion(outputs, target_indices)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        interval_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        total_correct += (predicted == target_indices).sum().item()
        total_samples += labels.size(0)

        if batch_idx % 1000 == 999:
            interval_loss = interval_loss / 1000
            interval_losses.append(interval_loss)
            interval_loss = 0.0
            print("\033[94m#\033[00m", end='', flush=True)

    avg_loss = total_loss / (batch_idx + 1)
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    print_green("Average loss: {:.10f}".format(avg_loss))
    return avg_acc, interval_losses

def save_loss_plot(loss_history):
    if len(loss_history) > 0:
        plt.figure(figsize=(20, 8))
        indices = range(1, len(loss_history) + 1)
        plt.plot(indices, loss_history, label='Interval Avg Loss')
        plt.xlabel('Batch Index')
        plt.ylabel('Loss')
        plt.title('Average Loss per 1000 Batches')
        plt.legend()
        plt.grid()
        plt.savefig('interval_losses.png')
        plt.close()


def save_accuracy_plot(train_accuracy, test_accuracy, filename="accuracy_plot.png"):
    plt.figure(figsize=(20, 8))
    epochs = range(0, len(train_accuracy))

    plt.plot(epochs, train_accuracy, label="Train Accuracy", linestyle="-")
    plt.plot(epochs, test_accuracy, label="Test Accuracy", linestyle="-")
    plt.title("Train and Test Accuracy Over Epochs")
    plt.xlabel("Evaluation Step (every 100 epochs)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()
    print(f"Accuracy plot saved as {filename}")

def save_checkpoint(model, optimizer, epoch, checkpoint_path="checkpoint.pth"):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path="checkpoint.pth"):
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = state["epoch"] + 1
        print(f"Checkpoint loaded from {checkpoint_path}, starting at epoch {start_epoch}")
        return start_epoch
    else:
        print("No checkpoint found, starting from scratch.")
        return 0

def random_augment(board, liberty, label, board_size=19):

    k = np.random.choice([0, 1, 2, 3])

    aug_board = np.rot90(board, k=k)
    aug_liberty = np.rot90(liberty, k=k)
    aug_label = rotate_label(label, k, board_size)
    flip_type = np.random.choice(["none", "horizontal", "vertical"])

    if flip_type == "horizontal":
        aug_board = np.fliplr(aug_board)
        aug_liberty = np.fliplr(aug_liberty)
        aug_label = flip_label(aug_label, flip_type, board_size)

    elif flip_type == "vertical":
        aug_board = np.flipud(aug_board)
        aug_liberty = np.flipud(aug_liberty)
        aug_label = flip_label(aug_label, flip_type, board_size)

    return aug_board, aug_liberty, aug_label


def rotate_label(label, k, board_size=19):

    row, col = label
    if k == 0:
        return np.array([row, col], dtype=np.int64)
    elif k == 1:
        return np.array([col, board_size - 1 - row], dtype=np.int64)
    elif k == 2:
        return np.array([board_size - 1 - row, board_size - 1 - col], dtype=np.int64)
    elif k == 3:
        return np.array([board_size - 1 - col, row], dtype=np.int64)

def flip_label(label, flip_type, board_size=19):

    row, col = label
    if flip_type == "horizontal":
        return np.array([row, board_size - 1 - col], dtype=np.int64)
    elif flip_type == "vertical":
        return np.array([board_size - 1 - row, col], dtype=np.int64)
    else:
        return label



if __name__ == "__main__":
    # Detect device availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print_red("GPU not available. Using CPU instead.")

    print_blue(f">>> Using device: {device}")

    print_blue(">>> Main: Loading dataset objects...")
    test_dataset = GoDataset('Dataset/test/test_small.h5', 'test')
    test_loader = DeviceDataLoader(test_dataset,
                                   batch_size=512,
                                   shuffle=False,
                                   device=device,
                                   num_workers=24,
                                   pin_memory=True)

    print_blue("\n>>> Preloading test data into GPU...")
    test_inputs, test_labels = [], []
    for inputs, labels in test_loader:
        test_inputs.append(inputs.to(device))
        test_labels.append(labels.to(device))

    test_inputs = torch.cat(test_inputs, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    test_data = (test_inputs, test_labels)
    print_blue("\n>>> Test data preloaded.")

    print_blue(">>> Main: Initializing model...")
    model = GoNet()
    model.to(device)
    model.apply(initialize_weights)

    criterion = nn.CrossEntropyLoss()

    print_blue(">>> Main: Start training...")
    max_epoch = 10
    optimizers = []
    optimizers.append(torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4))
    optimizers.append(torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5))

    schedulers = []
    schedulers.append(StepLR(optimizers[0], step_size=2, gamma=0.95))
    schedulers.append(StepLR(optimizers[1], step_size=1, gamma=0.95))

    train_accuracy_history = []
    test_accuracy_history = []
    loss_history = []

    for epoch in range(max_epoch):
        if epoch == 0:
            train_one_epoch(model,
                            test_data,
                            criterion,
                            optimizers[0],
                            epoch,
                            device=device,
                            plot_loss=True,
                            train_accuracy_history = train_accuracy_history,
                            test_accuracy_history = test_accuracy_history,
                            loss_history = loss_history)
        else:
            train_one_epoch(model,
                            test_data,
                            criterion,
                            optimizers[1],
                            epoch,
                            device=device,
                            plot_loss=False,
                            train_accuracy_history = train_accuracy_history,
                            test_accuracy_history = test_accuracy_history,
                            loss_history = loss_history)
            schedulers[1].step()

    print_blue(">>> Main: Training finished. Saving model...")
    torch.save(model.state_dict(), 'final_net.pth')
    print_blue(">>> Main: Model saved.")
