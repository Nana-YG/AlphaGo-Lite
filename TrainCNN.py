import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import bisect
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init
import matplotlib.pyplot as plt
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
            self.cache[group_name] = {
                "boards": {key: group[key][:] for key in group if key.startswith("board")},
                "liberties": {key: group[key][:] for key in group if key.startswith("liberty")},
                "labels": {key: group[key][:] for key in group if key.startswith("nextMove")},
            }
            idx += 1

    def __len__(self):
        print("\nCalculating length of dataset...")
        if self.type == "test":
            return 209083

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

        # 第一层卷积：输入通道 4，输出通道 192，卷积核大小 5x5，步幅 1，填充 2（保持尺寸）
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=192, kernel_size=5, stride=1, padding=2)
        # 后续 11 层卷积：输入通道 192，输出通道 192，卷积核大小 3x3，步幅 1，填充 1（保持尺寸）
        self.hidden_convs = nn.Sequential(
            *[nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1) for _ in range(11)]
        )
        # 输出层：1x1 卷积，输入通道 192，输出通道 1（对应棋盘每个位置的概率）
        self.output_conv = nn.Conv2d(in_channels=192, out_channels=1, kernel_size=1, stride=1, padding=0)
        # 激活函数
        self.relu = nn.ReLU()
        # Softmax 层：在 2D 平面上按每个位置归一化概率
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 第一层卷积 + ReLU
        x = self.relu(self.conv1(x))
        # 11 层隐藏卷积 + ReLU
        for conv in self.hidden_convs:
            x = self.relu(conv(x))
        # 最后一层卷积
        x = self.output_conv(x)
        # 输出概率分布（flatten 为 [batch_size, 361]，然后应用 softmax）
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 361]
        x = self.softmax(x)  # Apply softmax to get probabilities
        return x

def initialize_weights(model, seed=42):
    torch.manual_seed(seed)
    if isinstance(model, nn.Conv2d):
        init.xavier_uniform_(model.weight)
        if model.bias is not None:
            init.zeros_(model.bias)
    elif isinstance(model, nn.Linear):
        init.kaiming_uniform_(model.weight, nonlinearity='relu')
        if model.bias is not None:
            init.zeros_(model.bias)

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
            _, predicted = torch.max(outputs, 1)
            target_indices = label_batch[:, 0] * 19 + label_batch[:, 1]

            correct += (predicted == target_indices).sum().item()
            total += label_batch.size(0)

    accuracy = correct / total
    return accuracy


# Training function
def train(model, test_data, criterion, optimizer, max_epoch, device):
    global last_accuracy
    last_accuracy = 0.0

    folder_path = "Dataset/"
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]

    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

    train_accuracy_history = []
    test_accuracy_history = []

    for epoch in range(max_epoch):
        for h5_file in h5_files:
            # 将多参数合并为单字符串
            print_blue(">>> Train: Loading from file: " + str(h5_file))
            train_file_path = "Dataset/" + h5_file
            train_dataset = GoDataset(train_file_path, 'train')
            train_loader = DeviceDataLoader(train_dataset,
                                            batch_size=512,
                                            shuffle=True,
                                            device=device,
                                            num_workers=25,
                                            pin_memory=True)

            print_blue(">>> Train: Data loaded, training on " + str(device) + "...")
            train_accuracy_history.append(
                train_one_file(model, train_loader, criterion, optimizer, device)
            )

            print_blue(">>> Train: Evaluating on test set...")
            accuracy = calculate_accuracy(model, test_data, batch_size=512)
            if accuracy > last_accuracy:
                save_checkpoint(model, optimizer, epoch)
            last_accuracy = accuracy

            test_accuracy_history.append(accuracy)
            save_accuracy_plot(train_accuracy_history, test_accuracy_history)
            print_blue(f"Test Accuracy after file {epoch + 1}: {accuracy:.10f}")

        print_blue(f"Test Accuracy after Epoch {epoch + 1}: {accuracy:.10f}")
        scheduler.step()


def train_one_file(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        target_indices = labels[:, 0] * 19 + labels[:, 1]
        loss = criterion(outputs, target_indices)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        total_correct += (predicted == target_indices).sum().item()
        total_samples += labels.size(0)

        if batch_idx % 200 == 0:
            print("\033[94m#\033[00m", end='', flush=True)

    avg_loss = total_loss / (batch_idx + 1)
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    print_blue("Average loss: {:.10f}".format(avg_loss))
    return avg_acc


def save_accuracy_plot(train_accuracy, test_accuracy, filename="accuracy_plot.png"):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_accuracy) + 1)

    plt.plot(epochs, train_accuracy, label="Train Accuracy", linestyle="-")
    plt.plot(epochs, test_accuracy, label="Test Accuracy", linestyle="-")
    plt.title("Train and Test Accuracy Over Epochs")
    plt.xlabel("Evaluation Step (every 100 epochs)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
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

    print_green(">>> File exist = " + str(os.path.exists('Dataset/board-move-pairs-train.h5')))

    print_blue(">>> Main: Loading dataset objects...")
    test_dataset = GoDataset('Dataset/test_0000.h5', 'test')
    test_loader = DeviceDataLoader(test_dataset,
                                   batch_size=512,
                                   shuffle=False,
                                   device=device,
                                   num_workers=25,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)

    print_blue(">>> Main: Start training...")
    max_epoch = 100
    train(model, test_data, criterion, optimizer, max_epoch, device=device)

    print_blue(">>> Main: Training finished. Saving model...")
    torch.save(model.state_dict(), 'go_model.pth')
    print_blue(">>> Main: Model saved.")
