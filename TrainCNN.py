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
import os


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
        print("Reading startingIndex from file...")
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

    def __len__(self):
        print("Calculating length of dataset...")

        if self.type == "train":
            return 106047633
        if self.type == "test":
            return 209083

        length = 0
        for group in self.group_names:
            group_keys = self.h5_file[group].keys()  # get all keys from current group
            # print("Group keys:", group_keys)
            length += len(group_keys)

        print ("Length =", length // 3)
        return length


    def __getitem__(self, idx):

        # Use binary search to find the correct group
        start_values = [self.start_indices[name] for name in self.group_names]
        group_idx = bisect.bisect_right(start_values, idx) - 1
        group_name = self.group_names[group_idx]

        # Calculate the local index within the group
        local_idx = idx - self.start_indices[group_name]
        # if self.type == "train":
        #     print(">>> Loaded: Index =", idx, "Group =", group_name, "Local =", local_idx)

        try:
            group = self.h5_file[group_name]
            board = group[f"board_{local_idx}"][:]
            liberty = group[f"liberty_{local_idx}"][:]
            label = group[f"nextMove_{local_idx}"][:]
        except Exception as e:
            # print(f"Error loading data at idx {idx}, group {group_name}, local_idx {local_idx}: {e}")
            # Use default values in case of error
            board = np.zeros((19, 19), dtype=np.float32)  # Default board filled with zeros
            liberty = np.zeros((19, 19), dtype=np.float32)  # Default liberty filled with zeros
            label = np.array([3, 3], dtype=np.int64)  # Default label at position (3, 3)

        # print("Board =\n", board)
        # print("Liberty =\n", liberty)
        # print("Label =\n", label)
        # print("<<<\n")

        ones = np.ones((19, 19), dtype=np.float32)  # One matrix
        zeros = np.zeros((19, 19), dtype=np.float32)  # Zero matrix
        board_self = np.where(board > 0, board, 0)
        board_opponent = np.where(board < 0, board, 0)
        liberty_self = np.where(liberty > 0, liberty, 0)
        liberty_opponent = np.where(liberty > 0, liberty, 0)
        empty_matrix = np.where(board == 0, 1, 0)


        # Check dimension
        if board.shape != (19, 19):
            raise ValueError(f"Invalid board shape: {board.shape}")
        if liberty.shape != (19, 19):
            raise ValueError(f"Invalid liberty shape: {liberty.shape}")

        # Combine board and liberty into a single NumPy array and then convert to Tensor
        combined_array = np.stack(
            [board,
             liberty,
             ones,
             ones,
             zeros,
             board_self,
             board_opponent,
             liberty_self,
             liberty_opponent,
             empty_matrix],
             axis=0)  # Shape: (10, 19, 19)
        input_tensor = torch.from_numpy(combined_array).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        if idx % 40000 == 0 and self.type == "test":
            print('#', end='', flush=True)

        return input_tensor, label_tensor



# Define the neural network
class GoNet(nn.Module):
    def __init__(self):
        super(GoNet, self).__init__()

        # 第一层卷积：输入通道 48，输出通道 192，卷积核大小 5x5，步幅 1，填充 2（保持尺寸）
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=192, kernel_size=5, stride=1, padding=2)
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

def initialize_weights(model):
    if isinstance(model, nn.Conv2d):
        init.xavier_uniform_(model.weight)  # Xavier 初始化
        if model.bias is not None:
            init.zeros_(model.bias)  # 偏置初始化为零
    elif isinstance(model, nn.Linear):
        init.kaiming_uniform_(model.weight, nonlinearity='relu')  # He 初始化
        if model.bias is not None:
            init.zeros_(model.bias)

def calculate_accuracy(model, test_data, batch_size):
    model.eval()
    correct = 0
    total = 0

    # Unpack test data
    inputs, labels = test_data

    with torch.no_grad():
        # Process data in batches
        for start_idx in range(0, inputs.size(0), batch_size):
            end_idx = min(start_idx + batch_size, inputs.size(0))
            input_batch = inputs[start_idx:end_idx]
            label_batch = labels[start_idx:end_idx]

            # Forward pass
            outputs = model(input_batch)
            _, predicted = torch.max(outputs, 1)
            target_indices = label_batch[:, 0] * 19 + label_batch[:, 1]

            # Accumulate correct predictions
            # print("predicted shape:", predicted.shape)
            # print("predicted =", predicted)
            # print("target indices shape", target_indices.shape)
            # print("target indices =", target_indices)
            correct += (predicted == target_indices).sum().item()
            total += label_batch.size(0)

    # Compute accuracy
    accuracy = correct / total
    return accuracy




# Training function
def train(model, train_loader, test_data, criterion, optimizer, max_epoch, device):

    global last_accuracy
    last_accuracy = 0.0

    # data iterator - manually load data
    data_iter = iter(train_loader)

    # learning rate curve
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.98)

    # Save accuracies
    train_accuracy_history = []
    test_accuracy_history = []

    for epoch in range(max_epoch):

        # Get one batch
        inputs, labels = next(data_iter)

        print(f">>> Epoch {epoch + 1}/", max_epoch)
        # Train the model, save accuracy every 100 epochs
        if epoch % 100 != 99:
            train_one_epoch(model, inputs, labels, criterion, optimizer, scheduler, device, False)
        else:
            train_accuracy_history.append(
                train_one_epoch(model, inputs, labels, criterion, optimizer, scheduler, device, True))
            print("Evaluating on test set...")
            accuracy = calculate_accuracy(model, test_data, batch_size = 512)
            if accuracy > last_accuracy:
                save_checkpoint(model, optimizer, epoch)
            last_accuracy = accuracy
            test_accuracy_history.append(accuracy)
            save_accuracy_plot(train_accuracy_history, test_accuracy_history)
            print(f"Test Accuracy after Epoch {epoch + 1}: {accuracy:.10f}")


def train_one_epoch(model, input_batch, label_batch, criterion, optimizer, scheduler, device, save_plot):
    model.train()

    # Move input and label batches to the same device
    input_batch = input_batch.to(device)
    label_batch = label_batch.to(device)

    optimizer.zero_grad()

    # Forward pass
    outputs = model(input_batch)  # Shape: [batch_size, 361]
    target_indices = label_batch[:, 0] * 19 + label_batch[:, 1]  # Shape: [batch_size]

    # Compute loss
    loss = criterion(outputs.view(outputs.size(0), -1), target_indices)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Compute training accuracy
    accuracy = 0
    if save_plot:
        _, predicted = torch.max(outputs, 1)  # Predicted indices, Shape: [batch_size]
        correct_predictions = (predicted == target_indices).sum().item()
        total_predictions = target_indices.size(0)
        accuracy = correct_predictions / total_predictions

    print(f"Training Loss: {loss.item():.10f}")
    return accuracy

def save_accuracy_plot(train_accuracy, test_accuracy, filename="accuracy_plot.png"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_accuracy) + 1)

    plt.plot(epochs, train_accuracy, label="Train Accuracy", marker="o", linestyle="-")
    plt.plot(epochs, test_accuracy, label="Test Accuracy", marker="o", linestyle="--")
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
        start_epoch = state["epoch"] + 1  # Start from the next epoch
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
        print("GPU not available. Using CPU instead.")
    print(f"Using device: {device}\n")

    # Examine the file exist
    print("File exist =", os.path.exists('Dataset/board-move-pairs-train.h5'))

    # # Check h5 file structure
    # with h5py.File('Dataset/board-move-pairs-training.h5', 'r') as f:
    #     def print_structure(name, obj):
    #         print(name, "->", obj)
    #     f.visititems(print_structure)

    # Load data with mini-batch size
    print(">>> Main: Loading dataset objects...")
    train_dataset = GoDataset('Dataset/board-move-pairs-train.h5', 'train')
    test_dataset = GoDataset('Dataset/board-move-pairs-test.h5', 'test')
    train_loader = DeviceDataLoader(train_dataset,
                                    batch_size=512,
                                    shuffle=True,
                                    device=device,
                                    num_workers=25,
                                    pin_memory=True)
    test_loader = DeviceDataLoader(test_dataset,
                                   batch_size=512,
                                   shuffle=False,
                                   device=device,
                                   num_workers=25,
                                   pin_memory=True)

    # Preload test data into GPU
    print(">>> Preloading test data into GPU...")
    test_inputs, test_labels = [], []
    for inputs, labels in test_loader:
        test_inputs.append(inputs.to(device))
        test_labels.append(labels.to(device))

    # Concatenate all test inputs and labels
    test_inputs = torch.cat(test_inputs, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    # Preloaded test data
    test_data = (test_inputs, test_labels)
    print("\n>>> Test data preloaded.")

    # Initialize model, loss function, and optimizer
    print(">>> Main: Initializing model...")

    model = GoNet()
    model.to(device)
    model.apply(initialize_weights)
    criterion = nn.CrossEntropyLoss()  # For classification of 361 positions
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)


    # Train the model
    print(">>> Main: Start training...")
    train(model, train_loader, test_data, criterion, optimizer, max_epoch = 200000, device = device)

    # Save the model
    print(">>> Main: Training finished. Saving model...")
    torch.save(model.state_dict(), 'go_model.pth')
    print(">>> Main: Model saved.")
