import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import bisect

# Define Dataset for HDF5
class GoDataset(Dataset):
    def __init__(self, h5_file):
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
        # length = 0
        # for group in self.group_names:
        #     group_keys = self.h5_file[group].keys()  # get all keys from current group
        #     # print("Group keys:", group_keys)
        #     length += len(group_keys)
        #
        # print ("Length =", length // 3)
        return 106047633


    def __getitem__(self, idx):

        # Use binary search to find the correct group
        start_values = [self.start_indices[name] for name in self.group_names]
        group_idx = bisect.bisect_right(start_values, idx) - 1
        group_name = self.group_names[group_idx]

        # Calculate the local index within the group
        local_idx = idx - self.start_indices[group_name]

        group = self.h5_file[group_name]
        print(">>> Loaded: Index =", idx, "Group =", group_name, "Local =", local_idx)

        try:
            group = self.h5_file[group_name]
            board = group[f"board_{local_idx}"][:]
            liberty = group[f"liberty_{local_idx}"][:]
            label = group[f"nextMove_{local_idx}"][:]
        except Exception as e:
            print(f"Error loading data at idx {idx}, group {group_name}, local_idx {local_idx}: {e}")
            # Use default values in case of error
            board = np.zeros((19, 19), dtype=np.float32)  # Default board filled with zeros
            liberty = np.zeros((19, 19), dtype=np.float32)  # Default liberty filled with zeros
            label = np.array([3, 3], dtype=np.int64)  # Default label at position (3, 3)

        # print("Board =\n", board)
        # print("Liberty =\n", liberty)
        # print("Label =\n", label)
        # print("<<<\n")

        # Check dimension
        if board.shape != (19, 19):
            raise ValueError(f"Invalid board shape: {board.shape}")
        if liberty.shape != (19, 19):
            raise ValueError(f"Invalid liberty shape: {liberty.shape}")

        # Combine board and liberty into a single NumPy array and then convert to Tensor
        combined_array = np.stack([board, liberty], axis=0)  # Shape: (2, 19, 19)
        input_tensor = torch.from_numpy(combined_array).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return input_tensor, label_tensor



# Define the neural network
class GoNet(nn.Module):
    def __init__(self):
        super(GoNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)  # 2 channels: board and liberty
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 19 * 19, 1024)
        self.fc2 = nn.Linear(1024, 361)  # Output 361 for 19x19 board positions and 362 for pass

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, input_tensor):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output = self(input_tensor.unsqueeze(0))  # Add batch dimension
            _, predicted = torch.max(output, 1)
        row, col = divmod(predicted.item(), 19)
        return row, col

    def calculate_accuracy(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)

                # Convert (row, col) to single index for prediction and comparison
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                target_indices = labels[:, 0] * 19 + labels[:, 1]

                correct += (predicted == target_indices).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        return accuracy

# Training function
def train(model, train_loader, test_loader, criterion, optimizer, max_epoch):

    data_iter = iter(train_loader)

    for epoch in range(max_epoch):

        # Get one batch
        inputs, labels = next(data_iter)

        print(f"Epoch {epoch + 1}/10000")
        train_one_epoch(model, inputs, labels, criterion, optimizer)
        print("Evaluating on test set...")
        accuracy = model.calculate_accuracy(model, test_loader)
        print(f"Test Accuracy after Epoch {epoch + 1}: {accuracy:.8f}")


def train_one_epoch(model, input_batch, label_batch, criterion, optimizer, ):
    model.train()
    total_loss = 0
    total_samples = 0

    for inputs, labels in input_batch, label_batch:

        # Move inputs and labels to the same device as the model
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # Convert (row, col) to single index for loss calculation
        target_indices = labels[:, 0] * 19 + labels[:, 1]
        loss = criterion(outputs, target_indices)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)  # Multiply by batch size
        total_samples += inputs.size(0)  # Count total samples

    average_loss = total_loss / total_samples  # Compute average loss
    print(f"Training Loss: {average_loss:.4f}")
    return average_loss



if __name__ == "__main__":

    # Examine the file exist
    print("File exist =", os.path.exists('Dataset/board-move-pairs-train.h5'))

    # # Check h5 file structure
    # with h5py.File('Dataset/board-move-pairs-training.h5', 'r') as f:
    #     def print_structure(name, obj):
    #         print(name, "->", obj)
    #     f.visititems(print_structure)

    # Load data with mini-batch size
    print(">>> Main: Loading datasets...")
    train_dataset = GoDataset('Dataset/board-move-pairs-train.h5')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataset = GoDataset('Dataset/board-move-pairs-test.h5')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)



    # Initialize model, loss function, and optimizer
    print(">>> Main: Initializing model...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("GPU not available. Using CPU instead.")
    print(f"Using device: {device}\n")

    model = GoNet()
    model.to(device)
    criterion = nn.CrossEntropyLoss()  # For classification of 361 positions
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Train the model
    print(">>> Main: Start training...")
    train(model, train_loader, test_loader, criterion, optimizer, max_epoch = 100000)

    # Save the model
    print(">>> Main: Training finished. Saving model...")
    torch.save(model.state_dict(), 'go_model.pth')
    print(">>> Main: Model saved.")
