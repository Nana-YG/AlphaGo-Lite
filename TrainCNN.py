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
        board = group[f"board_{local_idx}"][:]
        liberty = group[f"liberty_{local_idx}"][:]
        label = group[f"nextMove_{local_idx}"][:]

        print("Board =\n", board)
        print("Liberty =\n", liberty)
        print("Label =\n", label)
        print("<<<\n")

        # Check dimension
        if board.shape != (19, 19):
            raise ValueError(f"Invalid board shape: {board.shape}")
        if liberty.shape != (19, 19):
            raise ValueError(f"Invalid liberty shape: {liberty.shape}")

        # Combine to tensor
        input_tensor = torch.tensor([board, liberty], dtype=torch.float32)
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

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10000):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            # Convert (row, col) to single index for loss calculation
            target_indices = labels[:, 0] * 19 + labels[:, 1]
            loss = criterion(outputs, target_indices)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

if __name__ == "__main__":

    # Examine the file exist
    print("File exist =", os.path.exists('Dataset/board-move-pairs-train.h5'))

    # # Check h5 file structure
    # with h5py.File('Dataset/board-move-pairs-training.h5', 'r') as f:
    #     def print_structure(name, obj):
    #         print(name, "->", obj)
    #     f.visititems(print_structure)


    # Load data
    print(">>> Main: Loading dataset...")
    train_dataset = GoDataset('Dataset/board-move-pairs-train.h5')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize model, loss function, and optimizer
    print(">>> Main: Initializing model...")
    model = GoNet()
    criterion = nn.CrossEntropyLoss()  # For classification of 361 positions
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print(">>> Main: Start training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=10000)

    # Save the model
    print(">>> Main: Training finished. Saving model...")
    torch.save(model.state_dict(), 'go_model.pth')

    print(">>> Main: Model saved.")
