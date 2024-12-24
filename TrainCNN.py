import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define Dataset for HDF5
class GoDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5py.File(h5_file, 'r')
        self.boards = []
        self.liberties = []
        self.labels = []

        for group_name in self.h5_file.keys():
            group = self.h5_file[group_name]
            for key in group.keys():
                if key.startswith("board"):
                    self.boards.append(group[key][:])  # Load borad
                elif key.startswith("liberty"):
                    self.liberties.append(group[key][:])  # Load Liberty
                elif key.startswith("nextMove"):
                    self.labels.append(group[key][:])  # Load label

        # Change to numpy
        self.boards = np.array(self.boards)
        self.liberties = np.array(self.liberties)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]
        liberty = self.liberties[idx]
        label = self.labels[idx]

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

# Examine the file exist
print("File exist =", os.path.exists('Dataset/board-move-pairs-training.h5'))

# # Check h5 file structure
# with h5py.File('Dataset/board-move-pairs-training.h5', 'r') as f:
#     def print_structure(name, obj):
#         print(name, "->", obj)
#     f.visititems(print_structure)


# Load data
train_dataset = GoDataset('Dataset/board-move-pairs-training.h5')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model, loss function, and optimizer
model = GoNet()
criterion = nn.CrossEntropyLoss()  # For classification of 361 positions
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Save the model
torch.save(model.state_dict(), 'go_model.pth')
