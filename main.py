"""A ConvLSTM DL model for ECG data classification by Zibin ZHAO @HSING Group"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.tensorboard import SummaryWriter

# Initialize the TensorBoard SummaryWriter
writer = SummaryWriter()

# Seed everything
seed = 42
seed_everything(seed)


# Load data
X = pd.read_csv("./data/Concatenated_X.csv").values     # shape(40828, 3600)
y = pd.read_csv("./data/Concatenated_y.csv").values     # shape(40828, 1)

#print(X.shape, y.shape)

# Normalize data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)


# Label Encoding
AAMI = ['N','L','R','V','A','|','B']
# N:Normal
# L:Left bundle branch block beat
# R:Right bundle branch block beat
# V:Premature ventricular contraction
# A:Atrial premature contraction
# |:Isolated QRS-like artifact
# B:Left or right bundle branch block

le = LabelEncoder()
le = le.fit(AAMI)
y_encoded = le.transform(y)

# encoder = LabelEncoder()
# y_encoded = encoder.fit_transform(y)


# Split the dataset (60-20-20)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2



class ECGDataAugmentation:
    def __init__(self, shift_range=50, scale_range=0.1):
        self.shift_range = shift_range
        self.scale_range = scale_range

    def time_shift(self, signal):
        shift = np.random.randint(-self.shift_range, self.shift_range)
        return np.roll(signal, shift)

    def scale(self, signal):
        scale_factor = np.random.uniform(1 - self.scale_range, 1 + self.scale_range)
        return signal * scale_factor

    def __call__(self, signal):
        signal = self.time_shift(signal)
        signal = self.scale(signal)
        return signal

class ECGDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment
        self.ecg_data_augmentation = ECGDataAugmentation()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = self.ecg_data_augmentation(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


# Create data loaders
batch_size = 64

train_dataset = ECGDataset(X_train, y_train, augment=True)
val_dataset = ECGDataset(X_val, y_val)
test_dataset = ECGDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    total = target.size(0)
    return correct / total

# Model definition
class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()

        self.conv1 = nn.Conv1d(1, 128, kernel_size=5, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv1d(256, 512, kernel_size=5, padding=2)
        self.batch_norm3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(0.5)

        pool_output_size = 3600 // 2 // 2 // 2

        self.lstm = nn.LSTM(512 * pool_output_size, 256, num_layers=3, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 7)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 1, 3600)  # Reshape the input
        x = self.dropout1(self.pool1(self.relu1(self.batch_norm1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.batch_norm2(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.relu3(self.batch_norm3(self.conv3(x)))))

        # Calculate the output size of the last pooling layer
        pool_output_size = x.size(2)

        x = x.contiguous().view(x.size(0), -1, 512 * pool_output_size)  # Flatten the tensor and keep the batch size

        _, (h_n, _) = self.lstm(x)
        x = self.dropout_lstm(h_n[-1])  # Apply dropout after the LSTM layer
        x = self.fc(x)
        x = self.softmax(x)
        return x

    
# Set the device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = ConvLSTM()
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-5)


# Train the model
n_epochs = 10

for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy(outputs, labels)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # Validate the model
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_acc += accuracy(outputs, labels)
            # Inside the training loop, after the validation loop
            scheduler.step(val_loss)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    


    print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Test the model
model.eval()
test_loss = 0.0
test_acc = 0.0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        test_acc += accuracy(outputs, labels)

test_loss /= len(test_loader)
test_acc /= len(test_loader)

print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


# Save the model
torch.save(model.state_dict(), "trained_model.pth")
print("Model saved to trained_model.pth")

