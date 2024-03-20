import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve
from models.AudioCNN import AudioCNN

# Define paths to the data files
data_dir = "processed_data"
X_train_mfcc_path = os.path.join(data_dir, 'train_mfcc_features.npy')
X_train_log_spectrogram_path = os.path.join(data_dir, 'train_log_spectrogram_features.npy')
y_train_path = os.path.join(data_dir, 'train_labels.npy')

# Load the training data
X_train_mfcc = torch.from_numpy(np.load(X_train_mfcc_path)).float().unsqueeze(1)
X_train_log_spectrogram = torch.from_numpy(np.load(X_train_log_spectrogram_path)).float().unsqueeze(1)
y_train = torch.from_numpy(np.load(y_train_path)).long()

# Create a dataset and data loader
dataset = TensorDataset(X_train_mfcc, X_train_log_spectrogram, y_train)
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Specify the CUDA device
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# Instantiate the model, loss function, and optimizer
num_classes = len(np.unique(np.load(y_train_path)))
model = AudioCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for mfcc, log_spectrogram, labels in data_loader:
        mfcc, log_spectrogram, labels = mfcc.to(device), log_spectrogram.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(mfcc, log_spectrogram)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(data_loader)}")
"""
Here loss is a numerical value that represents the average loss per batch for that epoch. It's a measure of how well the model's predictions match the true labels.
"""

torch.save(model.state_dict(), 'savedModels/model.pth')
print("Model saved!")