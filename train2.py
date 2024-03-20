import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from multiprocessing import Pool
import torchaudio.transforms as T
import torchaudio

# Set CUDA device
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print("Using CUDA device:", torch.cuda.current_device())

def get_MFCC(
    file_path: str,
    sample_rate: int = 16000,
    n_mfcc: int = 256,
    n_fft: int = 2048,
    hop_length: int = 512,
    log: bool = True,
) -> torch.Tensor:
    """
    Computes the MFCCs of a waveform.
    Args:
        file_path: str
            The path to the audio file to compute the MFCCs of.
        sample_rate: int
            The sample rate of the waveform.
        n_mfcc: int
            The Number of mfc coefficients to retain.
        n_fft: int
            The length of the FFT window.
        hop_length: int
            The number of samples between successive frames.
        log: bool
            Whether to use log-mel spectrograms instead of db-scaled.
    Returns:
        Tensor (B, n_mfcc, T') where T' = ceil(T / hop_length)
            The MFCCs of the waveform.
    """

    waveform, _ = torchaudio.load(file_path, num_frames=n_fft)
    waveform = waveform.mean(dim=0).unsqueeze(0)

    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        log_mels=log,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "mel_scale": "htk",
        },
    )
    mfcc = mfcc_transform(waveform)
    return mfcc

# Function to process a dataframe
def process_df(df):
    features = []
    labels = []
    total_files = len(df)
    with Pool(os.cpu_count()) as p:
        mfccs_list = p.map(get_MFCC, df['path'].tolist())
    features.extend(mfccs_list)
    labels.extend(df['label'].tolist())

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    return np.array(features), y  # Returns a 4D array: (num_samples, num_frames, num_mfcc, time), and a 1D array: (num_samples,)

# Load the CSV files
train_df = pd.read_csv(os.path.join(csv_dir, 'train.csv'))
validate_df = pd.read_csv(os.path.join(csv_dir, 'validate.csv'))

# Process each dataframe
X_train, y_train = process_df(train_df)  # X_train: (num_train_samples, num_frames, num_mfcc, time), y_train: (num_train_samples,)
X_val, y_val = process_df(validate_df)  # X_val: (num_val_samples, num_frames, num_mfcc, time), y_val: (num_val_samples,)

# Convert the data to PyTorch tensors
X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32*40*40, 100)
        self.fc2 = nn.Linear(100, len(np.unique(y_train)))

    def forward(self, x):
        x = self.conv1(x)  # Input: (batch_size, 1, num_frames, num_mfcc), Output: (batch_size, 32, num_frames, num_mfcc)
        x = x.view(x.size(0), -1)  # Flatten the tensor: (batch_size, 32*40*40)
        x = self.fc1(x)  # Input: (batch_size, 32*40*40), Output: (batch_size, 100)
        x = self.fc2(x)  # Input: (batch_size, 100), Output: (batch_size, num_classes)
        return x

# Instantiate the model
model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(zip(X_train, y_train), 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # Print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')