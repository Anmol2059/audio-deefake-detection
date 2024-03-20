import torch
import torch.nn as nn
# Define the CNN model
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        
        # MFCC branch
        self.conv1_mfcc = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.pool1_mfcc = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2_mfcc = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool2_mfcc = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Log-spectrogram branch
        self.conv1_log = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.pool1_log = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2_log = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool2_log = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 10 * 25 + 64 * 256 * 25, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, mfcc, log_spectrogram):
        # MFCC branch
        x_mfcc = self.pool1_mfcc(nn.functional.relu(self.conv1_mfcc(mfcc)))
        x_mfcc = self.pool2_mfcc(nn.functional.relu(self.conv2_mfcc(x_mfcc)))
        x_mfcc = x_mfcc.view(-1, 64 * 10 * 25)
        
        # Log-spectrogram branch
        x_log = self.pool1_log(nn.functional.relu(self.conv1_log(log_spectrogram)))
        x_log = self.pool2_log(nn.functional.relu(self.conv2_log(x_log)))
        x_log = x_log.view(-1, 64 * 256 * 25)
        
        # Concatenate the features
        x = torch.cat((x_mfcc, x_log), dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
