import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_curve, auc
from models.AudioCNN import AudioCNN
import matplotlib.pyplot as plt


# Load the validation data
X_val_mfcc = torch.from_numpy(np.load('processed_data/validate_mfcc_features.npy')).float().unsqueeze(1)
X_val_log_spectrogram = torch.from_numpy(np.load('processed_data/validate_log_spectrogram_features.npy')).float().unsqueeze(1)
y_val = torch.from_numpy(np.load('processed_data/validate_labels.npy')).long()

# Create a dataset and data loader
dataset_val = TensorDataset(X_val_mfcc, X_val_log_spectrogram, y_val)
batch_size = 32  # Define the batch size here
data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

# Load the trained model
num_classes =2
model = AudioCNN(num_classes)
model.load_state_dict(torch.load('savedModels/model.pth'))
model.eval()

# Make predictions on the validation data and compute metrics
f1_scores = []
roc_aucs = []
eers = []
eer_thresholds = []

with torch.no_grad():
    for mfcc, log_spectrogram, labels in data_loader_val:
        outputs = model(mfcc, log_spectrogram)
        _, predicted = torch.max(outputs, 1)

        # Compute metrics
        f1 = f1_score(labels.numpy(), predicted.numpy(), average='weighted')
        f1_scores.append(f1)

        fpr, tpr, thresholds = roc_curve(labels.numpy(), outputs[:, 1].numpy())
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)

        far = fpr
        frr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((far - frr)))]
        eer = far[np.nanargmin(np.absolute((far - frr)))]
        eers.append(eer)
        eer_thresholds.append(eer_threshold)

# Print average metrics
print(f"Average F1 Score: {np.mean(f1_scores)}")
print(f"Average ROC AUC: {np.mean(roc_aucs)}")
print(f"Average EER: {np.mean(eers)*100:.2f}% at threshold {np.mean(eer_thresholds)}")

# Plot ROC curve for the last batch
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()