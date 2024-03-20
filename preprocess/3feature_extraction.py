import os
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder

# Define the directory to load CSV files
csv_dir = "../CSVfiles"

# Define the directory to save processed data
data_dir = "../processed_data"
os.makedirs(data_dir, exist_ok=True)

# Function to extract MFCC and log-spectrogram features
def extract_features(file_path, sr=16000, n_fft=2048, hop_length=512, max_pad_len=128000, max_mfcc_len=100):
    # Load the audio file and resample to the target sample rate
    audio, _ = librosa.load(file_path, sr=sr)

    # If the audio file is shorter than max_pad_len, pad it with zeros
    if len(audio) < max_pad_len:
        audio = np.pad(audio, pad_width=(0, max_pad_len - len(audio)), mode='constant')

    # If the audio file is longer than max_pad_len, truncate it
    else:
        audio = audio[:max_pad_len]

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    # If the MFCC feature vector is shorter than max_mfcc_len, pad it with zeros
    if mfccs.shape[1] < max_mfcc_len:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, max_mfcc_len - mfccs.shape[1])), mode='constant')

    # If the MFCC feature vector is longer than max_mfcc_len, truncate it
    else:
        mfccs = mfccs[:, :max_mfcc_len]

    # Extract log-spectrogram features
    spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    log_spectrogram = librosa.amplitude_to_db(np.abs(spectrogram))

    # If the log-spectrogram feature vector is shorter than max_mfcc_len, pad it with zeros
    if log_spectrogram.shape[1] < max_mfcc_len:
        log_spectrogram = np.pad(log_spectrogram, pad_width=((0, 0), (0, max_mfcc_len - log_spectrogram.shape[1])), mode='constant')

    # If the log-spectrogram feature vector is longer than max_mfcc_len, truncate it
    else:
        log_spectrogram = log_spectrogram[:, :max_mfcc_len]

    return mfccs, log_spectrogram

# Function to process a dataframe
def process_df(df, X_mfcc_save_path, X_log_spectrogram_save_path, y_save_path):
    mfcc_features = []
    log_spectrogram_features = []
    labels = []
    total_files = len(df)
    for i, (index, row) in enumerate(df.iterrows(), 1):
        file_name = row['path']
        class_label = row['label']
        mfccs, log_spectrogram = extract_features(file_name)
        mfcc_features.append(mfccs)
        log_spectrogram_features.append(log_spectrogram)
        labels.append(class_label)
        print(f"Processed {i}/{total_files} files", end='\r')  # \r to overwrite the previous print

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # If the .npy files already exist, delete them
    if os.path.exists(X_mfcc_save_path):
        os.remove(X_mfcc_save_path)
    if os.path.exists(X_log_spectrogram_save_path):
        os.remove(X_log_spectrogram_save_path)
    if os.path.exists(y_save_path):
        os.remove(y_save_path)

    # Save the processed data
    np.save(X_mfcc_save_path, mfcc_features)
    np.save(X_log_spectrogram_save_path, log_spectrogram_features)
    np.save(y_save_path, y)

    
# Load the CSV files
train_df = pd.read_csv(os.path.join(csv_dir, 'train.csv'))
validate_df = pd.read_csv(os.path.join(csv_dir, 'validate.csv'))
evaluate_df = pd.read_csv(os.path.join(csv_dir, 'evaluate.csv'))

# Process each dataframe
process_df(train_df, os.path.join(data_dir, 'train_mfcc_features.npy'), os.path.join(data_dir, 'train_log_spectrogram_features.npy'), os.path.join(data_dir, 'train_labels.npy'))
process_df(validate_df, os.path.join(data_dir, 'validate_mfcc_features.npy'), os.path.join(data_dir, 'validate_log_spectrogram_features.npy'), os.path.join(data_dir, 'validate_labels.npy'))
process_df(evaluate_df, os.path.join(data_dir, 'evaluate_mfcc_features.npy'), os.path.join(data_dir, 'evaluate_log_spectrogram_features.npy'), os.path.join(data_dir, 'evaluate_labels.npy'))