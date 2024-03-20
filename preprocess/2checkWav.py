import os
import librosa
import pandas as pd
from tqdm import tqdm

# Define the directory to load CSV files
csv_dir = "../CSVfiles"

# Load the CSV files
train_df = pd.read_csv(os.path.join(csv_dir, 'train.csv'))
validate_df = pd.read_csv(os.path.join(csv_dir, 'validate.csv'))
evaluate_df = pd.read_csv(os.path.join(csv_dir, 'evaluate.csv'))

# Function to get audio length and sample rate
def get_audio_info(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    duration = len(audio) / sr  # Calculate duration in seconds
    return duration, sr

def analyze_set(df, set_name):
    print(f"\nAnalyzing {set_name} set...")
    audio_durations = []
    sample_rates = []
    less_than_5s = 0 
    greater_than_5s = 0
    greater_than_10s = 0
    greater_than_15s = 0
    greater_than_30s = 0
    for file_path in tqdm(df['path']):
        duration, sr = get_audio_info(file_path)
        audio_durations.append(duration)
        sample_rates.append(sr)
        if duration < 5:
            less_than_5s += 1
        if duration > 5:
            greater_than_5s += 1
        if duration > 10:
            greater_than_10s += 1
        if duration > 15:
            greater_than_15s += 1
        if duration > 30:
            greater_than_30s += 1

    print(f"{set_name} set statistics:")
    print(f"Average audio duration: {sum(audio_durations) / len(audio_durations):.2f} seconds")
    print(f"Average sample rate: {sum(sample_rates) / len(sample_rates):.2f} Hz")
    print(f"Minimum audio duration: {min(audio_durations):.2f} seconds")
    print(f"Maximum audio duration: {max(audio_durations):.2f} seconds")
    print(f"Minimum sample rate: {min(sample_rates)} Hz")
    print(f"Maximum sample rate: {max(sample_rates)} Hz")
    print(f"Number of audio files less than 5 seconds: {less_than_5s}") 
    print(f"Number of audio files greater than 5 seconds: {greater_than_5s}")
    print(f"Number of audio files greater than 10 seconds: {greater_than_10s}")
    print(f"Number of audio files greater than 15 seconds: {greater_than_15s}")
    print(f"Number of audio files greater than 30 seconds: {greater_than_30s}")

# Use the function to analyze each set
analyze_set(train_df, "train")
analyze_set(validate_df, "validate")
analyze_set(evaluate_df, "evaluate")