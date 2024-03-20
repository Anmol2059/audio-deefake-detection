import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from models.AudioCNN import AudioCNN
import torch

# Load the trained model
num_classes = 2  # bonafide and spoof
model = AudioCNN(num_classes)
model.load_state_dict(torch.load('savedModels/model.pth'))
model.eval()

uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])

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


st.title('Audio Deepfake Detection')

if uploaded_file is not None:
    audio_data, sr = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner('Processing...'):
        # Extract features
        mfcc, log_spectrogram = extract_features(uploaded_file, sr)
        mfcc = torch.from_numpy(mfcc).float().unsqueeze(0).unsqueeze(0)
        log_spectrogram = torch.from_numpy(log_spectrogram).float().unsqueeze(0).unsqueeze(0)

        # Make predictions
        with torch.no_grad():
            outputs = model(mfcc, log_spectrogram)
            _, predicted = torch.max(outputs, 1)

        # Display the prediction
        if predicted.item() == 0:
            st.markdown('**Prediction: Bonafide**', unsafe_allow_html=True)
        else:
            st.markdown('**Prediction: Spoof**', unsafe_allow_html=True)

        # Display the waveform
        st.subheader('Audio Waveform')
        plt.figure(figsize=(10, 2))
        plt.plot(np.linspace(0, len(audio_data)/sr, len(audio_data)), audio_data)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Audio Waveform')
        st.pyplot(plt)

        # Display the log-spectrogram
        st.subheader('Spectrogram')
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        st.pyplot(plt)

        # Display the MFCC
        st.subheader('MFCCs')
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc.cpu().squeeze().numpy(), sr=sr, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCCs')
        st.pyplot(plt)

        # Get the probabilities for the 'bonafide' and 'spoof' classes
        proba = torch.nn.functional.softmax(outputs, dim=1)
        proba_np = proba.numpy()[0]

        # Display class probabilities
        st.subheader('Class Probabilities')
        classes = ['bonafide', 'spoof']
        plt.bar(classes, proba_np)
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Class Probabilities')
        st.pyplot(plt)

        # Explain the class probabilities in plain language
        bonafide_prob = proba_np[0]
        spoof_prob = proba_np[1]
        st.write(f"The model is {bonafide_prob * 100:.2f}% confident that the audio is genuine (bonafide), and {spoof_prob * 100:.2f}% confident that the audio is fake (spoof).")

        # Display model architecture
        st.text(str(model))

st.markdown('Made by Anmol', unsafe_allow_html=True)