import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Optional: Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress TensorFlow warnings
import logging
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras.models import load_model

# Define class labels
kelas_label = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

# Load the trained model
model_path = 'model/babycrymodell.h5'  # Update path to your model
loaded_model = load_model(model_path)

# Define feature extraction function
def extract_features(file_path, max_length=10000, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Pad or truncate MFCCs to max_length
    if mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    
    # Reshape for CNN input (n_features, max_length, 1)
    mfccs = np.expand_dims(mfccs, axis=-1)
    return mfccs

# Function to plot waveform
def plot_waveform(y, sr):
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(y) / sr, len(y)), y)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    st.pyplot(plt)

# Function to plot spectrogram
def plot_spectrogram(y, sr):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    st.pyplot(plt)

# Function to predict audio category
def predict_audio_category(model, audio_file_path, max_length=10000):
    y, sr = librosa.load(audio_file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    if mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
    mfccs = np.expand_dims(mfccs, axis=-1)  # Add channel dimension
    
    predictions = model.predict(mfccs)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = kelas_label[predicted_class_index]
    
    return predicted_class_label

# Title of the web app
st.title('Baby Cry Prediction App')

# File uploader
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load audio file
    y, sr = librosa.load("temp.wav", sr=None)
    
    # Display audio player
    st.audio("temp.wav")
    
    # Plot and display waveform
    st.subheader("Waveform")
    plot_waveform(y, sr)
    
    # Plot and display spectrogram
    st.subheader("Spectrogram")
    plot_spectrogram(y, sr)
    
    # Predict audio category
    st.subheader("Prediction")
    predicted_label = predict_audio_category(loaded_model, "temp.wav")
    st.write(f"The predicted class is: {predicted_label}")
