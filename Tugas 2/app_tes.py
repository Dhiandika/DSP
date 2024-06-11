import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import random
import streamlit as st
import tempfile
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
dataset_dir = '/content/drive/MyDrive/Project/donateacry_corpus_cleaned_and_updated_data/'

# Initialize lists for storing audio features and labels
audio_features = []
labels = []
all_audio_files = []

# Maximum length of MFCC features (number of frames)
max_length = 300

# Function to extract MFCC features from audio file
def extract_features(file_path, max_length=300, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Trim or pad MFCCs to have the shape (n_mfcc, max_length)
    if mfccs.shape[1] > max_length:
        mfccs = mfccs[:, :max_length]
    else:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    
    # Expand dimensions to match (1, n_mfcc, max_length)
    mfccs = np.expand_dims(mfccs, axis=0)
    
    return mfccs

# Iterate through each category folder to extract audio features
for category in os.listdir(dataset_dir):
    category_dir = os.path.join(dataset_dir, category)
    for filename in os.listdir(category_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(category_dir, filename)
            all_audio_files.append((filepath, category))

# Extract features and labels
for filepath, category in all_audio_files:
    mfccs = extract_features(filepath, max_length)
    audio_features.append(mfccs)
    labels.append(category)

# Convert labels to numerical values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(audio_features, labels, test_size=0.2, random_state=42)

# Convert data to NumPy arrays
X_train = np.array(X_train).reshape(-1, 13, 300)
X_test = np.array(X_test).reshape(-1, 13, 300)

# Normalize features
scaler = StandardScaler()
n_samples, n_features, n_frames = X_train.shape
X_train = X_train.reshape(n_samples, n_features * n_frames)
X_train = scaler.fit_transform(X_train)
X_train = X_train.reshape(n_samples, n_features, n_frames)

n_samples, n_features, n_frames = X_test.shape
X_test = X_test.reshape(n_samples, n_features * n_frames)
X_test = scaler.transform(X_test)
X_test = X_test.reshape(n_samples, n_features, n_frames)

# Reshape for CNN input
X_train_cnn = np.expand_dims(X_train, axis=-1)
X_test_cnn = np.expand_dims(X_test, axis=-1)

# Build RNN model with LSTM
rnn_model = Sequential([
    LSTM(512, return_sequences=True, input_shape=(13, 300), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train RNN model
rnn_history = rnn_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Build CNN model
input_shape = (13, 300, 1)
cnn_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train CNN model
cnn_history = cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Save models
rnn_model.save('Model_RNN.h5')
cnn_model.save('Model_CNN.h5')

# Function to predict audio category
def predict_audio_category(model, mfccs):
    print(f"MFCCs shape before prediction: {mfccs.shape}")
    predictions = model.predict(mfccs)
    return predictions

# Streamlit application
st.title("Audio Category Prediction")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file:
    logger.info('Audio file uploaded: %s', uploaded_file.name)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio_path = os.path.join(temp_dir, "temp.wav")
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        y, sr = librosa.load(temp_audio_path, sr=None)
        st.audio(temp_audio_path)

        # Extract MFCC features
        mfccs = extract_features(temp_audio_path, max_length)
        mfccs_cnn = np.expand_dims(mfccs, axis=-1)  # Shape: (1, 13, 300, 1)

        # Log MFCCs shape
        logger.info(f"Extracted MFCCs shape: {mfccs.shape}")

        # Visualization Tabs
        tab1, tab2, tab3 = st.tabs(["Waveform", "Spectrogram", "Mel-spectrogram"])

        with tab1:
            st.subheader("Waveform")
            waveform_fig = plt.figure(figsize=(10, 4))
            plt.plot(y)
            plt.title("Waveform")
            st.pyplot(waveform_fig)

        with tab2:
            st.subheader("Spectrogram")
            spectrogram_fig = plt.figure(figsize=(10, 4))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title("Spectrogram")
            st.pyplot(spectrogram_fig)

        with tab3:
            st.subheader("Mel-spectrogram")
            mel_spectrogram_fig = plt.figure(figsize=(10, 4))
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title("Mel-spectrogram")
            st.pyplot(mel_spectrogram_fig)

        # Predictions for RNN and CNN
        rnn_predictions = predict_audio_category(rnn_model, mfccs)
        cnn_predictions = predict_audio_category(cnn_model, mfccs_cnn)

        rnn_predicted_label = label_encoder.inverse_transform([np.argmax(rnn_predictions)])[0]
        cnn_predicted_label = label_encoder.inverse_transform([np.argmax(cnn_predictions)])[0]

        st.subheader("Prediction Results")
        st.write(f"RNN Predicted Label: {rnn_predicted_label}")
        st.write(f"CNN Predicted Label: {cnn_predicted_label}")

        advice = ""
        if rnn_predicted_label == "belly_pain":
            advice = "Bayi Anda mungkin mengalami sakit perut."
        elif rnn_predicted_label == "burping":
            advice = "Bayi Anda mungkin perlu bersendawa."
        elif rnn_predicted_label == "discomfort":
            advice = "Bayi Anda mungkin merasa tidak nyaman."
        elif rnn_predicted_label == "hungry":
            advice = "Bayi Anda mungkin lapar."
        elif rnn_predicted_label == "tired":
            advice = "Bayi Anda mungkin mengantuk."

        st.write(f"RNN Advice: {advice}")

        advice = ""
        if cnn_predicted_label == "belly_pain":
            advice = "Bayi Anda mungkin mengalami sakit perut."
        elif cnn_predicted_label == "burping":
            advice = "Bayi Anda mungkin perlu bersendawa."
        elif cnn_predicted_label == "discomfort":
            advice = "Bayi Anda mungkin merasa tidak nyaman."
        elif cnn_predicted_label == "hungry":
            advice = "Bayi Anda mungkin lapar."
        elif cnn_predicted_label == "tired":
            advice = "Bayi Anda mungkin mengantuk."

        st.write(f"CNN Advice: {advice}")
