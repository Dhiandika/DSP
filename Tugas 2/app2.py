import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
import tempfile

# Load environment variables
load_dotenv()

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

# Initialize session state for model and conversation
if 'model' not in st.session_state:
    # Configure GenAI API Key
    genai.configure(api_key=os.getenv('API_KEY'))
    # Configure GenAI Model
    st.session_state.model = genai.GenerativeModel("gemini-1.5-flash-latest")
    st.session_state.convo = st.session_state.model.start_chat(history=[])
    print("INITIALIZE")

# Initialize session state for storing responses
if 'responses' not in st.session_state:
    st.session_state['responses'] = []

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
def plot_waveform(y, sr, temp_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(y) / sr, len(y)), y)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    waveform_path = os.path.join(temp_dir, 'waveform.png')
    plt.savefig(waveform_path)
    plt.close()
    return waveform_path

# Function to plot spectrogram
def plot_spectrogram(y, sr, temp_dir):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    spectrogram_path = os.path.join(temp_dir, 'spectrogram.png')
    plt.savefig(spectrogram_path)
    plt.close()
    return spectrogram_path

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

if st.button("Generate Response"):
    if uploaded_file:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded file temporarily
            temp_audio_path = os.path.join(temp_dir, "temp.wav")
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load audio file
            y, sr = librosa.load(temp_audio_path, sr=None)
            
            # Display audio player
            st.audio(temp_audio_path)
            
            # Plot and save waveform
            st.subheader("Waveform")
            waveform_path = plot_waveform(y, sr, temp_dir)
            st.image(waveform_path)
            
            # Plot and save spectrogram
            st.subheader("Spectrogram")
            spectrogram_path = plot_spectrogram(y, sr, temp_dir)
            st.image(spectrogram_path)
            
            # Predict audio category
            st.subheader("Prediction")
            predicted_label = predict_audio_category(loaded_model, temp_audio_path)
            st.write(f"The predicted class is: {predicted_label}")
            
            # Generate response for the prediction
            if predicted_label == 'belly_pain':
                advice = "Bayimu terdeteksi mengalami sakit perut. Anda sebaiknya memeriksa apakah bayi Anda merasa kembung atau memerlukan bantuan untuk buang air besar."
            elif predicted_label == 'burping':
                advice = "Bayimu terdeteksi bersendawa. Anda sebaiknya membantu bayi Anda bersendawa setelah menyusui untuk mengeluarkan udara yang tertelan."
            elif predicted_label == 'discomfort':
                advice = "Bayimu terdeteksi tidak nyaman. Anda sebaiknya memeriksa popoknya atau mencoba menenangkan bayi Anda dengan menggendongnya."
            elif predicted_label == 'hungry':
                advice = "Bayimu terdeteksi lapar. Anda sebaiknya memberi ASI atau susu formula untuk mengatasi rasa laparnya."
            elif predicted_label == 'tired':
                advice = "Bayimu terdeteksi mengantuk. Anda sebaiknya membantu bayi Anda tidur dengan menidurkannya di tempat yang nyaman dan tenang."
            else:
                advice = "Tidak dapat mendeteksi kondisi bayi. Silakan coba lagi."

            # Combine prediction and advice
            input_text = f"Audio Prediction: {predicted_label} \n\n Advice: {advice}"
            
            # Load images using PIL
            waveform_image = Image.open(waveform_path)
            spectrogram_image = Image.open(spectrogram_path)
            
            # Ensure images are closed properly
            waveform_image.close()
            spectrogram_image.close()
            
            # Send the input_text to AI model
            st.session_state.convo.send_message(content=input_text)
            response_text = st.session_state.convo.last.text
            response_text = response_text.replace("AI Response:", "Asisten Pintar:").replace("The advice given in the audio prediction seems sound and is a good starting point for addressing a baby's belly pain. Here's a breakdown:", "Rekomendasi:").replace("What the advice is suggesting:", "Saran:").replace("Why this advice is good:", "Mengapa saran ini baik:").replace("Important Considerations:", "Pertimbangan penting:").replace("Additional Tips:", "Tips tambahan:").replace("Overall:", "Keseluruhan:")
            st.session_state.responses.append(f"Audio Prediction: {predicted_label} \n \n Advice: {advice} \n \n Asisten Pintar: {response_text}")

# Display all responses in a scrollable container
st.subheader("Minta Saran Asisten Pintar")
response_container = st.container()
with response_container:
    for resp in st.session_state['responses']:
        st.markdown(f"<div style='border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; background-color: #31363F;'>{resp}</div>", unsafe_allow_html=True)
        st.markdown("---")  # Add a horizontal line for better separation
