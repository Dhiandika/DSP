import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import logging
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
import tempfile
from tensorflow.keras.models import load_model

# Load environment variables
load_dotenv()

# Configure TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel(logging.ERROR)

# Define class labels
kelas_label = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

# Load the trained model
model_path = 'model/babycrymodell.h5'
loaded_model = load_model(model_path)

# Initialize session state
if 'model' not in st.session_state:
    genai.configure(api_key=os.getenv('API_KEY'))
    st.session_state.model = genai.GenerativeModel("gemini-1.5-flash-latest")
    st.session_state.convo = st.session_state.model.start_chat(history=[])
    print("INITIALIZE")

if 'responses' not in st.session_state:
    st.session_state['responses'] = []

# Feature extraction function
def extract_features(file_path, max_length=10000, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    mfccs = np.expand_dims(mfccs, axis=-1)
    return mfccs

# Waveform plotting function
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

# Spectrogram plotting function
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

# Prediction function
def predict_audio_category(model, audio_file_path, max_length=10000):
    y, sr = librosa.load(audio_file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    mfccs = np.expand_dims(mfccs, axis=0)
    mfccs = np.expand_dims(mfccs, axis=-1)
    predictions = model.predict(mfccs)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = kelas_label[predicted_class_index]
    return predicted_class_label

# Streamlit app layout
st.title('Baby Cry Prediction App')
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if st.button("Generate Response"):
    if uploaded_file:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio_path = os.path.join(temp_dir, "temp.wav")
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            y, sr = librosa.load(temp_audio_path, sr=None)
            st.audio(temp_audio_path)
            st.subheader("Waveform")
            waveform_path = plot_waveform(y, sr, temp_dir)
            st.image(waveform_path)
            st.subheader("Spectrogram")
            spectrogram_path = plot_spectrogram(y, sr, temp_dir)
            st.image(spectrogram_path)
            st.subheader("Prediction")
            predicted_label = predict_audio_category(loaded_model, temp_audio_path)
            st.write(f"The predicted class is: {predicted_label}")

            advice = ""
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

            input_text = f"Prediksi Audio: {predicted_label} \n\n Saran: {advice}"

            waveform_image = Image.open(waveform_path)
            spectrogram_image = Image.open(spectrogram_path)
            waveform_image.close()
            spectrogram_image.close()

            st.session_state.convo.send_message(content=input_text)
            response_text = st.session_state.convo.last.text
            response_text = response_text.replace("AI Response:", "Asisten Pintar:")
            response_text = response_text.replace("This audio prediction is very accurate. It accurately identifies the sound of a baby's discomfort and provides useful advice.", 
                                                  "Prediksi audio ini sangat akurat. Ini mengidentifikasi suara ketidaknyamanan bayi dengan tepat dan memberikan saran yang berguna.")
            response_text = response_text.replace("Here's a breakdown of why the prediction works:", "Berikut adalah penjelasan mengapa prediksi ini berhasil:")
            response_text = response_text.replace("Sound Recognition: The AI likely uses advanced sound recognition algorithms to identify specific sounds associated with baby discomfort. This could include cries, whimpers, fussiness, and other noises that babies make when they're unhappy.",
                                                  "Pengenalan Suara: AI kemungkinan menggunakan algoritma pengenalan suara canggih untuk mengidentifikasi suara-suara spesifik yang terkait dengan ketidaknyamanan bayi. Ini bisa termasuk tangisan, rengekan, rewel, dan suara lain yang dibuat bayi ketika mereka tidak senang.")
            response_text = response_text.replace("Contextual Understanding: The AI likely considers the context of the audio. For example, if the audio includes other sounds like baby babbling or cooing, it can distinguish between normal baby noises and those indicating discomfort.",
                                                  "Pemahaman Kontekstual: AI kemungkinan mempertimbangkan konteks audio. Misalnya, jika audio termasuk suara lain seperti celoteh atau ocehan bayi, AI dapat membedakan antara suara bayi yang normal dan yang menunjukkan ketidaknyamanan.")
            response_text = response_text.replace("Relevant Advice: The AI provides appropriate advice based on the recognized discomfort. Checking the diaper and comforting the baby are two of the most common and effective ways to soothe a distressed infant.",
                                                  "Saran yang Relevan: AI memberikan saran yang tepat berdasarkan ketidaknyamanan yang dikenali. Memeriksa popok dan menenangkan bayi adalah dua cara yang paling umum dan efektif untuk menenangkan bayi yang gelisah.")
            response_text = response_text.replace("Overall, this is a well-designed audio prediction system that can be helpful for parents and caregivers who are trying to understand their baby's needs.",
                                                  "Secara keseluruhan, ini adalah sistem prediksi audio yang dirancang dengan baik yang dapat membantu orang tua dan pengasuh yang berusaha memahami kebutuhan bayi mereka.")
            
            st.session_state.responses.append(f"Prediksi Audio: {predicted_label} \n\n Saran: {advice} \n\n Asisten Pintar: {response_text}")

st.subheader("Minta Saran Asisten Pintar")
response_container = st.container()
with response_container:
    for resp in st.session_state['responses']:
        st.markdown(f"<div style='border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; background-color: #31363F;'>{resp}</div>", unsafe_allow_html=True)
        st.markdown("---")