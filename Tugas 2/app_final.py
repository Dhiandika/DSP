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
import plotly.graph_objs as go
import plotly.express as px

# Load environment variables
load_dotenv()

# Configure TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define class labels
kelas_label = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

# Load the trained model
model_path = 'model/Model RNN.h5'
logger.info('Loading model from %s', model_path)
loaded_model = load_model(model_path)
logger.info('Model loaded successfully')

# Initialize session state
if 'model' not in st.session_state:
    genai.configure(api_key=os.getenv('API_KEY'))
    st.session_state.model = genai.GenerativeModel("gemini-1.5-flash-latest")
    st.session_state.convo = st.session_state.model.start_chat(history=[])
    logger.info('Generative model initialized')

if 'responses' not in st.session_state:
    st.session_state['responses'] = []

if 'history' not in st.session_state:
    st.session_state['history'] = []

# Feature extraction function
def extract_features(file_path, max_length=300, n_mfcc=13):
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
def plot_waveform(y, sr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.linspace(0, len(y) / sr, len(y)), y=y, mode='lines'))
    fig.update_layout(title='Waveform', xaxis_title='Time (s)', yaxis_title='Amplitude')
    return fig

# Spectrogram plotting function
def plot_spectrogram(y, sr):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig = px.imshow(D, x=librosa.frames_to_time(np.arange(D.shape[1])), y=librosa.fft_frequencies(sr=sr), aspect='auto', origin='lower', color_continuous_scale='Viridis')
    fig.update_layout(title='Spectrogram', xaxis_title='Time (s)', yaxis_title='Frequency (Hz)')
    return fig

# Mel-spectrogram plotting function
def plot_mel_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    return fig

# Prediction function
def predict_audio_category(model, mfccs):
    mfccs = np.expand_dims(mfccs, axis=0)
    predictions = model.predict(mfccs)
    return predictions

# Streamlit app layout
st.title('Baby Cry Prediction App')

# Adding description
st.markdown("""
<div style='background-color: #31363F; padding: 10px; border-radius: 10px;'>
    <h3 style='color: #4CAF50;'>Selamat Datang di Aplikasi Prediksi Tangisan Bayi</h3>
    <p>Aplikasi ini menggunakan model pembelajaran mesin untuk mendeteksi jenis tangisan bayi berdasarkan audio yang Anda unggah. 
    Ini bisa membantu orang tua untuk memahami kebutuhan bayi mereka dengan lebih baik.</p>
</div>
""", unsafe_allow_html=True)

# Upload audio file
uploaded_file = st.file_uploader("Pilih file audio...", type=["wav"])

if uploaded_file:
    logger.info('Audio file uploaded: %s', uploaded_file.name)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio_path = os.path.join(temp_dir, "temp.wav")
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        y, sr = librosa.load(temp_audio_path, sr=None)
        st.audio(temp_audio_path)
        
        # Extract MFCC features
        mfccs = extract_features(temp_audio_path)
        
        # Visualization Tabs
        tab1, tab2, tab3 = st.tabs(["Waveform", "Spectrogram", "Mel-spectrogram"])
        
        with tab1:
            st.subheader("Waveform")
            waveform_fig = plot_waveform(y, sr)
            st.plotly_chart(waveform_fig)
        
        with tab2:
            st.subheader("Spectrogram")
            spectrogram_fig = plot_spectrogram(y, sr)
            st.plotly_chart(spectrogram_fig)
        
        with tab3:
            st.subheader("Mel-spectrogram")
            mel_spectrogram_fig = plot_mel_spectrogram(y, sr)
            st.pyplot(mel_spectrogram_fig)
        
        # Predictions
        predictions = predict_audio_category(loaded_model, mfccs)
        predicted_label = kelas_label[np.argmax(predictions)]
        
        st.subheader("Prediction Results")
        st.write(f"Predicted Label: {predicted_label}")
        
        advice = ""
        if predicted_label == "belly_pain":
            advice = "Bayi Anda mungkin mengalami sakit perut."
        elif predicted_label == "burping":
            advice = "Bayi Anda mungkin perlu bersendawa."
        elif predicted_label == "discomfort":
            advice = "Bayi Anda mungkin merasa tidak nyaman."
        elif predicted_label == "hungry":
            advice = "Bayi Anda mungkin lapar."
        elif predicted_label == "tired":
            advice = "Bayi Anda mungkin mengantuk."
        
        st.write(f"Advice: {advice}")
        
        # Generate AI response
        if st.button("Minta Saran Asisten Pintar"):
            response = st.session_state.convo.send_message(f"Apa yang harus dilakukan jika {advice.lower()}?")
            response_text = response.text
            
            st.session_state.responses.append({
                "Prediksi Audio": predicted_label,
                "Saran": advice,
                "Asisten Pintar": response_text
            })
            st.subheader("Asisten Pintar")
            for i, response in enumerate(st.session_state.responses):
                with st.container():
                    st.markdown(f"""
                    <div style='background-color: #373A40; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1);'>
                        <h4 style='color: #4CAF50;'>Response {i+1}</h4>
                        <p><strong>Prediksi Audio:</strong> {response["Prediksi Audio"]}</p>
                        <p><strong>Saran:</strong> {response["Saran"]}</p>
                        <p><strong>Asisten Pintar:</strong> {response["Asisten Pintar"]}</p>
                    </div>
                    """, unsafe_allow_html=True)

        # Show prediction distribution
        st.subheader("Prediction Distribution")
        y_probs = predictions[0]
        prob_fig = px.bar(x=kelas_label, y=y_probs, labels={'x': 'Classes', 'y': 'Probability'})
        prob_fig.update_layout(title='Prediction Probabilities', xaxis_title='Class', yaxis_title='Probability')
        st.plotly_chart(prob_fig)

        # Model comparison
        model_paths = ['model/Model CNN.h5', 'model/Model RNN.h5']
        models = [load_model(path) for path in model_paths]
        model_names = ['Model 1 CNN', 'Model 2 RNN']

        predictions = []
        for model in models:
            mfccs = extract_features(temp_audio_path)
            predictions.append(predict_audio_category(model, mfccs))

        st.subheader("Model Comparison")
        comparison_fig = px.bar(x=model_names, y=[np.argmax(pred) for pred in predictions], labels={'x': 'Model', 'y': 'Predicted Class'})
        comparison_fig.update_layout(title='Model Comparison', xaxis_title='Model', yaxis_title='Predicted Class')
        st.plotly_chart(comparison_fig)


