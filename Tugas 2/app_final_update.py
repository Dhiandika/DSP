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
from textblob import TextBlob
import pandas as pd
import sounddevice as sd
import soundfile as sf  # for saving audio file

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
model_path = 'model/Model RNN (1).h5'
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

if 'prediksi_statistik' not in st.session_state:
    st.session_state['prediksi_statistik'] = {label: 0 for label in kelas_label}

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

# Zero Crossing Rate plotting function
def plot_zcr(y, sr):
    zcr = librosa.feature.zero_crossing_rate(y)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.linspace(0, len(y) / sr, len(zcr[0])), y=zcr[0], mode='lines'))
    fig.update_layout(title='Zero Crossing Rate', xaxis_title='Time (s)', yaxis_title='ZCR')
    return fig

# Chroma Feature plotting function
def plot_chroma(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='Chroma Feature')
    return fig

# Tonnetz plotting function
def plot_tonnetz(y, sr):
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(tonnetz, y_axis='tonnetz', x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='Tonnetz')
    return fig

# RMS Energy plotting function
def plot_rms(y, sr):
    rms = librosa.feature.rms(y=y)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.linspace(0, len(y) / sr, len(rms[0])), y=rms[0], mode='lines'))
    fig.update_layout(title='RMS Energy', xaxis_title='Time (s)', yaxis_title='RMS Energy')
    return fig

# Prediction function
def predict_audio_category(model, mfccs):
    mfccs = np.expand_dims(mfccs, axis=0)
    predictions = model.predict(mfccs)
    return predictions

# Sentiment analysis function
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Save results function
def save_results_to_csv():
    df = pd.DataFrame(st.session_state.responses)
    df.to_csv('hasil_prediksi.csv', index=False)

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

# Simple Help/Guide Section (User Instructions)
st.markdown("""
### Panduan Penggunaan Aplikasi:
- **Upload File Audio**: Pilih opsi "Upload file audio" untuk memprediksi tangisan bayi berdasarkan file audio yang sudah ada.
- **Rekam Audio**: Tekan tombol "Rekam" untuk merekam audio langsung dari mikrofon Anda. Durasi rekaman default adalah 10 detik.
- **Minta Saran Asisten Pintar**: Setelah prediksi, Anda dapat meminta saran dari asisten pintar tentang langkah selanjutnya.
- **Simpan Hasil**: Simpan hasil prediksi ke dalam file CSV untuk referensi atau analisis lebih lanjut.
- **Clear Session State**: Hapus semua respons dan kembalikan aplikasi ke kondisi awal.
""")

# Clear Session State button
if st.button("Clear Session State"):
    st.session_state['responses'] = []
    st.session_state['history'] = []
    st.session_state['prediksi_statistik'] = {label: 0 for label in kelas_label}
    st.success("Session state has been cleared.")

# Display general information about baby cries
st.subheader("Informasi Jenis Tangisan Bayi")
st.markdown("""
- **Sakit Perut (belly_pain)**: Bayi menangis karena sakit perut.
- **Perlu Bersendawa (burping)**: Bayi menangis karena perlu bersendawa.
- **Tidak Nyaman (discomfort)**: Bayi menangis karena merasa tidak nyaman.
- **Lapar (hungry)**: Bayi menangis karena lapar.
- **Lelah (tired)**: Bayi menangis karena lelah.
""")

# Audio File Upload and Recording Section
uploaded_file = st.file_uploader("Upload File Audio (WAV format)", type=["wav"])

if uploaded_file is not None or 'recorded_audio' in st.session_state:
    if uploaded_file is not None:
        temp_audio_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
            temp_audio_file.write(uploaded_file.read())
            temp_audio_path = temp_audio_file.name
    else:
        temp_audio_path = st.session_state['recorded_audio']

    y, sr = librosa.load(temp_audio_path, sr=None)
    st.audio(temp_audio_path)

    # Extract MFCC features
    mfccs = extract_features(temp_audio_path)

    # Visualization Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Waveform", "Spectrogram", "Mel-spectrogram", "Zero Crossing Rate", "Chroma", "Tonnetz","RMS Energy"])

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

    with tab4:
        st.subheader("Zero Crossing Rate")
        zcr_fig = plot_zcr(y, sr)
        st.plotly_chart(zcr_fig)

    with tab5:
        st.subheader("Chroma Feature")
        chroma_fig = plot_chroma(y, sr)
        st.pyplot(chroma_fig)

    with tab6:
        st.subheader("Tonnetz")
        tonnetz_fig = plot_tonnetz(y, sr)
        st.pyplot(tonnetz_fig)

    with tab7:
        st.subheader("RMS Energy")
        rms_fig = plot_rms(y, sr)
        st.plotly_chart(rms_fig)

    # Predictions
    predictions = predict_audio_category(loaded_model, mfccs)
    predicted_label = kelas_label[np.argmax(predictions)]
    st.subheader("Hasil Prediksi")
    st.write(f"Prediksi Label: {predicted_label}")
    advice = ""
    if predicted_label == "belly_pain":
        advice = "bayi Anda mungkin merasa sakit perut. Cobalah untuk memeriksa dan memberikan pijatan lembut pada perut bayi."
    elif predicted_label == "burping":
        advice = "bayi Anda mungkin perlu bersendawa. Angkat bayi Anda dan bantu mereka untuk bersendawa."
    elif predicted_label == "discomfort":
        advice = "bayi Anda mungkin merasa tidak nyaman. Periksa popok mereka atau pastikan mereka dalam posisi yang nyaman."
    elif predicted_label == "hungry":
        advice = "bayi Anda mungkin lapar. Cobalah untuk memberi mereka makan."
    elif predicted_label == "tired":
        advice = "bayi Anda mungkin lelah. Cobalah untuk menidurkan mereka."
    st.write(f"Saran: {advice}")

    if st.button("Minta Saran Asisten Pintar"):
        response = st.session_state.convo.send_message(f"Apa yang harus dilakukan jika {advice.lower()}?")
        response_text = response.text
        sentiment_polarity, sentiment_subjectivity = analyze_sentiment(response_text)
        st.session_state.responses.append({
            "Prediksi Audio": predicted_label,
            "Saran": advice,
            "Asisten Pintar": response_text,
            "Sentimen": sentiment_polarity,
            "Subjektivitas": sentiment_subjectivity
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
                    <p><strong>Sentimen:</strong> {'Positif' if response["Sentimen"] > 0 else 'Negatif'}</p>
                    <p><strong>Subjektivitas:</strong> {response["Subjektivitas"]}</p>
                </div>
                """, unsafe_allow_html=True)

    # Show prediction distribution
    st.subheader("Distribusi Prediksi")
    y_probs = predictions[0]
    prob_fig = px.bar(x=kelas_label, y=y_probs, labels={'x': 'Kelas', 'y': 'Probabilitas'})
    prob_fig.update_layout(title='Probabilitas Prediksi', xaxis_title='Kelas', yaxis_title='Probabilitas')
    st.plotly_chart(prob_fig)

    # Model comparison
    model_paths = ['model/Model CNN (1).h5', 'model/Model RNN (1).h5']
    models = [load_model(path) for path in model_paths]
    model_names = ['Model 1 CNN', 'Model 2 RNN']
    predictions = []
    for model in models:
        mfccs = extract_features(temp_audio_path)
        predictions.append(predict_audio_category(model, mfccs))
    st.subheader("Perbandingan Model")
    comparison_fig = px.bar(x=model_names, y=[np.argmax(pred) for pred in predictions], labels={'x': 'Model', 'y': 'Kelas yang Diprediksi'})
    comparison_fig.update_layout(title='Perbandingan Model', xaxis_title='Model', yaxis_title='Kelas yang Diprediksi')
    st.plotly_chart(comparison_fig)

    # Update prediction statistics
    st.session_state['prediksi_statistik'][predicted_label] += 1

# Audio Recording Section
if st.button("Rekam Audio"):
    duration = 10  # seconds
    with st.spinner(f"Merekam {duration} detik audio ..."):
        recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float64')
        sd.wait()  # Wait until recording is finished
    temp_dir = tempfile.mkdtemp()
    temp_audio_path = os.path.join(temp_dir, 'temp.wav')
    sf.write(temp_audio_path, recording, 44100)
    st.audio(temp_audio_path, format='audio/wav')
    st.session_state['recorded_audio'] = temp_audio_path

# Save results button
if st.button("Simpan Hasil"):
    save_results_to_csv()
    st.success("Hasil berhasil disimpan ke hasil_prediksi.csv")
