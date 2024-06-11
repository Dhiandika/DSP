import streamlit as st
import streamlit.components.v1 as components
import asyncio
import websockets
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import os
import tempfile
from sklearn.preprocessing import LabelEncoder

# Load the models
rnn_model = load_model('model/Model RNN.h5')
cnn_model = load_model('model/Model CNN.h5')

# Function to create label_classes.npy if it does not exist
def create_label_classes():
    labels = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    np.save('label_classes.npy', label_encoder.classes_)
    return label_encoder

# Load the label encoder
if not os.path.exists('label_classes.npy'):
    label_encoder = create_label_classes()
else:
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('label_classes.npy', allow_pickle=True)

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

async def websocket_handler(websocket, path):
    async for message in websocket:
        if message == "audio":
            # Simulate processing an uploaded audio file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                audio_path = temp_audio.name

            mfccs = extract_features(audio_path, max_length=300)
            mfccs_cnn = np.expand_dims(mfccs, axis=-1)  # Shape: (1, 13, 300, 1)

            # Predict with both models
            rnn_predictions = rnn_model.predict(mfccs)
            cnn_predictions = cnn_model.predict(mfccs_cnn)

            rnn_predicted_label = label_encoder.inverse_transform([np.argmax(rnn_predictions)])[0]
            cnn_predicted_label = label_encoder.inverse_transform([np.argmax(cnn_predictions)])[0]

            response = f"RNN: {rnn_predicted_label}, CNN: {cnn_predicted_label}"
            await websocket.send(response)
        else:
            response = f"Support: You said '{message}'"
            await websocket.send(response)

async def start_server():
    async with websockets.serve(websocket_handler, "localhost", 8501):
        await asyncio.Future()  # run forever

st.title("Smart Assistant Chat")
components.html(open("chatbox.html").read(), height=600)

# Run the WebSocket server in an asyncio loop
if 'server' not in st.session_state:
    st.session_state.server = asyncio.run(start_server())
