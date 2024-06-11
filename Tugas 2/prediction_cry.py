import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, LabelEncoder
import warnings
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import librosa
import librosa.display
import IPython.display as ipd

warnings.filterwarnings('ignore')

# Assuming Google Colab, otherwise comment out
from google.colab import drive
drive.mount('/content/drive')

dataset_dir = '/content/drive/MyDrive/Dataset/babycry/'

# Initialize lists for storing audio features and labels
audio_features = []
labels = []

# Function to extract features from audio files
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

# Step 2: Load Dataset and Extract Features
sample_files = {}
folder_counts = {}

# Iterate over each folder (category)
for category in os.listdir(dataset_dir):
    category_dir = os.path.join(dataset_dir, category)
    if os.path.isdir(category_dir):
        wav_files = [filename for filename in os.listdir(category_dir) if filename.endswith(".wav")]
        file_count = len(wav_files)
        folder_counts[category] = file_count
        if file_count > 0:
            sample_files[category] = os.path.join(category_dir, wav_files[0])
            for wav_file in wav_files:
                file_path = os.path.join(category_dir, wav_file)
                features = extract_features(file_path)
                audio_features.append(features)
                labels.append(category)

# Convert lists to numpy arrays
audio_features = np.array(audio_features)
labels = np.array(labels)

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(audio_features, labels_categorical, test_size=0.2, random_state=42)

# Normalize the features
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

# Determine input shape
n_features = X_train.shape[1]
input_shape = (n_features, 1)

# Reshape for CNN input
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Step 3: Build the Model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(n_features, 1, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(labels)), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Step 5: Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Step 6: Visualize Training Progress
# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
