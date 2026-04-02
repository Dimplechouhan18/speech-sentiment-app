import numpy as np
import librosa
import pickle
import gradio as gr
import tensorflow as tf

# Load Model & Encoder
model = tf.keras.models.load_model("final_model.h5")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Feature Extraction
def extract_features(file):
    audio, sr = librosa.load(file, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    feature = np.hstack([
        np.mean(mfcc.T, axis=0),
        np.mean(chroma.T, axis=0)
    ])
    return feature

# Prediction
def predict_emotion(audio):
    features = extract_features(audio)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    label = le.inverse_transform([predicted_class])[0]
    return f"Prediction: {label} | Confidence: {confidence:.2f}"

# Gradio Interface
gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="🎙️ Audio Sentiment Analyzer",
    description="Upload audio file to detect sentiment (Positive / Negative / Neutral)"
).launch()
