🎤 Speech Sentiment Analysis System
AI-Powered Audio Emotion Detection

🚀 Overview

This project is a Deep Learning-based Speech Sentiment Analysis System that detects emotions from audio input.
It analyzes speech signals using advanced feature extraction techniques and a trained CNN model to classify emotions as Positive, Negative, or Neutral.
The system is deployed using Hugging Face Spaces with an interactive and user-friendly UI.

🎯 Key Features

🎤 Audio-based sentiment detection
🤖 Deep Learning model (CNN)
📊 Real-time emotion prediction
⚡ Fast and efficient processing
🎧 Supports audio file input
📈 Feature extraction using MFCC & Chroma
🎨 Clean and interactive UI (Gradio / Streamlit-based)

🧠 How It Works
User uploads or records an audio file
Audio is processed using Librosa
Features extracted:
MFCC (Mel Frequency Cepstral Coefficients)
Chroma Features
Features are scaled and reshaped
CNN model predicts emotion
System classifies output as:
😊 Positive
😐 Neutral
😠 Negative

📌 Output displays:
Predicted Sentiment
Confidence Score
Visual Feedback

Feedback
🏗️ Tech Stack
Python
TensorFlow / Keras
Librosa
NumPy
Pandas
Scikit-learn
Hugging Face Spaces
Gradio / Streamlit

📂 Project Structure
speech-sentiment-analysis/
│
├── app.py                  # Main application
├── model.h5               # Trained CNN model
├── scaler.pkl             # Feature scaler
├── speech_sentiment.py    # Training & preprocessing script
├── requirements.txt       # Dependencies
└── README.md              # Documentation

🌐 Live Demo

👉 https://huggingface.co/spaces/dimplechouhan/speech-sentiment
