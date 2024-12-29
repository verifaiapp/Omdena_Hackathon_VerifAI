import streamlit as st
from PIL import Image
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import soundfile as sf  # Import soundfile
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import string
from nltk.corpus import stopwords

class DeepfakeImageDetector:
    # Initialize the detector with the model we trained
    def __init__(self, model_dir, label_map):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
        self.model = AutoModelForImageClassification.from_pretrained(model_dir)
        self.label_map = label_map

    # Preprocess the uploaded image for the model
    def preprocess_image(self, image):
        image = image.convert("RGB")  # Ensure the image is in RGB format
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return inputs

    # Function for verification / prediction
    def predict(self, image):
        inputs = self.preprocess_image(image)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).squeeze().numpy()
            predicted_label_id = logits.argmax(dim=-1).item()
            predicted_label = self.label_map[predicted_label_id]
        return predicted_label, probabilities[predicted_label_id]

class AITextDetector:
    # Initialize the text detector with the model we trained
    def __init__(self, model_path):
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)
        self.stop_words = set(stopwords.words('english'))

    # Preprocess the text by removing punctuation, converting to lowercase, and removing stopwords
    def preprocess_text(self, text):
        text = text.lower()
        text = ''.join([ch for ch in text if ch not in string.punctuation])
        words = text.split()
        clean_words = [word for word in words if word not in self.stop_words]
        return ' '.join(clean_words)

    # Function for classification / verification
    def classify_text(self, input_text):
        preprocessed_text = self.preprocess_text(input_text)
        pred = self.model.predict([preprocessed_text])[0]
        classifier = self.model.named_steps['classifier']
        vectorizer = self.model.named_steps['vectorizer']
        tfidf = self.model.named_steps['tfidf']
        vectorized_text = vectorizer.transform([preprocessed_text])
        tfidf_text = tfidf.transform(vectorized_text)
        pred_prob = classifier.predict_proba(tfidf_text)[0]
        confidence = np.max(pred_prob)
        label = "AI-generated" if pred == 1 else "Human-generated"
        return label, confidence

class DeepfakeAudioDetector:
    # Initialize the audio detector with the model we trained
    def __init__(self, model_path):
        self.model = load_model(model_path)

    # Preprocessing the audio to a format that the model expects
    def audio_to_spectrogram(self, audio_path, output_path, spectrogram_size=(224, 224), dpi=300):
        y, sr = librosa.load(audio_path, sr=None, mono=True)  # Load audio with original sampling rate

        # Generate Mel spectrogram
        spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=256, n_mels=128
        )

        # Convert to decibel scale and normalize
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram_db = (spectrogram_db + 80) / 80

        # Plot and save the spectrogram
        plt.figure(figsize=(8, 6), dpi=dpi)
        plt.axis('off')
        librosa.display.specshow(
            spectrogram_db, sr=sr, x_axis='time', y_axis='mel', hop_length=256, cmap='inferno'
        )
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()

        # Resize the saved image
        img = Image.open(output_path)
        img = img.resize(spectrogram_size, Image.LANCZOS)
        img.save(output_path)

    # Preprocess the spectrogram image for the model
    def preprocess_image(self, image_path, target_size=(224, 224)):
        img = load_img(image_path, target_size=target_size)
        img = img.convert('RGB')
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        return np.expand_dims(img_array, axis=0)

    # Function for verifying / classifying
    def predict_audio_label(self, audio_path, target_size=(224, 224), spectrogram_path='temp_spectrogram.png'):
        self.audio_to_spectrogram(audio_path, spectrogram_path, spectrogram_size=target_size)
        image_array = self.preprocess_image(spectrogram_path, target_size=target_size)
        prediction = self.model.predict(image_array)
        label = "REAL" if prediction[0][0] > 0.5 else "FAKE"
        confidence = prediction[0][0] if label == "REAL" else 1 - prediction[0][0]
        return label, confidence

class MaliciousLinkDetector:
    # Initialize the phishing detector with the model we trained
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    # Function for classifying
    def check_link_validity(self, link):
        inputs = self.tokenizer(link, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()

        id2label = {0: "SAFE", 1: "DANGEROUS"}
        label = id2label[predicted_class]
        return label, confidence
