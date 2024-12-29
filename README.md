# VerifAI: Where AI Meets Authentication

### Streamlit Link: [VerifAI App](https://verifaiapp.streamlit.app)  
### Video Demo: [Infomercial](https://youtu.be/t7jWHUiVTco)

## The Story Behind VerifAI
**[VerifAI](https://verifaiapp.streamlit.app)** is a passion project by [Jen Patrick Nataba](https://www.linkedin.com/in/cytojen/) and [John Ferry Lagman](https://www.linkedin.com/in/thatjohnlagman/).
With the sheer volume of digital content today, it’s becoming harder to tell what’s real and what’s not. For instance, a parent might share a heartwarming image online, believing it to be genuine, completely unaware that it’s been altered or is a deepfake. At the same time, many people fall victim to phishing scams, unknowingly clicking on malicious links that steal their personal information or drain their savings. These scenarios emphasize the urgent need for greater awareness and reliable tools to navigate a world filled with digital deception.  

But these aren't just stories; they are real challenges faced by millions of people. And it's these challenges that inspired **VerifAI**.

**VerifAI** started with a simple question: "How can we help people trust what they see, hear, and read?" From classrooms to social media feeds, VerifAI empowers users with cutting-edge tools to detect manipulated images, misleading audio, phishing links, and even AI-generated text. Designed with real-world scenarios in mind, our tools aim to make the digital space a safer and more credible place for everyone.

Our motivation is rooted in the experiences of vulnerable groups, particularly older users on social media and educators seeking academic integrity. By combining AI with purpose, VerifAI offers clarity in an increasingly complex digital age.

---

## Features
### 🔍 Deepfake Image Detector
- **Purpose**: Distinguish between real and AI-generated images.
- **Technology**: Utilizes state-of-the-art models from the Hugging Face library, enhanced with tailored datasets for real-world accuracy.
- **How It Works**: Preprocesses uploaded images, extracts features using pre-trained models, and classifies them as "Real" or "Fake" with confidence scores.

### 🎙️ Deepfake Audio Detector
- **Purpose**: Verify the authenticity of audio recordings.
- **Technology**: Converts audio to spectrograms and classifies them using a convolutional neural network (CNN).
- **How It Works**: Generates a Mel spectrogram, processes it through a deep learning model, and outputs a label ("Real" or "Fake") with confidence scores.

### 🌐 Phishing Link Detector
- **Purpose**: Protect users from malicious phishing links.
- **Technology**: Leverages Hugging Face's DistilBERT for sequence classification.
- **How It Works**: Tokenizes the input URL, processes it through a fine-tuned DistilBERT model, and classifies the link as "Safe" or "Dangerous" with confidence scores.

### 📝 Extras: AI Text Detector (Experimental)
- **Purpose**: Identify AI-generated text.
- **Technology**: Custom-trained machine learning model leveraging TF-IDF and Scikit-learn pipelines.
- **How It Works**: Preprocesses input text by removing stopwords and punctuation, tokenizes it, and predicts whether the text is human-written or AI-generated.

---

## Research Credits
This project acknowledges the invaluable contributions of the following research papers and frameworks, which guided the development and optimization of VerifAI's detection capabilities:

- [Audio Classification using CNN](https://ieeexplore.ieee.org/document/10072823): Provided insights into leveraging convolutional neural networks for audio analysis.
- [DistilBERT](https://arxiv.org/abs/1910.01108): A key resource for understanding and implementing lightweight transformer-based NLP models.
- [The DistilBERT Model: A Promising Approach to Improve Machine Reading Comprehension Models](https://www.researchgate.net/publication/374123033_The_DistilBERT_Model_A_Promising_Approach_to_Improve_Machine_Reading_Comprehension_Models): Inspired advancements in NLP tasks for phishing link detection.
- [Deep Learning Applications in Fake Content Detection](https://ieeexplore.ieee.org/document/10132112): Helped shape methodologies for identifying synthetic media and content.

---

## Usage

Navigate through the tabs in the app to:
- Upload images, audio, or text files for verification.
- Test phishing links with sample URLs or your own input.

### Interface Overview
- **Deepfake Audio Detector**: This feature allows users to upload or select an audio file for verification. Users can:
  - Upload their own audio recordings in formats like WAV or MP3.
  - Choose from provided test files for demonstration purposes.
  - View confidence scores indicating how likely the audio is real or fake.
  - See spectrogram visualizations of the uploaded audio, which represent the frequencies analyzed by the model.

- **Deepfake Image Detector**: This tool lets users verify the authenticity of images. Features include:
  - Uploading images in formats like PNG or JPEG for analysis.
  - Selecting from test image files provided in the application for quick testing.
  - Real-time processing of the image to determine if it is real or AI-generated.
  - A confidence score that quantifies the result, helping users make informed decisions.

- **Phishing Link Detector**: Aimed at preventing phishing scams, this tool enables users to:
  - Enter a URL manually to check its safety.
  - Select from a list of sample URLs (both safe and dangerous) for testing.
  - Receive a clear classification of the link as "Safe" or "Dangerous," along with the confidence percentage.
  - Get visual feedback through color-coded results, ensuring intuitive interpretation.

- **Extras (AI Text Detector)**: This experimental feature focuses on detecting AI-generated text. Users can:
  - Type or paste text directly into the input area for analysis.
  - Upload text files to process longer documents.
  - Choose from sample texts categorized as human- or AI-generated for comparison.
  - Receive detailed feedback on whether the text is classified as "Human-generated" or "AI-generated," with an accompanying confidence score.

---

## Project Structure
```plaintext
VerifAI/
├── app.py                # Main application file
├── helpers.py            # Helper functions and model classes
├── requirements.txt      # Required dependencies
├── models/               # Directory for downloaded models
├── images/               # Placeholder for logos and visuals
├── test_files/           # Placeholder for test data (images/audio)
└── README.md             # Project documentation
```

---

## Technologies Used
- **Streamlit**: Provides an interactive and user-friendly web interface.
- **TensorFlow**: Powers the deep learning models for audio and image detection.
- **Transformers by Hugging Face**: Backbone for NLP models in phishing and text detection.
- **Scikit-learn**: Used for custom machine learning pipelines in text analysis.
- **Librosa**: Processes audio files to generate spectrograms.
- **Matplotlib**: Visualizes spectrograms for audio analysis.

---

## Model Training Overview
Each detector in VerifAI is meticulously developed and trained using datasets tailored to real-world scenarios. The training process ensures that the models are robust, accurate, and capable of addressing their respective tasks effectively:

- **Deepfake Image Detection**: Developed and trained using a curated dataset of deepfake and real images, focusing on identifying subtle features that distinguish AI-generated visuals.
- **Deepfake Audio Detection**: Built with a custom convolutional neural network (CNN) trained on spectrograms generated from authentic and synthetic audio samples.
- **AI Text Detection**: Developed using a pipeline specifically trained on a carefully curated dataset of academic outputs and AI-generated texts. The training process involved feature extraction using TF-IDF and classification with a machine learning model.
- **Phishing Link Detection**: Fine-tuned on a dataset of malicious and legitimate URLs, ensuring effective classification of phishing attempts.

---

## Acknowledgments
Special thanks to:
- Educators, parents, and social media users who highlighted the need for this project.
- Open-source communities and datasets that made training these models possible.

---

## Contributing
We welcome contributions! Please:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

### Areas for Contribution
- Enhancing model accuracy with additional training data.
- Expanding functionality to include more types of deepfake detection.
- Improving the user interface for better accessibility.

---

## Future Plans
- **Multilingual Support**: Extend phishing detection to non-English languages.
- **Real-Time Detection**: Implement live detection capabilities for audio and video.
- **Community Datasets**: Allow users to contribute datasets for improving model accuracy.
- **Mobile Application**: Develop an Android/iOS app for on-the-go verification.

- **Real-Time Detection**: Implement live detection capabilities for audio and video.
- **Community Datasets**: Allow users to contribute datasets for improving model accuracy.
- **Mobile Application**: Develop an Android/iOS app for on-the-go verification.
