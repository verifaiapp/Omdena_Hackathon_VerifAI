import streamlit as st
from PIL import Image
import os
import helpers
import gdown
import nltk
nltk.download('stopwords')

#  Google Drive folder model IDs for AI detectors
MODEL_FOLDER_ID = {
    "ai_text_detector": "1N1EkWbTd8S3UiicvNM1eI8dWn21XPH2T",
    "audio_deepfake_detector": "1l818hmq09HkK-Tl9ziRw3FUrZDwOrL8t",
    "image_deepfake_detector": "1CkhLsnhRaCTMK9frwNdMIWWgOq8-OedX",
    "malicious_link_detector": "1Bhmcb6TPZlDKpBjS8xA4tdz_awtE2eup",
}

# Google Drive folder IDs for test files
IMAGE_TEST_FILES_FOLDER_ID = "10_ElyRhMRkV2sDXRt3JeBwLOadRXhsZY"
AUDIO_TEST_FILES_FOLDER_ID = "1X0Dl4o2Ecd5Aez3OPecs0ASeCeCzkQG-"

def download_models(model_folder_ids):
    for model_name, folder_id in model_folder_ids.items():
        model_path = os.path.join("models", model_name)

        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            print(f"Downloading {model_name} model from Google Drive...")
            gdown.download_folder(id=folder_id, output=model_path, quiet=False)\
        
        # Specific verification for the AI text detector model
        if model_name == "ai_text_detector":
            expected_file = "ai_text_detector_model.pkl"
            model_file = os.path.join(model_path, expected_file)
            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"Model file {model_file} not found after download. "
                    f"Ensure the file is available in the Google Drive folder: {folder_id}"
                )
            else:
                print(f"Verified: {model_file} is ready to use.")

        print(f"Contents of {model_path}: {os.listdir(model_path)}")

# Initialize Streamlit with custom settings
def init_streamlit():
    st.set_page_config(
        page_title="VerifAI: Where AI Meets Authentication",
        page_icon=os.path.join("images", "Logo.png"),
        layout="wide",
        initial_sidebar_state="auto"
    )
    
    # Adding custom CSS for styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

        *, *::before, *::after {
            font-family: 'Poppins', sans-serif;
        }

        button[data-baseweb="tab"] {
            font-size: 24px;
            margin: 0;
            width: 100%;
        }

        button[title="View fullscreen"] { visibility: hidden; }
        .reportview-container { margin-top: -2em; }
        #MainMenu { visibility: hidden; }
        .stDeployButton { display: none; }
        footer { visibility: hidden; }
        #stDecoration { display: none; }

        .stApp {
            padding-top: 20px;
        }

        .stProgress > div > div > div {
            background-color: #4284f2 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Self-created navbar
def display_navbar():
    # VerifAI logo
    image_path = os.path.join("images", "1.svg")
    st.image(image_path, use_container_width=True)

    st.markdown("""
        <div style="text-align: center;">
            <h1 style="font-size: 2.5em; margin-bottom: 0.1em;">VerifAI: Where AI Meets Authentication</h1>
            <h3 style="font-weight: normal; color: gray; margin-top: -0.8em; margin-bottom: 0.7em;">Spot fakes and trust with confidence—powered by AI algorithms.</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
            .stTabs [role="tab"] {
                font-size: 1.5em !important;
                font-weight: bold !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Tabs for different tools each one is labeled with its purpose respectively
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Deepfake Audio Detector",
         "Deepfake Image Detector",
         "Phishing Link Detector",
         "Extras"]
    )

    return tab1, tab2, tab3, tab4

# Experimental AI text detector
def extras_tab(detector):
    st.title("AI Text Detector (Experimental)")
    st.write("Type your own text, upload a text file, or select from the sample texts to check if it is **Human-generated** or **AI-generated**.")
    # AI text detectors should be taken with a grain of salt
    st.warning(
        "Disclaimer: The AI Text Detector is in the experimental phase and may not produce accurate results. "
        "Use it cautiously and consider it as a supplementary tool rather than definitive."
    )
    # Users can select sample texts that are provided for testing
    sample_texts = [
        ("HUMAN-GENERATED", "Yesterday, I walked to the park to clear my head. The air was crisp, and the leaves crunched underfoot. I couldn’t help but feel a sense of nostalgia, remembering how my siblings and I used to play here when we were kids. It’s funny how places can hold so many memories."),
        ("AI-GENERATED", "The park represents a tranquil environment where individuals can contemplate life's deeper meaning and connect with the natural world. The gentle rustling of leaves, combined with the soft caress of a breeze, crafts an atmosphere of relaxation and mental clarity. Parks play a vital role in promoting social interaction, encouraging well-being, and maintaining the ecological balance of urban areas.")
    ]

    # Allowing users to choose input method
    input_choice = st.radio("Choose input method:", ("Type text", "Upload text file", "Select sample text"))

    user_text = ""

    # Type
    if input_choice == "Type text":
        user_text = st.text_area("Enter text:", height=300, placeholder="Start typing your text here...")
    # Upload
    elif input_choice == "Upload text file":
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"], key="file_uploader_text")
        if uploaded_file is not None:
            user_text = uploaded_file.read().decode("utf-8")
    # Select from sample texts
    elif input_choice == "Select sample text":
        selected_sample = st.selectbox("Select a sample text:", options=[f"{label}: {text[:50]}..." for label, text in sample_texts])
        if selected_sample:
            selected_label = selected_sample.split(": ", 1)[0]
            user_text = next(text for label, text in sample_texts if label == selected_label)
            st.text_area("Selected Sample Text:", value=user_text, height=200, disabled=True)

    # Click to verify
    if st.button("Verify Text"):
        if user_text.strip():
            try:
                label, confidence = detector.classify_text(user_text)

                color = "green" if label == "Human-generated" else "red"
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <h3 style="display: inline-block; margin-left: 30px;">
                            Result: <span style="color: {color};">{label}</span>
                        </h3>
                        <p style="display: inline-block; font-size: 20px; margin-left: -6px;">
                            Confidence: {confidence*100:.2f}%
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error during classification: {e}")
        else:
            st.error("Please enter, upload, or select some text to verify.")

# Deepfake URL detector
def maliciouslink_detection_navbar(maliciouslink_detector):
    st.title("Phishing Detection")
    st.write("Enter a URL or select from the sample URLs, and the model will verify it as **SAFE** or **DANGEROUS**.")

    # Users can select sample URLs to test the model
    sample_urls = [
        ("DANGEROUS", "https://míсrоsоft-update-check.net"),
        ("DANGEROUS", "https://fácebook-login.com"),
        ("DANGEROUS", "https://аpple-security-check.com"),
        ("SAFE", "https://www.facebook.com/"),
        ("SAFE", "https://www.microsoft.com/"),
        ("SAFE", "https://www.apple.com/"),
        ("SAFE", "https://www.omdena.com/")
    ]

    # Input choices 
    input_choice = st.radio("Choose input method:", ("Type your own URL", "Select a sample URL"))

    user_url = ""

    # Type
    if input_choice == "Type your own URL":
        user_url = st.text_input("Enter URL:", placeholder="https://www.enter-your-link-here.com")
    # Select from sample URLs
    elif input_choice == "Select a sample URL":
        selected_sample = st.selectbox("Select a sample URL:", options=[f"{label}: {url}" for label, url in sample_urls])
        if selected_sample:
            user_url = selected_sample.split(": ", 1)[1]
            st.text_input("Selected URL:", value=user_url, disabled=True)

    # Verify the URL
    if st.button("Verify URL"):
        if user_url:
            label, confidence = maliciouslink_detector.check_link_validity(user_url)

            color = "green" if label == "SAFE" else "red"

            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h3 style="display: inline-block; margin-left: 20px;">Result: <span style="color: {color};">{label}</span></h3>
                    <p style="display: inline-block; font-size: 20px; margin-left: -6px;">Confidence: {confidence*100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True)
        else:
            st.error("Please enter or select a URL to verify.")

# Deepfake audio detector
def deepfake_audio_detector_menu(detector, test_files):
    st.title("Audio Deepfake Detector")
    st.write("Upload an audio file, or choose from the test files provided, and the AI will verify it as **Real** or **Fake**.")

    # Users can select options to upload, test, or type
    input_choice = st.radio("Choose input method:", ("Upload audio file", "Use test file"))

    selected_file_path = None

    # Upload your own audio file
    if input_choice == "Upload audio file":
        uploaded_audio = st.file_uploader("Upload an Audio File", type=["wav", "mp3"], key="audio_upload")
        if uploaded_audio is not None:
            selected_file_path = os.path.join("temp_uploaded_audio.mp3")
            with open(selected_file_path, "wb") as f:
                f.write(uploaded_audio.read())
            st.audio(selected_file_path, format="audio/mp3", start_time=0)

    # Use sample audio files
    elif input_choice == "Use test file":
        test_files_info = test_files["audio_files"]
        test_files_folder = test_files_info["folder"]
        test_files_list = test_files_info["files"]

        selected_test_file = st.selectbox("Select a test file:", test_files_list, key="audio_test_select")
        if selected_test_file:
            selected_file_path = os.path.join(test_files_folder, selected_test_file)
            st.audio(selected_file_path, format="audio/mp3", start_time=0)

    classify_audio_key = "classify_audio_upload" if input_choice == "Upload audio file" else "classify_audio_test"

    # Verify
    if st.button("Verify Audio", key=classify_audio_key):
        if selected_file_path:
            try:
                predicted_label, confidence = detector.predict_audio_label(selected_file_path)
                color = "green" if predicted_label.upper() == "REAL" else "red"

                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <h3 style="display: inline-block; margin-left: 30px;">
                            Result: <span style="color: {color};">{predicted_label}</span>
                        </h3>
                        <p style="display: inline-block; font-size: 20px; margin-left: -6px;">
                            Confidence: {confidence*100:.2f}%
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"An error occurred while processing the audio file: {e}")
        else:
            st.error("Please select or upload an audio file to verify.")

# Deepfake image detector
def deepfake_image_detector_menu(detector, test_files):
    st.title("Image Deepfake Detector")
    st.write("Upload an image file, or choose from the test files provided, and the AI will verify it as **Real** or **Fake**.")

    # Let the user choose how they want to provide input
    input_choice = st.radio("Choose input method:", ("Upload image file", "Use test file"))

    selected_file_path = None

    # Uplaod
    if input_choice == "Upload image file":
        uploaded_image = st.file_uploader("Upload an Image File", type=["png", "jpg", "jpeg"])
        if uploaded_image is not None:
            selected_file_path = os.path.join("temp_uploaded_image.jpg")
            with open(selected_file_path, "wb") as f:
                f.write(uploaded_image.read())

            st.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{image_to_base64(Image.open(selected_file_path))}" alt="Uploaded Image" width="400"/>
                    <p><strong>Uploaded Image</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Try using test files
    elif input_choice == "Use test file":
        folder_path = test_files["image_files"]["folder"]
        test_files_list = test_files["image_files"]["files"]
        selected_test_file = st.selectbox("Select a test file:", test_files_list)
        if selected_test_file:
            selected_file_path = os.path.join(folder_path, selected_test_file)
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{image_to_base64(Image.open(selected_file_path))}" alt="Test Image" width="400"/>
                    <p><strong>Selected Test Image</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Verify
    if st.button("Verify Image"):
        if selected_file_path:
            try:
                image = Image.open(selected_file_path)

                predicted_label, confidence = detector.predict(image)

                color = "green" if predicted_label.upper() == "REAL" else "red"

                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <h3 style="display: inline-block; margin-left: 30px;">
                            Result: <span style="color: {color};">{predicted_label}</span>
                        </h3>
                        <p style="display: inline-block; font-size: 20px; margin-left: -6px;">
                            Confidence: {confidence*100:.2f}%
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"An error occurred while processing the image file: {e}")
        else:
            st.error("Please select or upload an image file to verify.")

# Converts a PIL Image to a Base64-encoded string
def image_to_base64(image):
    import io
    import base64

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    # Returns (str) The Base64-encoded string representation of the image
    return base64.b64encode(buffered.getvalue()).decode()

# Save to cache to prevent long download times or reloading of models
@st.cache_resource
def load_maliciouslink_detector():
    import os
    model_path = os.path.join("models", "malicious_link_detector")
    return helpers.MaliciousLinkDetector(model_path)

@st.cache_resource
def load_audio_detector():
    model_path = os.path.join("models", "audio_deepfake_detector", "cnn_audio_deepfake_best_model.h5")
    return helpers.DeepfakeAudioDetector(model_path=model_path)

@st.cache_resource
def load_image_detector():
    model_dir = os.path.join("models", "image_deepfake_detector")
    label_map = {0: "Real", 1: "Fake"}
    return helpers.DeepfakeImageDetector(model_dir=model_dir, label_map=label_map)

@st.cache_resource
def load_text_detector():
    model_path = os.path.join("models", "ai_text_detector", "ai_text_detector_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file {model_path} not found. Ensure the file is properly downloaded and placed in the correct directory."
        )

    print(f"Loading text detector model from {model_path}...")
    return helpers.AITextDetector(model_path=model_path)

@st.cache_resource
def download_test_files(folder_id, local_folder_name):
    """Download test files from Google Drive."""
    folder_path = os.path.join("models", local_folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"Downloading {local_folder_name} test files from Google Drive...")
        gdown.download_folder(id=folder_id, output=folder_path, quiet=False)
    return folder_path

@st.cache_resource
def preload_test_files():
    """Preload test files from Google Drive."""
    audio_folder = download_test_files(AUDIO_TEST_FILES_FOLDER_ID, "audio_test_files")
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith((".wav", ".mp3"))]

    image_folder = download_test_files(IMAGE_TEST_FILES_FOLDER_ID, "image_test_files")
    image_files = [f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

    return {
        "audio_files": {"folder": audio_folder, "files": audio_files},
        "image_files": {"folder": image_folder, "files": image_files},
    }

def main():
    """
    Main function to initialize the Streamlit app, load models, 
    and display the navigation tabs with their respective functionalities.
    """
    init_streamlit()

    # Download required models to the local directory if not already present
    download_models(MODEL_FOLDER_ID)

    # Preload test files to use in the app for demonstration or testing
    test_files = preload_test_files()

    # Set up navigation tabs for different detection functionalities
    tab1, tab2, tab3, tab4 = display_navbar()

    # Load the pre-trained models for different detection tasks
    audio_detector = load_audio_detector()
    image_detector = load_image_detector()
    text_detector = load_text_detector()
    maliciouslink_detector = load_maliciouslink_detector()

    # Define the content and functionality for each navigation tab
    with tab1:
        deepfake_audio_detector_menu(audio_detector, test_files)
    with tab2:
        deepfake_image_detector_menu(image_detector, test_files)
    with tab3:
        maliciouslink_detection_navbar(maliciouslink_detector)
    with tab4:
        extras_tab(text_detector)

if __name__ == "__main__":
    main()
