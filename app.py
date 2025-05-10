import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import pytesseract
from PIL import Image
import cv2
import tempfile
import requests
import random

# Optional: Change this to your Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Mapping reliable sources based on keywords
reliable_sources = {
    "politics": ["https://www.politifact.com", "https://www.factcheck.org"],
    "health": ["https://www.cdc.gov", "https://www.who.int"],
    "technology": ["https://www.techcrunch.com", "https://www.theverge.com"],
    "finance": ["https://www.wsj.com", "https://www.bloomberg.com"],
    "climate": ["https://www.climate.gov", "https://www.nationalgeographic.com/environment"],
    "general": ["https://www.reuters.com", "https://apnews.com", "https://www.bbc.com/news"]
}

# Function to get reliable sources based on the text's context
def get_reliable_sources_for_text(text):
    text = text.lower()
    if "election" in text or "government" in text or "senate" in text:
        return reliable_sources["politics"]
    elif "covid" in text or "vaccine" in text or "hospital" in text:
        return reliable_sources["health"]
    elif "ai" in text or "technology" in text or "gadget" in text:
        return reliable_sources["technology"]
    elif "stock" in text or "market" in text or "finance" in text:
        return reliable_sources["finance"]
    elif "climate" in text or "environment" in text or "global warming" in text:
        return reliable_sources["climate"]
    else:
        return reliable_sources["general"]

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "./bert_model"
    model_name = "textattack/bert-base-uncased-imdb"

    try:
        if os.path.exists(model_path):
            tokenizer = BertTokenizer.from_pretrained(model_path)
            model = BertForSequenceClassification.from_pretrained(model_path)
        else:
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertForSequenceClassification.from_pretrained(model_name)
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error("Error loading model/tokenizer: " + str(e))
        return None, None

# Function for fake news prediction
def predict_fake_news(text):
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        return "Model loading failed."

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        prediction = torch.argmax(logits, dim=-1).item()
    return "FAKE" if prediction == 1 else "REAL"

# Function to simulate a bias score
def predict_bias(text):
    return random.randint(0, 100)  # Simulate a bias score between 0 and 100

# Function to send image for analysis to Flask API
def analyze_image(image_file):
    files = {"file": image_file}
    try:
        response = requests.post("http://localhost:5000/api/analyze/image", files=files)

        if response.status_code == 200:
            return response.json().get("status", "No status")
        else:
            return f"Error: {response.json().get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error during request: {str(e)}"

# Function to send video for analysis to Flask API
def analyze_video(video_file):
    files = {"file": video_file}
    try:
        response = requests.post("http://localhost:5000/api/analyze/video", files=files)

        if response.status_code == 200:
            return response.json().get("result", "No result")
        else:
            return f"Error: {response.json().get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error during request: {str(e)}"

# Streamlit app UI
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("Fake News Detection System")

# Dark Theme Styling
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #EAEAEA;
    }
    .stButton>button {
        background-color: #FF6347;
        color: white;
        font-size: 16px;
        border-radius: 5px;
        padding: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF4500;
    }
    .stProgress {
        background-color: #2C6B2F;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["📝 Text", "🖼️ Image", "🎥 Video"])

with tab1:
    st.header("Text-based Detection", anchor="text-detection")
    input_text = st.text_area("Enter a news article or headline:")
    
    if st.button("Analyze Text"):
        if not input_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                result = predict_fake_news(input_text)
                bias_score = predict_bias(input_text)  # Get bias score
                st.subheader("Fake News Detection Result:")
                if result == "REAL":
                    st.success("✅ This news appears to be REAL.")
                else:
                    st.error("🚨 This news appears to be FAKE.")
                
                # Display Bias Meter
                st.subheader("Bias Meter:")
                st.progress(bias_score)  # Show the bias meter as a progress bar
                st.write(f"Bias Score: {bias_score}/100")
                if bias_score > 75:
                    st.warning("⚠️ This news seems to have strong bias!")
                elif bias_score < 25:
                    st.info("✅ This news seems neutral with little to no bias.")
                else:
                    st.warning("⚠️ This news may have some bias.")

                # Suggestion Bar with Reliable Sources based on keywords
                sources = get_reliable_sources_for_text(input_text)
                st.subheader("Suggestion:")
                if result == "FAKE":
                    st.markdown(
                        f"""
                        <div style="background-color:#2e2e2e; padding:15px; border-radius:10px; border:1px solid #444;">
                            <p style="color:#FFD700; font-size:16px;">This news may be fake. Verify it using these trusted sources:</p>
                            {' | '.join([f'<a href="{src}" target="_blank" style="color:#00BFFF;">{src}</a>' for src in sources])}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.info("You can verify this news from these trusted sources:")
                    for src in sources:
                        st.markdown(f"- [{src}]({src})")

with tab2:
    st.header("Image-based Detection", anchor="image-detection")
    image_file = st.file_uploader("Upload a news image", type=["png", "jpg", "jpeg"])
    if image_file and st.button("Analyze Image"):
        with st.spinner("Extracting text and analyzing..."):
            result = analyze_image(image_file)
            if result == "REAL":
                st.success("✅ Image contains REAL news.")
            else:
                st.error("🚨 Image contains FAKE news.")

with tab3:
    st.header("Video-based Detection", anchor="video-detection")
    video_file = st.file_uploader("Upload a news video", type=["mp4", "avi", "mov"])
    if video_file and st.button("Analyze Video"):
        with st.spinner("Extracting frames and analyzing..."):
            result = analyze_video(video_file)
            if result == "REAL":
                st.success("✅ Video content appears to be REAL.")
            else:
                st.error("🚨 Video content appears to be FAKE.")

# Add feedback
st.markdown(
    """
    <p style="font-size: 20px; color: #32CD32; text-align: center;">
        Thank you for using the Fake News Detection system!
    </p>
    """, unsafe_allow_html=True
)

# Optional: Downloadable analysis result
st.sidebar.header("Download Analysis Results")
if st.button("Download Result as Text"):
    result_text = "Fake News Analysis Result"
    st.download_button(
        label="Download Result",
        data=result_text,
        file_name="fake_news_analysis.txt",
        mime="text/plain"
    )
