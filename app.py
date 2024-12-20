import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model('best_model_DenseNet121.keras')

# Fungsi untuk prediksi gambar
def predict_image(image):
    img = image.resize((128, 128))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)

    prob = prediction[0][0] * 100
    if prob > 50:
        result = "PNEUMONIA DETECTED"
        message = "Check further to confirm this prediction and obtain a more accurate diagnosis regarding the possibility of pneumonia."
    else:
        result = "NORMAL"
        message = "No signs of pneumonia detected."

    return result, prob, message

# Streamlit App
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

st.markdown(
    """
    <style>
    .title-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        margin-bottom: 20px;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        margin-bottom: 40px;
    }
    .result {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .probability {
        font-size: 24px;
        text-align: center;
        margin-top: 10px;
    }
    .message {
        font-size: 20px;
        font-style: italic;
        text-align: center;
        margin-top: 20px;
        color: #FFD700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display title with healthcare emoji icons
st.markdown(
    '<div class="title-container">'
    '   <span style="font-size: 48px;">🩺</span>'  # Stethoscope emoji
    '   <div class="title">Pneumonia Detection from X-Ray</div>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="subtitle">Upload an X-Ray image to detect Pneumonia</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=['jpeg', 'jpg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)
    result, prob, message = predict_image(image)
    
    result_color = "#FF4B4B" if "PNEUMONIA" in result else "#4CAF50"
    st.markdown(f'<div class="result" style="color: {result_color};">Prediction: {result}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="probability">Probability: {prob:.2f}%</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="message">{message}</div>', unsafe_allow_html=True)
