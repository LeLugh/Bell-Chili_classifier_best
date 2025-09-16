import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load  trained model
from keras.models import load_model

MODEL_PATH = "Bell-Chili_classifier_best.h5"
model = load_model(MODEL_PATH)

# Match the size used in training (128x128)
IMG_SIZE = (128, 128)

# Class names (in order)
class_names = ["bell", "chili"]

# Streamlit app
st.title("🌶️ Bell Pepper vs Chili Classifier 🫑")
st.write("Upload an image of a pepper, and the model will predict whether it is a **Bell Pepper** or a **Chili**.")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize(IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0  # normalize like training

    # Prediction
    pred = model.spredict(x)[0][0]

    # If sigmoid (1 unit)
    if pred >= 0.5:
        label = class_names[1]  # chili
        confidence = pred
    else:
        label = class_names[0]  # bell
        confidence = 1 - pred

    st.markdown(f"### ✅ Prediction: **{label.upper()}**")
    st.write(f"Confidence: {confidence:.2%}")
