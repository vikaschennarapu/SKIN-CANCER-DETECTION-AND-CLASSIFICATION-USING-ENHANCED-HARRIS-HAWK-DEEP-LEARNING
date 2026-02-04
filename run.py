import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf


@st.cache_resource
def load_cnn_model():
    model = load_model("model.h5")
    return model

model = load_cnn_model()


CLASS_LABELS = sorted(["Melanoma","Vascular Lesion","Squamous Cell Carcinoma","Dermatofibroma","Melanocytic Nevus (also just “Nevus”)","Benign Keratosis-like Lesion","Basal Cell Carcinoma","Actinic Keratosis"])

st.set_page_config(page_title="Skin Cancer Classifier", layout="centered")
st.title("🩺 Skin Cancer Detection App")
st.markdown("Upload a **dermoscopic skin lesion image** to classify it using the trained CNN model (EHHO-optimized).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("")


    def preprocess_image(image_file):
        img = load_img(image_file, target_size=(28, 28))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    input_image = preprocess_image(uploaded_file)


    with st.spinner("Analyzing image..."):
        preds = model.predict(input_image)
        predicted_class_index = np.argmax(preds)
        predicted_class = CLASS_LABELS[predicted_class_index]
        confidence = np.max(preds) * 100
    st.success(f"**Predicted Class:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    st.subheader("Class Probabilities")
    st.bar_chart(dict(zip(CLASS_LABELS, preds.flatten())))
else:
    st.warning("👆 Please upload a dermoscopic image to begin.")
