import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Helper function to load model and cache it
@st.cache_resource
def load_model():
    model_path = 'leaf_disease_classifier2.h5'
    st.write(f"Checking if model file exists: {model_path}")
    if not os.path.exists(model_path):
        st.error(f"Model file {model_path} does not exist. Please check the path.")
        return None
    st.write(f"Model file found: {model_path}")
    return tf.keras.models.load_model(model_path)

# Load the model
model = load_model()
if model is None:
    st.stop()  # Stop execution if model could not be loaded

# Streamlit App
st.title("Sistem Pendeteksi Penyakit Tanaman Semangka")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for prediction
    img = image.resize((150, 150))  # Resize the image to the desired dimensions
    img_array = np.array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch
    img_array = img_array / 255.0  # Normalize the image data

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    class_list = ['Anthracnose', 'Downy_Mildew', 'Healthy', 'Mosaic_Virus']
    predicted_class = class_list[predicted_class_index]

    # Display the predicted class
    st.write(f"Predicted class: {predicted_class}")
