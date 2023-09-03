import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('brain_tumor_model.h5')

st.title("Brain Tumor Detection")

# Upload an image for prediction
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    img = image.resize((128, 128))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img)
    class_names = ["No Tumor", "Tumor"]
    result = class_names[int(np.round(prediction[0][0]))]

    st.write(f"Prediction: {result}")
