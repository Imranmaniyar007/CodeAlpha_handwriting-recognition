import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model("cnn_model.h5")

st.title("✍ Handwritten Digit Recognition")

st.write("Upload an image of a digit (0–9)")

uploaded = st.file_uploader("Choose image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("L")
    img = img.resize((28,28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1,28,28,1)

    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    st.image(img, caption="Uploaded Image", width=150)
    st.success(f"Predicted Digit: {digit}")
