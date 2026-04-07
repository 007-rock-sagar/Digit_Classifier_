import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = load_model("mnist_cnn.h5")

st.title("MNIST Digit Classifier")

# --- Option 1: Upload an image ---
st.header("Upload a digit image")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    st.image(image, caption="Uploaded Digit", use_column_width=True)

    prediction = model.predict(img_array)
    st.write(f"Predicted Digit: **{np.argmax(prediction)}**")

    # Confidence chart
    probs = prediction[0]
    st.bar_chart(probs)

# --- Option 2: Draw on canvas ---
st.header("Or draw a digit below")

# Use a dynamic key to reset canvas when cleared
canvas_key = "canvas_default"
if st.button("Clear Canvas"):
    canvas_key = f"canvas_{np.random.randint(0, 100000)}"  # unique key each time

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    width=200,
    height=200,
    drawing_mode="freedraw",
    key=canvas_key,
)

if canvas_result.image_data is not None:
    # Convert canvas to 28x28 grayscale
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
    img = img.resize((28, 28))
    img = Image.eval(img, lambda x: 255 - x)  # invert colors
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    st.write(f"Predicted Digit: **{np.argmax(prediction)}**")

    # Confidence chart
    probs = prediction[0]
    st.bar_chart(probs)
