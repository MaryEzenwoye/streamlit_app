import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np


from tensorflow.keras.applications import VGG16

# 1. Download the pre-trained model (trained on 1,000+ objects)
model = VGG16(weights='imagenet')

# 2. Save it to your current project folder as 'best_model.h5'
model.save('best_model.h5')

print("Model saved successfully as 'best_model.h5' in your project folder!")

# --- 1. App Configuration ---
st.set_page_config(page_title="Image Classifier", layout="centered")
st.title("🖼️ AI Image Classifier")
st.write("Upload an image to see the model's prediction in real-time.")


# --- 2. Load the Model ---
# We use st.cache_resource so the model doesn't reload every time you click a button
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('best_model.h5')
    return model


model = load_my_model()


# --- 3. Image Preprocessing ---
def preprocess_image(image_data):
    size = (224, 224)  # Match your model's expected input size
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)

    # Normalize if your model expects [0, 1] or [-1, 1]
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1

    # Add batch dimension (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data


# --- 4. File Uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preview the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    st.write("---")
    st.write("### 🤖 Classification")

    with st.spinner('Thinking...'):
        # Process and Predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        # Logic for Output
        # Assuming you have a list of classes: classes = ['Cat', 'Dog']
        # result = np.argmax(prediction)
        # confidence = prediction[0][result]

        # Placeholder for demonstration
        st.success(f"Prediction: Sample Class")
        st.info(f"Confidence Score: {np.max(prediction) * 100:.2f}%")

        # Optional: Probability Chart
        st.bar_chart(prediction[0])