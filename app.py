import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('m_f_prediction.h5')  # Update this path to your model file

# Define image size for the model
img_width, img_height = 200, 200

# Preprocess the image for prediction
def preprocess_image(img):
    img = img.resize((img_width, img_height))
    img_arr = np.array(img, dtype=np.float32)
    img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension
    img_arr /= 255  # Normalize pixel values
    return img_arr

def predict_image(img):
    # Predict the class of the image
    img_arr = preprocess_image(img)
    predictions = model.predict(img_arr)
    if predictions[0] > 0.5:
        return 'male'
    else:
        return 'female'

# Streamlit UI
st.title('Gender Classification')
st.write("Upload an image to classify whether the person is male or female.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image')
    st.write("")
    
    # Make prediction
    prediction = predict_image(img)
    st.write(f'Prediction: {prediction}')
