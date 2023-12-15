import streamlit as st

import pandas as pd
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from PIL import ImageOps


st.sidebar.image('./logo UTM.png', caption='', use_column_width=True)
st.sidebar.write("Tugas Akhir")
st.sidebar.write("Achmad Rizky Rino Saputra (190441100090)")

# Load the pre-trained Keras model
model_path = './new_model/model_fine_5.keras'
model = load_model(model_path)

# Image preprocessing function
def preprocess_image(image):
    adj = cv2.convertScaleAbs(image, alpha=1.90, beta=-40)
    thresh = cv2.inRange(adj, np.array([150, 0, 90]), np.array([255, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = 255 - morph
    result = cv2.bitwise_and(image, image, mask=mask)

    resized_image = cv2.resize(result, (224, 224))
    normalized_image = resized_image / 255.0  # Normalize to [0, 1]
    return np.expand_dims(normalized_image, axis=0)  # Add batch dimension

# Streamlit app
def main():
    st.title("Klasifikasi Citra Penyakit Batang Jagung menggunakan VGG-19")
    st.write("Keterangan : Citra input harus menggunakan latar belakang putih (seperti pada contoh di bawah).")

    # Upload image through Streamlit file uploader
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2, col3 = st.columns(3)
        with col2:
            st.image(uploaded_file, caption='Citra Asli', use_column_width=True)

        # Perform predictions on the uploaded image
        col1, col2, col3 = st.columns(3)
        # Read and preprocess the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        processed_image = preprocess_image(image)

        # Make predictions
        predictions = model.predict(processed_image)

        # Display the predictions
        predictions_df = pd.DataFrame(predictions)
        with col1:
            st.subheader('Anthracnose')
            st.subheader(format(predictions_df.iloc[0][0], ".2%"))
        with col2:
            st.subheader('Fusarium')
            st.subheader(format(predictions_df.iloc[0][1], ".2%"))
        with col3:
            st.subheader('Normal')
            st.subheader(format(predictions_df.iloc[0][2], ".2%"))

    else:
        col1, col2, col3 = st.columns(3)
        with col2:
            image = Image.open('./contoh.jpg')
            st.image(image, caption='Contoh Citra', use_column_width=True)

        img = np.array(image, dtype="uint8")
        adj = cv2.convertScaleAbs(img, alpha=1.90, beta=-40)
        thresh = cv2.inRange(adj, np.array([150, 0, 90]), np.array([255, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = 255 - morph
        result = cv2.bitwise_and(img, img, mask=mask)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(adj, caption='Citra Preprocessing', use_column_width=True)
        with col2:
            st.image(mask, caption='Citra Mask', use_column_width=True)
        with col3:
            st.image(result, caption='Hasil Mask', use_column_width=True)

if __name__ == "__main__":
    main()