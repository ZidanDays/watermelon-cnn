import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('leaf_disease_classifier100.h5')

# Streamlit App
st.title("Sistem Pendeteksi Dini Penyakit Tanaman Semangka")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
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
        max_prob = np.max(prediction)  # Get the highest probability
        predicted_class_index = np.argmax(prediction)
        class_list = ['Anthracnose', 'Downy_Mildew', 'Healthy', 'Mosaic_Virus']
        predicted_class = class_list[predicted_class_index]

        # Check if the prediction is confident enough
        # threshold = 0.6  # Define a confidence threshold (adjustable)
        threshold = 0.8  # Define a confidence threshold (adjustable)  
        if max_prob < threshold:
            st.warning("Gambar yang diunggah bukan gambar penyakit yang terdeteksi oleh sistem. Harap unggah gambar tanaman yang relevan.")
        else:
            # Display the predicted class
            st.write(f"Predicted class: {predicted_class}")

            # Add handling recommendations
            if predicted_class == 'Anthracnose':
                st.write("**Penanganan Anthracnose:**")
                st.write("- Gunakan fungisida berbahan aktif tembaga.")
                st.write("- Potong dan buang bagian tanaman yang terinfeksi.")
                st.write("- Jaga kelembapan tanah tanpa membasahi daun.")
            elif predicted_class == 'Downy_Mildew':
                st.write("**Penanganan Downy Mildew:**")
                st.write("- Gunakan fungisida sistemik atau kontak.")
                st.write("- Hindari kelembapan tinggi dengan meningkatkan sirkulasi udara.")
                st.write("- Jangan menyiram tanaman di malam hari.")
            elif predicted_class == 'Healthy':
                st.write("Tanaman dalam kondisi **sehat**! Berikut beberapa tips perawatan:")
                st.write("- Jaga tanah tetap subur dengan pemberian pupuk organik.")
                st.write("- Periksa secara rutin untuk mendeteksi gejala penyakit sejak dini.")
                st.write("- Pastikan tanaman mendapatkan cukup sinar matahari.")
            elif predicted_class == 'Mosaic_Virus':
                st.write("**Penanganan Mosaic Virus:**")
                st.write("- Hancurkan tanaman yang terinfeksi untuk mencegah penyebaran.")
                st.write("- Kendalikan serangga vektor seperti kutu daun dengan insektisida.")
                st.write("- Gunakan varietas tahan penyakit untuk penanaman berikutnya.")

    except Exception as e:
        st.error("Gambar tidak valid. Silakan unggah gambar lain.")
