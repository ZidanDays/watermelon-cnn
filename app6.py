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
                st.write("### Penanganan **Anthracnose**")
                st.write("- Gunakan fungisida berbahan aktif **tembaga**.")
                st.write("- Obat yang disarankan: **Antracol 70 WP**, **Kocide 3000**.")
                st.write("- Potong dan buang bagian tanaman yang terinfeksi.")
                st.write("- Jaga kelembapan tanah tanpa membasahi daun.")
                st.write("### Rekomendasi Pupuk:")
                st.write("- **NPK Mutiara** untuk memperkuat daya tahan tanaman.")
                st.write("- **KCL** untuk meningkatkan ketahanan terhadap serangan jamur.")
                st.write("### Pencegahan:")
                st.write("- Gunakan benih bebas patogen.")
                st.write("- Rotasi tanaman secara berkala.")
                st.write("- Hindari penyiraman berlebih terutama di malam hari.")

            elif predicted_class == 'Downy_Mildew':
                st.write("### Penanganan **Downy Mildew**")
                st.write("- Gunakan fungisida sistemik atau kontak seperti berbahan aktif **mankozeb** atau **metalaxyl**.")
                st.write("- Obat yang disarankan: **Ridomil Gold**, **Dithane M-45**.")
                st.write("- Tingkatkan sirkulasi udara di sekitar tanaman.")
                st.write("- Jangan menyiram tanaman di malam hari.")
                st.write("### Rekomendasi Pupuk:")
                st.write("- **Pupuk Daun Gandasil D** untuk menjaga kekuatan daun.")
                st.write("- **Pupuk MKP (Mono Kalium Phospat)** untuk mempercepat pertumbuhan akar dan mengurangi kelembapan berlebih.")
                st.write("### Pencegahan:")
                st.write("- Lakukan penyemprotan preventif sebelum musim hujan.")
                st.write("- Hindari penanaman terlalu rapat.")
                st.write("- Gunakan mulsa untuk mengurangi kelembapan tanah.")

            elif predicted_class == 'Healthy':
                st.write("### Tanaman dalam kondisi **sehat** 🌱")
                st.write("Berikut beberapa tips untuk mempertahankan kesehatannya:")
                st.write("- Lakukan pemeriksaan rutin terhadap daun dan batang.")
                st.write("- Pastikan sirkulasi udara baik dan sinar matahari cukup.")
                st.write("- Gunakan pupuk organik seperti **pupuk kandang fermentasi** atau **kompos jerami**.")
                st.write("### Pupuk Rekomendasi:")
                st.write("- **GDM Organik**, **NASA**, atau **Biojoglo**.")
                st.write("### Pencegahan:")
                st.write("- Jangan menyiram tanaman terlalu sering.")
                st.write("- Hindari penggunaan alat yang sebelumnya menyentuh tanaman sakit.")
                st.write("- Gunakan varietas unggul tahan penyakit.")

            elif predicted_class == 'Mosaic_Virus':
                st.write("### Penanganan **Mosaic Virus**")
                st.write("- Hancurkan segera tanaman yang terinfeksi untuk mencegah penyebaran.")
                st.write("- Kendalikan serangga vektor seperti **kutu daun**.")
                st.write("- Gunakan insektisida seperti **Confidor**, **Actara**, atau **Decis**.")
                st.write("- Tanam varietas tahan virus.")
                st.write("### Rekomendasi Pupuk:")
                st.write("- **Pupuk KNO3 Merah** atau **Dekastar** untuk meningkatkan kekebalan tanaman.")
                st.write("- **Pupuk mikro** seperti **Zinc, Boron, dan Mangan** untuk memperkuat jaringan tanaman.")
                st.write("### Pencegahan:")
                st.write("- Sterilkan alat pertanian secara rutin.")
                st.write("- Bersihkan gulma di sekitar tanaman.")
                st.write("- Jangan menggunakan benih dari tanaman yang sakit.")

    except Exception as e:
        st.error("Gambar tidak valid. Silakan unggah gambar lain.")
