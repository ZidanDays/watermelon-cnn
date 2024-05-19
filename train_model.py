import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random
from shutil import copyfile
import numpy as np

# Definisikan direktori dataset
base_dir = 'D:/Pendeteksi_CNN - Copy/dataset'
bahan_dir = os.path.join(base_dir, 'bahan')
train_dir = os.path.join(base_dir, 'latih')
validation_dir = os.path.join(base_dir, 'validasi')

Anthracnose_dir = os.path.join(bahan_dir, 'Anthracnose/')
Downy_Mildew_dir = os.path.join(bahan_dir, 'Downy_Mildew/')
Healthy_dir = os.path.join(bahan_dir, 'Healthy/')
Mosaic_Virus_dir = os.path.join(bahan_dir, 'Mosaic_Virus/')

# Menyiapkan dataset
def train_val_split(source, train, val, train_ratio):
    total_size = len(os.listdir(source))
    train_size = int(train_ratio * total_size)

    # Mengacak file
    randomized = random.sample(os.listdir(source), total_size)
    train_files = randomized[:train_size]
    val_files = randomized[train_size:]

    # Memindahkan file ke direktori train dan validasi
    for file in train_files:
        src_file = os.path.join(source, file)
        dst_file = os.path.join(train, file)
        copyfile(src_file, dst_file)
    for file in val_files:
        src_file = os.path.join(source, file)
        dst_file = os.path.join(val, file)
        copyfile(src_file, dst_file)

# Pembagian Training dan Validasi
train_val_split(Anthracnose_dir, os.path.join(train_dir, 'Anthracnose/'), os.path.join(validation_dir, 'Anthracnose/'), 0.2)
train_val_split(Downy_Mildew_dir, os.path.join(train_dir, 'Downy_Mildew/'), os.path.join(validation_dir, 'Downy_Mildew/'), 0.2)
train_val_split(Healthy_dir, os.path.join(train_dir, 'Healthy/'), os.path.join(validation_dir, 'Healthy/'), 0.2)
train_val_split(Mosaic_Virus_dir, os.path.join(train_dir, 'Mosaic_Virus/'), os.path.join(validation_dir, 'Mosaic_Virus/'), 0.2)

# Membuat generator untuk data training dan validasi
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    horizontal_flip=True,
    shear_range=0.3,
    fill_mode='nearest',
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode='categorical'
)

# Verify the number of classes
num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")

# Membuat model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dropout(0.3, seed=112),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dropout(0.5, seed=112),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Pastikan ini sesuai dengan jumlah kelas yang Anda miliki
])

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

# Melatih model
history = model.fit(
      train_generator,
      steps_per_epoch=train_generator.n // train_generator.batch_size,
      epochs=25,
      validation_data=val_generator,
      validation_steps=val_generator.n // val_generator.batch_size,
      verbose=1
)

# Simpan model
model.save('leaf_disease_classifier2.h5')
