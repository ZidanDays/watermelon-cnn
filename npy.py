import numpy as np

# Ambil data dari validation generator
val_generator.reset()  # Pastikan generator mulai dari awal
X_test = []  # List untuk menyimpan data gambar
y_test = []  # List untuk menyimpan label asli

# Iterasi melalui seluruh batch di validation generator
for _ in range(val_generator.samples // val_generator.batch_size):
    images, labels = next(val_generator)  # Ambil batch gambar dan label
    X_test.append(images)
    y_test.append(labels)

# Konversi list ke numpy array
X_test = np.concatenate(X_test, axis=0)  # Gabungkan semua batch menjadi satu array
y_test = np.concatenate(y_test, axis=0)  # Gabungkan semua label menjadi satu array

# Simpan X_test dan y_test dalam file .npy
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("X_test.npy dan y_test.npy berhasil dibuat dan disimpan!")
