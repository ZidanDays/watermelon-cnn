import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Model .h5
model = load_model("leaf_disease_classifier100.h5")  # Ganti dengan nama file model kamu

# 2. Load atau siapkan data uji (X_test, y_test)
# Contoh dummy (replace dengan data asli)
X_test = np.load("X_test.npy")  # Data fitur uji
y_test = np.load("y_test.npy")  # Label sebenarnya

# 3. Lakukan Prediksi
y_pred_probs = model.predict(X_test)  # Output probabilitas
y_pred = np.argmax(y_pred_probs, axis=1)  # Konversi ke label prediksi jika klasifikasi multi-class

# 4. Hitung Metrik Evaluasi
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 5. Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 6. Hitung F1-score, Precision, Recall, dan Akurasi
f1 = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
accuracy = accuracy_score(y_test, y_pred)

print(f"Akurasi: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
