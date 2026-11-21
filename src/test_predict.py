
import sys
import pandas as pd

from preprocessing import load_dataset, prepare_features
from predict import predict_network

# Mengambil data testing
_, test_data = load_dataset()

X_test, y_test = prepare_features(test_data)

# Menguji 10 data pertama
sample = X_test.iloc[:10]

results = predict_network(sample)

print("===== PENGUJIAN MODUL PREDIKSI =====")

for i, result in enumerate(results):
    print(
        f"Data {i+1}: "
        f"Prediksi={result['label']}, "
        f"Probabilitas={result['probability']:.4f}"
    )

print("\nModul prediksi berhasil dijalankan!")
