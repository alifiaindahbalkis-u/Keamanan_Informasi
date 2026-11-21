
import os
import joblib
import numpy as np

from preprocessing import load_dataset, prepare_features

MODEL_PATH = "models/final_model.pkl"

print("===== VALIDASI MODEL FINAL =====")

# 1. Periksa file model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("File final_model.pkl tidak ditemukan!")

print("\n[1] File model ditemukan.")

# 2. Muat model
saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
threshold = saved_model["threshold"]

print("[2] Model berhasil dimuat.")
print("Threshold:", threshold)

# 3. Ambil sebagian data testing
_, test_data = load_dataset()

X_test, y_test = prepare_features(test_data)

X_sample = X_test.iloc[:10]
y_sample = y_test.iloc[:10]

# 4. Prediksi
probabilities = model.predict_proba(X_sample)[:, 1]
predictions = (probabilities >= threshold).astype(int)

print("\n[3] Hasil prediksi 10 data:")
for i in range(len(predictions)):
    actual = "Attack" if y_sample.iloc[i] == 1 else "Normal"
    predicted = "Attack" if predictions[i] == 1 else "Normal"

    print(
        f"Data {i+1}: "
        f"Aktual={actual}, "
        f"Prediksi={predicted}, "
        f"Probabilitas={probabilities[i]:.4f}"
    )

# 5. Validasi
assert len(predictions) == 10
assert np.isin(predictions, [0, 1]).all()

print("\n[4] Validasi berhasil!")
print("Model siap digunakan untuk tahap integrasi.")
