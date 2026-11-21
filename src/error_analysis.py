
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import confusion_matrix

from preprocessing import load_dataset, prepare_features


# 1. Memuat model final
print("Memuat model final...")

saved_data = joblib.load("models/final_model.pkl")

model = saved_data["model"]
threshold = saved_data["threshold"]


# 2. Membaca dataset testing
print("\nMembaca dataset KDDTest+...")

_, test = load_dataset()

X_test, y_test = prepare_features(test)

# Menyimpan nama asli jenis serangan
attack_labels = test["label"].astype(str).str.strip()


# 3. Melakukan prediksi
print("\nMelakukan prediksi...")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (probabilities >= threshold).astype(int)


# 4. Mengidentifikasi kesalahan prediksi
results = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": predictions,
    "attack_type": attack_labels.to_numpy()
})

# False Positive: Normal diprediksi Attack
false_positive = results[
    (results["actual"] == 0) &
    (results["predicted"] == 1)
]

# False Negative: Attack diprediksi Normal
false_negative = results[
    (results["actual"] == 1) &
    (results["predicted"] == 0)
]


# 5. Analisis False Positive
print("\n===== FALSE POSITIVE ANALYSIS =====")

print("Jumlah False Positive:", len(false_positive))

print("\nData normal yang salah dikategorikan sebagai Attack:")

print(
    false_positive["attack_type"]
    .value_counts()
)


# 6. Analisis False Negative berdasarkan jenis serangan
print("\n===== FALSE NEGATIVE BY ATTACK TYPE =====")

fn_analysis = (
    false_negative["attack_type"]
    .value_counts()
    .rename_axis("attack_type")
    .reset_index(name="false_negative")
)

# Menghitung total data untuk setiap jenis serangan
attack_totals = (
    results[results["actual"] == 1]
    ["attack_type"]
    .value_counts()
    .rename_axis("attack_type")
    .reset_index(name="total_data")
)

# Menggabungkan hasil
fn_analysis = attack_totals.merge(
    fn_analysis,
    on="attack_type",
    how="left"
)

fn_analysis["false_negative"] = (
    fn_analysis["false_negative"].fillna(0).astype(int)
)

# Menghitung persentase kegagalan deteksi
fn_analysis["miss_rate"] = (
    fn_analysis["false_negative"] /
    fn_analysis["total_data"] * 100
)

fn_analysis = fn_analysis.sort_values(
    by="false_negative",
    ascending=False
).reset_index(drop=True)


print(
    fn_analysis.to_string(
        index=False,
        formatters={
            "miss_rate": lambda x: f"{x:.2f}%"
        }
    )
)


# 7. Menyimpan hasil analisis
Path("models").mkdir(exist_ok=True)

false_positive.to_csv(
    "models/false_positive.csv",
    index=False
)

false_negative.to_csv(
    "models/false_negative.csv",
    index=False
)

fn_analysis.to_csv(
    "models/false_negative_by_attack.csv",
    index=False
)


# 8. Confusion Matrix sebagai pemeriksaan
cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

print("\n===== CONFUSION MATRIX CHECK =====")
print(cm)

print("\nFile analisis berhasil disimpan:")
print("models/false_positive.csv")
print("models/false_negative.csv")
print("models/false_negative_by_attack.csv")

print("\nERROR ANALYSIS SELESAI!")
