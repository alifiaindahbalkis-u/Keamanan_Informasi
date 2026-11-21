
import pandas as pd
import joblib

from pathlib import Path
from preprocessing import load_dataset, prepare_features


# 1. Memuat model final
print("Memuat model Random Forest...")

saved_data = joblib.load("models/final_model.pkl")

model = saved_data["model"]
threshold = saved_data["threshold"]


# 2. Membaca dataset training dan testing
print("\nMembaca dataset NSL-KDD...")

train, test = load_dataset()

print("Training:", train.shape)
print("Testing :", test.shape)


# 3. Mengambil jenis serangan
def normalize_label(labels):
    return (
        labels.astype(str)
        .str.strip()
        .str.lower()
        .str.rstrip(".")
    )


train_labels = normalize_label(train["label"])
test_labels = normalize_label(test["label"])

# Mengambil jenis serangan selain normal
known_attacks = set(
    train_labels[train_labels != "normal"].unique()
)

test_attacks = set(
    test_labels[test_labels != "normal"].unique()
)

# Serangan yang tidak ditemukan pada training
unseen_attacks = test_attacks - known_attacks

print("\n===== PERBANDINGAN JENIS SERANGAN =====")

print("Jenis serangan di training :", len(known_attacks))
print("Jenis serangan di testing  :", len(test_attacks))
print("Jenis serangan baru        :", len(unseen_attacks))


# 4. Menampilkan serangan yang tidak ditemukan di training
print("\n===== UNSEEN ATTACK TYPES =====")

for attack in sorted(unseen_attacks):
    print("-", attack)


# 5. Melakukan prediksi pada KDDTest+
print("\nMelakukan prediksi pada KDDTest+...")

X_test, y_test = prepare_features(test)

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)


# 6. Membuat tabel hasil prediksi
results = pd.DataFrame({
    "attack_type": test_labels.to_numpy(),
    "actual": y_test.to_numpy(),
    "predicted": predictions
})

# Mengambil data serangan saja
attack_results = results[
    results["actual"] == 1
].copy()

# Menandai apakah jenis serangan dikenal saat training
attack_results["attack_category"] = (
    attack_results["attack_type"]
    .apply(
        lambda x: "Known" if x in known_attacks else "Unseen"
    )
)


# 7. Menghitung performa setiap jenis serangan
analysis = (
    attack_results.groupby(
        ["attack_type", "attack_category"]
    )
    .agg(
        total_data=("actual", "count"),
        detected=("predicted", "sum")
    )
    .reset_index()
)

analysis["missed"] = (
    analysis["total_data"] - analysis["detected"]
)

analysis["detection_rate"] = (
    analysis["detected"] /
    analysis["total_data"] * 100
)

analysis["miss_rate"] = (
    analysis["missed"] /
    analysis["total_data"] * 100
)

analysis = analysis.sort_values(
    by=["detection_rate", "total_data"],
    ascending=[True, False]
).reset_index(drop=True)


# 8. Menampilkan hasil per jenis serangan
print("\n===== DETECTION RATE PER ATTACK TYPE =====")

print(
    analysis.to_string(
        index=False,
        formatters={
            "detection_rate": lambda x: f"{x:.2f}%",
            "miss_rate": lambda x: f"{x:.2f}%"
        }
    )
)


# 9. Membandingkan Known dan Unseen Attacks
category_summary = (
    attack_results.groupby("attack_category")
    .agg(
        total_data=("actual", "count"),
        detected=("predicted", "sum")
    )
    .reset_index()
)

category_summary["missed"] = (
    category_summary["total_data"] -
    category_summary["detected"]
)

category_summary["detection_rate"] = (
    category_summary["detected"] /
    category_summary["total_data"] * 100
)

category_summary["miss_rate"] = (
    category_summary["missed"] /
    category_summary["total_data"] * 100
)

print("\n===== KNOWN VS UNSEEN ATTACKS =====")

print(
    category_summary.to_string(
        index=False,
        formatters={
            "detection_rate": lambda x: f"{x:.2f}%",
            "miss_rate": lambda x: f"{x:.2f}%"
        }
    )
)


# 10. Menyimpan hasil analisis
Path("models").mkdir(exist_ok=True)

analysis.to_csv(
    "models/attack_analysis.csv",
    index=False
)

category_summary.to_csv(
    "models/known_vs_unseen.csv",
    index=False
)

unseen_df = pd.DataFrame({
    "unseen_attack": sorted(unseen_attacks)
})

unseen_df.to_csv(
    "models/unseen_attacks.csv",
    index=False
)


print("\nFile berhasil disimpan:")
print("models/attack_analysis.csv")
print("models/known_vs_unseen.csv")
print("models/unseen_attacks.csv")

print("\nATTACK ANALYSIS SELESAI!")
