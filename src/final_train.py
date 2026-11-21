import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocessing import (
    load_dataset,
    prepare_features,
    CATEGORICAL_FEATURES
)


# 1. Membaca dataset
print("Membaca dataset...")

train, test = load_dataset()

X_train, y_train = prepare_features(train)
X_test, y_test = prepare_features(test)

print("\nJumlah data:")
print("Training :", X_train.shape)
print("Testing  :", X_test.shape)


# 2. Membuat preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES
        )
    ],
    remainder="passthrough"
)


# 3. Membuat model Random Forest
model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
                min_samples_leaf=1
            )
        )
    ]
)


# 4. Training menggunakan seluruh KDDTrain+
print("\nMemulai final training...")

model.fit(X_train, y_train)

print("Final training selesai!")


# 5. Menentukan threshold terbaik
best_threshold = 0.5

print("\nThreshold yang digunakan:", best_threshold)


# 6. Prediksi menggunakan KDDTest+
print("\nMelakukan pengujian pada KDDTest+...")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= best_threshold
).astype(int)


# 7. Menghitung evaluasi
accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)


# 8. Confusion Matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# 9. Menampilkan hasil
print("\n===== HASIL EVALUASI FINAL =====")

print(f"Threshold : {best_threshold}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\n===== DETAIL HASIL DETEKSI =====")
print("True Negative  (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive  (TP):", tp)

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Attack"],
        zero_division=0
    )
)


# 10. Menyimpan model final
Path("models").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": model,
        "threshold": best_threshold
    },
    "models/final_model.pkl"
)


# 11. Menyimpan hasil evaluasi
results = pd.DataFrame([
    {
        "threshold": best_threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }
])

results.to_csv(
    "models/final_evaluation.csv",
    index=False
)

print("\nModel final disimpan di:")
print("models/final_model.pkl")

print("\nHasil evaluasi disimpan di:")
print("models/final_evaluation.csv")

print("\nFINAL TRAINING DAN EVALUASI SELESAI!")