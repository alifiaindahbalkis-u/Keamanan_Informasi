
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

# Lokasi penyimpanan model
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# 1. Membaca dataset
print("Membaca dataset...")

train, test = load_dataset()

X_train, y_train = prepare_features(train)
X_test, y_test = prepare_features(test)

# 2. Menyiapkan encoding
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

# 3. Membuat pipeline Random Forest
model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
        )
    ]
)

# 4. Training model
print("\nMemulai training Random Forest...")

model.fit(X_train, y_train)

print("Training selesai!")

# 5. Prediksi data testing
print("\nMelakukan prediksi data testing...")

y_pred = model.predict(X_test)

# 6. Evaluasi model
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

cm = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
)

print("\n===== HASIL EVALUASI RANDOM FOREST =====")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"],
        zero_division=0
    )
)

# 7. Menyimpan model dan preprocessing
model_path = MODEL_DIR / "random_forest_pipeline.pkl"

joblib.dump(model, model_path)

print("\nModel berhasil disimpan di:", model_path)
