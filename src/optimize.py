
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocessing import (
    load_dataset,
    prepare_features,
    CATEGORICAL_FEATURES
)

# 1. Membaca data training saja
print("Membaca dataset training...")

train, _ = load_dataset()
X, y = prepare_features(train)

# 2. Membagi training dan validation
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nPembagian data:")
print("Training  :", X_train.shape)
print("Validation:", X_val.shape)

# 3. Membuat preprocessing
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

# 4. Membuat model Random Forest
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

# 5. Training
print("\nMemulai training model...")

model.fit(X_train, y_train)

print("Training selesai!")

# 6. Mengambil probabilitas prediksi
probabilities = model.predict_proba(X_val)[:, 1]

# 7. Menguji beberapa threshold
thresholds = [0.30, 0.40, 0.50, 0.60]

results = []

print("\n===== HASIL OPTIMASI THRESHOLD =====")

for threshold in thresholds:

    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print(f"\nThreshold: {threshold}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

# 8. Memilih threshold dengan F1 tertinggi
results_df = pd.DataFrame(results)

best_result = results_df.loc[
    results_df["f1_score"].idxmax()
]

best_threshold = best_result["threshold"]

print("\n===== THRESHOLD TERBAIK =====")
print(best_result)

# 9. Confusion Matrix validation
best_predictions = (
    probabilities >= best_threshold
).astype(int)

cm = confusion_matrix(
    y_val,
    best_predictions,
    labels=[0, 1]
)

print("\nConfusion Matrix Validation:")
print(cm)

# 10. Menyimpan hasil optimasi
Path("models").mkdir(exist_ok=True)

results_df.to_csv(
    "models/optimization_results.csv",
    index=False
)

print("\nHasil optimasi disimpan di:")
print("models/optimization_results.csv")

print("\nOPTIMASI VALIDATION SELESAI")
