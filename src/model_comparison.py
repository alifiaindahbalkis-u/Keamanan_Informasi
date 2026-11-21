
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
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


# 1. Membaca dataset training
print("Membaca dataset training...")

train, _ = load_dataset()

X, y = prepare_features(train)


# 2. Membagi data training dan validation
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


# 3. Membuat fungsi preprocessing
def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ],
        remainder="passthrough"
    )


# 4. Membuat dua model
models = {
    "Decision Tree": Pipeline(
        steps=[
            ("preprocessing", create_preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            ("preprocessing", create_preprocessor()),
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
}


# 5. Training dan evaluasi
results = []

print("\n===== MODEL COMPARISON =====")

for name, model in models.items():

    print(f"\nMelatih {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

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

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

    print("Accuracy :", f"{accuracy:.4f}")
    print("Precision:", f"{precision:.4f}")
    print("Recall   :", f"{recall:.4f}")
    print("F1-Score :", f"{f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


# 6. Membuat tabel perbandingan
results_df = pd.DataFrame(results)

print("\n===== HASIL PERBANDINGAN MODEL =====")

print(
    results_df.to_string(
        index=False,
        formatters={
            "accuracy": lambda x: f"{x:.4f}",
            "precision": lambda x: f"{x:.4f}",
            "recall": lambda x: f"{x:.4f}",
            "f1_score": lambda x: f"{x:.4f}"
        }
    )
)


# 7. Menentukan model dengan F1 tertinggi
best_model = results_df.loc[
    results_df["f1_score"].idxmax()
]

print("\n===== MODEL TERBAIK BERDASARKAN F1 =====")
print("Model:", best_model["model"])
print("F1-Score:", f"{best_model['f1_score']:.4f}")


# 8. Menyimpan hasil perbandingan
Path("models").mkdir(exist_ok=True)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)


# 9. Membuat grafik perbandingan
metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1_score"
]

plot_data = results_df.set_index("model")[metrics]

plot_data.plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1)
)

plt.title("Decision Tree vs Random Forest")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=0)
plt.legend(title="Evaluation Metrics")
plt.tight_layout()

plt.savefig(
    "models/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFile berhasil disimpan:")
print("models/model_comparison.csv")
print("models/model_comparison.png")

print("\nMODEL COMPARISON SELESAI!")
