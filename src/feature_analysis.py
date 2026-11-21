
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path


# 1. Membaca model final
print("Memuat model Random Forest...")

saved_data = joblib.load("models/final_model.pkl")

model = saved_data["model"]

print("Model berhasil dimuat!")


# 2. Mengambil Random Forest dari Pipeline
random_forest = model.named_steps["classifier"]

# Mengambil nama fitur setelah preprocessing
feature_names = (
    model.named_steps["preprocessing"].get_feature_names_out()
)

# Mengambil nilai feature importance
importances = random_forest.feature_importances_


# 3. Membuat tabel feature importance
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

# Mengurutkan dari nilai terbesar
importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
).reset_index(drop=True)

# Menambahkan ranking
importance_df.insert(
    0,
    "rank",
    range(1, len(importance_df) + 1)
)


# 4. Menampilkan 10 fitur teratas
print("\n===== TOP 10 FEATURE IMPORTANCE =====")

print(
    importance_df.head(10).to_string(
        index=False,
        formatters={
            "importance": lambda x: f"{x:.6f}"
        }
    )
)


# 5. Menyimpan seluruh hasil ke CSV
Path("models").mkdir(exist_ok=True)

importance_df.to_csv(
    "models/feature_importance.csv",
    index=False
)

print("\nData feature importance disimpan:")
print("models/feature_importance.csv")


# 6. Membuat grafik Top 10
top_10 = importance_df.head(10).sort_values(
    by="importance",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_10["feature"],
    top_10["importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Fitur Jaringan")
plt.title("Top 10 Feature Importance - Random Forest")

plt.tight_layout()

plt.savefig(
    "models/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGrafik disimpan:")
print("models/feature_importance.png")

print("\nANALISIS FEATURE IMPORTANCE SELESAI!")
