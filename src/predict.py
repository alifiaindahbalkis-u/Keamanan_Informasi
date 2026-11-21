
import joblib
import pandas as pd

MODEL_PATH = "models/final_model.pkl"

# Memuat model final
saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
threshold = saved_model["threshold"]


def predict_network(data):
    """
    Memprediksi data jaringan:
    0 = Normal
    1 = Attack
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input harus berupa pandas DataFrame")

    if data.shape[1] != 41:
        raise ValueError(
            f"Model membutuhkan 41 fitur, "
            f"tetapi menerima {data.shape[1]} fitur."
        )

    probabilities = model.predict_proba(data)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    results = []

    for prediction, probability in zip(
        predictions, probabilities
    ):
        results.append({
            "prediction": int(prediction),
            "label": (
                "Attack" if prediction == 1 else "Normal"
            ),
            "probability": float(probability)
        })

    return results
