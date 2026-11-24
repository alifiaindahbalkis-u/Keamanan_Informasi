import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.preprocessing import COLUMNS, load_dataset, prepare_features

from src.predict import predict_network

from src.security_report import generate_security_report

# ==========================================
# KONFIGURASI DASHBOARD
# ==========================================

st.set_page_config(page_title="AI-NIDS Dashboard", page_icon="🛡️", layout="wide")

FEATURE_COLUMNS = COLUMNS[:41]


# ==========================================
# FUNGSI MEMBACA DATA
# ==========================================


def read_uploaded_data(uploaded_file):

    raw = pd.read_csv(uploaded_file)

    if raw.empty:
        raise ValueError("Dataset kosong. Tidak ada data yang dapat dianalisis.")

    # Jika CSV memiliki header fitur
    if set(FEATURE_COLUMNS).issubset(raw.columns):
        pass

    else:
        # Jika CSV tidak memiliki header
        uploaded_file.seek(0)
        raw = pd.read_csv(uploaded_file, header=None)

        if raw.shape[1] == 43:
            raw.columns = COLUMNS

        elif raw.shape[1] == 41:
            raw.columns = FEATURE_COLUMNS

        else:
            raise ValueError(
                "CSV harus memiliki 41 fitur atau " "43 kolom format NSL-KDD."
            )

    if all(col in raw.columns for col in FEATURE_COLUMNS):
        X = raw[FEATURE_COLUMNS].copy()
    else:
        raise ValueError("Kolom fitur tidak lengkap.")

    y = None

    if "label" in raw.columns:
        X, y = prepare_features(raw)

    return raw, X, y


# ==========================================
# RISK LEVEL CLASSIFICATION
# ==========================================


def classify_risk(probability):

    if probability < 0.40:
        return "Low"

    elif probability < 0.70:
        return "Medium"

    else:
        return "High"


# ==========================================
# HEADER
# ==========================================

st.title("🛡️ AI-Based Network Intrusion Detection System")

st.write(
    "Sistem deteksi intrusi jaringan menggunakan "
    "algoritma Random Forest dan dataset NSL-KDD."
)

st.caption("Dataset-based detection | Random Forest | Threshold: 0.5")

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Menu")

menu = st.sidebar.radio(
    "Pilih halaman",
    [
        "Dashboard",
        "Deteksi Intrusi",
        "Evaluasi Model",
        "Informasi Model",
        "Model Explainability",
        "Riwayat Deteksi",
    ],
)


# ==========================================
# DATA INPUT
# ==========================================


@st.cache_data
def get_sample_data():
    _, test_data = load_dataset()
    return test_data


if "results" not in st.session_state:
    st.session_state.results = None

if "actual" not in st.session_state:
    st.session_state.actual = None

if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================
# HALAMAN DASHBOARD
# ==========================================

if menu == "Dashboard":

    st.header("Dashboard Overview")

    st.write(
        "Unggah dataset untuk melakukan analisis "
        "dan deteksi potensi intrusi jaringan."
    )

    uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

    use_sample = st.checkbox("Gunakan dataset pengujian KDDTest+")

    if st.button("Mulai Analisis", type="primary"):

        try:

            if uploaded_file is not None:
                raw, X, y = read_uploaded_data(uploaded_file)

            elif use_sample:
                raw = get_sample_data()
                X, y = prepare_features(raw)

            else:
                st.warning("Upload CSV atau pilih dataset KDDTest+.")
                st.stop()

            predictions = predict_network(X)

            result_df = pd.DataFrame(predictions)

            result_df = result_df.rename(
                columns={
                    "label": "Prediksi",
                    "prediction": "Kode Prediksi",
                    "probability": "Probabilitas Attack",
                }
            )

            result_df["Risk Level"] = result_df["Probabilitas Attack"].apply(
                classify_risk
            )

            result_df["Status"] = result_df["Prediksi"]

            if y is not None:
                result_df["Aktual"] = y.map({0: "Normal", 1: "Attack"}).values

                result_df["Jenis Serangan Aktual"] = (
                    raw["label"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .str.rstrip(".")
                    .values
                )

                st.session_state.actual = y.values

            else:
                st.session_state.actual = None

            st.session_state.results = result_df

            history_record = {
                "Waktu": pd.Timestamp.now().strftime("%d-%m-%Y %H:%M:%S"),
                "Total Data": len(result_df),
                "Normal": (result_df["Kode Prediksi"] == 0).sum(),
                "Attack": (result_df["Kode Prediksi"] == 1).sum(),
                "Low": (result_df["Risk Level"] == "Low").sum(),
                "Medium": (result_df["Risk Level"] == "Medium").sum(),
                "High": (result_df["Risk Level"] == "High").sum(),
            }

            st.session_state.history.append(history_record)

            st.success("Analisis berhasil dilakukan!")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

    if st.session_state.results is not None:

        result_df = st.session_state.results

        total = len(result_df)

        attack_count = (result_df["Kode Prediksi"] == 1).sum()

        normal_count = (result_df["Kode Prediksi"] == 0).sum()

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Data", f"{total:,}")
        col2.metric("Normal", f"{normal_count:,}")
        col3.metric("Attack", f"{attack_count:,}")

        st.subheader("Distribusi Hasil Deteksi")

        chart_data = pd.DataFrame(
            {"Kategori": ["Normal", "Attack"], "Jumlah": [normal_count, attack_count]}
        )

        st.bar_chart(chart_data.set_index("Kategori"))

        # DISTRIBUSI RISK LEVEL
        st.subheader("Distribusi Risk Level")

        risk_counts = (
            result_df["Risk Level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
        )

        risk_data = pd.DataFrame({"Jumlah": risk_counts})

        st.bar_chart(risk_data)

        st.subheader("Persentase Risk Level")

        risk_total = risk_counts.sum()

        risk_cols = st.columns(3)

        for col, level in zip(risk_cols, ["Low", "Medium", "High"]):
            count = risk_counts[level]
            percentage = count / risk_total * 100 if risk_total > 0 else 0

            col.metric(f"{level} ({percentage:.2f}%)", f"{count:,}")

        # ATTACK TYPE ANALYSIS
        if "Jenis Serangan Aktual" in result_df.columns:

            st.subheader("Analisis Jenis Serangan")

            attack_data = result_df[result_df["Jenis Serangan Aktual"] != "normal"]

            attack_analysis = attack_data.groupby("Jenis Serangan Aktual").agg(
                Total=("Jenis Serangan Aktual", "count"),
                Terdeteksi=("Kode Prediksi", "sum"),
            )

            attack_analysis["False Negative"] = (
                attack_analysis["Total"] - attack_analysis["Terdeteksi"]
            )

            attack_analysis["Detection Rate (%)"] = (
                attack_analysis["Terdeteksi"] / attack_analysis["Total"] * 100
            ).round(2)

            attack_analysis = attack_analysis.sort_values("Total", ascending=False)

            st.dataframe(attack_analysis, use_container_width=True)

        else:
            st.info(
                "Analisis jenis serangan membutuhkan dataset "
                "yang memiliki kolom label."
            )

        st.subheader("Hasil Prediksi")

        st.dataframe(result_df.head(100), use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Hasil Prediksi",
            data=csv,
            file_name="hasil_deteksi_ai_nids.csv",
            mime="text/csv",
        )

    # AUTOMATED SECURITY REPORT

        pdf_report = generate_security_report(result_df)

        st.download_button(
            "Download Automated Security Report (PDF)",
            data=pdf_report,
            file_name="AI_NIDS_Security_Report.pdf",
            mime="application/pdf",
        )


# ==========================================
# HALAMAN DETEKSI
# ==========================================

elif menu == "Deteksi Intrusi":

    st.header("Deteksi Intrusi Jaringan")

    if st.session_state.results is None:

        st.info(
            "Silakan lakukan analisis dataset pada "
            "halaman Dashboard terlebih dahulu."
        )

    else:

        result_df = st.session_state.results

        st.write("Ringkasan hasil klasifikasi:")

        attack = result_df[result_df["Kode Prediksi"] == 1]

        normal = result_df[result_df["Kode Prediksi"] == 0]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Attack")
            st.write(f"Jumlah: {len(attack)}")

        with col2:
            st.subheader("Normal")
            st.write(f"Jumlah: {len(normal)}")

        st.subheader("Detail Prediksi")

        st.dataframe(result_df, use_container_width=True)


# ==========================================
# HALAMAN EVALUASI
# ==========================================

elif menu == "Evaluasi Model":

    st.header("Evaluasi Model")

    result_df = st.session_state.results
    actual = st.session_state.actual

    if result_df is None:

        st.info("Jalankan analisis dataset terlebih dahulu.")

    elif actual is None:

        st.warning(
            "Data tidak memiliki label aktual. "
            "Metrik evaluasi dan confusion matrix "
            "tidak dapat dihitung."
        )

    else:

        y_true = actual

        y_pred = result_df["Kode Prediksi"].values

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")
        col4.metric("F1-Score", f"{f1:.4f}")

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        tn, fp, fn, tp = cm.ravel()

        st.subheader("Confusion Matrix")

        cm_df = pd.DataFrame(
            cm,
            index=["Aktual Normal", "Aktual Attack"],
            columns=["Prediksi Normal", "Prediksi Attack"],
        )

        st.dataframe(cm_df)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("TN", int(tn))
        col2.metric("FP", int(fp))
        col3.metric("FN", int(fn))
        col4.metric("TP", int(tp))


# ==========================================
# HALAMAN INFORMASI MODEL
# ==========================================

elif menu == "Informasi Model":

    st.header("Informasi Model")

    st.write("Algoritma: Random Forest")
    st.write("Dataset: NSL-KDD")
    st.write("Jumlah fitur: 41")
    st.write("Threshold: 0.5")

    st.write("Klasifikasi:")
    st.write("- 0 = Normal")
    st.write("- 1 = Attack")

    st.write("Evaluasi final pada KDDTest+:")

    col1, col2 = st.columns(2)

    col1.metric("Accuracy", "78.84%")
    col2.metric("Precision", "96.87%")

    col3, col4 = st.columns(2)

    col3.metric("Recall", "64.93%")
    col4.metric("F1-Score", "77.75%")

    st.info(
        "Nilai evaluasi di atas merupakan hasil "
        "pengujian final pada KDDTest+, bukan "
        "hasil prediksi dataset upload."
    )

# ==========================================
# HALAMAN MODEL EXPLAINABILITY
# ==========================================

elif menu == "Model Explainability":

    st.header("Model Explainability")

    st.write(
        "Halaman ini menampilkan fitur jaringan "
        "yang paling berpengaruh terhadap keputusan "
        "model Random Forest."
    )

    try:

        importance_df = pd.read_csv(
            "models/feature_importance.csv"
        )

        # Membersihkan nama fitur
        importance_df["Feature"] = (
            importance_df["feature"]
            .str.replace(
                r"^(remainder__|categorical__)",
                "",
                regex=True
            )
        )

        # Mengubah importance menjadi persentase
        importance_df["Importance (%)"] = (
            importance_df["importance"] * 100
        ).round(2)

        # Mengambil 10 fitur teratas
        top_features = (
            importance_df
            .sort_values(
                "Importance (%)",
                ascending=False
            )
            .head(10)
        )

        st.subheader("Top 10 Feature Importance")

        st.bar_chart(
            top_features.set_index("Feature")[
                "Importance (%)"
            ]
        )

        st.subheader("Detail Feature Importance")

        display_df = top_features[
            ["rank", "Feature", "Importance (%)"]
        ].rename(
            columns={
                "rank": "Ranking"
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # Fitur paling berpengaruh
        highest_feature = top_features.iloc[0]

        st.subheader("Interpretasi Model")

        st.metric(
            "Fitur Paling Berpengaruh",
            highest_feature["Feature"],
            f'{highest_feature["Importance (%)"]:.2f}%'
        )

        st.info(
            "Feature Importance menunjukkan kontribusi "
            "relatif setiap fitur dalam proses pengambilan "
            "keputusan Random Forest. Nilai yang lebih "
            "tinggi menunjukkan pengaruh yang lebih besar "
            "dalam model, tetapi tidak menunjukkan hubungan "
            "sebab-akibat."
        )

    except FileNotFoundError:

        st.error(
            "File feature_importance.csv tidak ditemukan."
        )

    except Exception as e:

        st.error(
            f"Gagal memuat Feature Importance: {e}"
        )

# ==========================================
# HALAMAN RIWAYAT DETEKSI
# ==========================================

elif menu == "Riwayat Deteksi":

    st.header("Riwayat Deteksi")

    if not st.session_state.history:

        st.info(
            "Belum ada riwayat analisis. "
            "Silakan lakukan analisis pada halaman Dashboard."
        )

    else:

        history_df = pd.DataFrame(st.session_state.history)

        st.write(f"Total analisis: {len(history_df)}")

        st.dataframe(history_df, use_container_width=True)

        csv_history = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Riwayat",
            data=csv_history,
            file_name="riwayat_deteksi_ai_nids.csv",
            mime="text/csv",
        )

        if st.button("Hapus Semua Riwayat"):
            st.session_state.history = []
            st.rerun()

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption("AI-NIDS | Network Intrusion Detection System " "| Random Forest")
