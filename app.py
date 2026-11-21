
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.preprocessing import (
    COLUMNS,
    load_dataset,
    prepare_features
)

from src.predict import predict_network


# ==========================================
# KONFIGURASI DASHBOARD
# ==========================================

st.set_page_config(
    page_title="AI-NIDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

FEATURE_COLUMNS = COLUMNS[:41]


# ==========================================
# FUNGSI MEMBACA DATA
# ==========================================

def read_uploaded_data(uploaded_file):

    raw = pd.read_csv(uploaded_file)

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
                "CSV harus memiliki 41 fitur atau "
                "43 kolom format NSL-KDD."
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
# HEADER
# ==========================================

st.title("🛡️ AI-Based Network Intrusion Detection System")

st.write(
    "Sistem deteksi intrusi jaringan menggunakan "
    "algoritma Random Forest dan dataset NSL-KDD."
)

st.caption(
    "Dataset-based detection | Random Forest | Threshold: 0.5"
)

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
        "Informasi Model"
    ]
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


# ==========================================
# HALAMAN DASHBOARD
# ==========================================

if menu == "Dashboard":

    st.header("Dashboard Overview")

    st.write(
        "Unggah dataset untuk melakukan analisis "
        "dan deteksi potensi intrusi jaringan."
    )

    uploaded_file = st.file_uploader(
        "Upload dataset CSV",
        type=["csv"]
    )

    use_sample = st.checkbox(
        "Gunakan dataset pengujian KDDTest+"
    )

    if st.button("Mulai Analisis", type="primary"):

        try:

            if uploaded_file is not None:
                raw, X, y = read_uploaded_data(uploaded_file)

            elif use_sample:
                raw = get_sample_data()
                X, y = prepare_features(raw)

            else:
                st.warning(
                    "Upload CSV atau pilih dataset KDDTest+."
                )
                st.stop()

            predictions = predict_network(X)

            result_df = pd.DataFrame(predictions)

            result_df = result_df.rename(
                columns={
                    "label": "Prediksi",
                    "prediction": "Kode Prediksi",
                    "probability": "Probabilitas Attack"
                }
            )

            result_df["Status"] = result_df[
                "Prediksi"
            ]

            if y is not None:
                result_df["Aktual"] = y.map(
                    {0: "Normal", 1: "Attack"}
                ).values

                st.session_state.actual = y.values

            else:
                st.session_state.actual = None

            st.session_state.results = result_df

            st.success("Analisis berhasil dilakukan!")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")


    if st.session_state.results is not None:

        result_df = st.session_state.results

        total = len(result_df)

        attack_count = (
            result_df["Kode Prediksi"] == 1
        ).sum()

        normal_count = (
            result_df["Kode Prediksi"] == 0
        ).sum()

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Data", f"{total:,}")
        col2.metric("Normal", f"{normal_count:,}")
        col3.metric("Attack", f"{attack_count:,}")

        st.subheader("Distribusi Hasil Deteksi")

        chart_data = pd.DataFrame({
            "Kategori": ["Normal", "Attack"],
            "Jumlah": [normal_count, attack_count]
        })

        st.bar_chart(
            chart_data.set_index("Kategori")
        )

        st.subheader("Hasil Prediksi")

        st.dataframe(
            result_df.head(100),
            use_container_width=True
        )

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Hasil Prediksi",
            data=csv,
            file_name="hasil_deteksi_ai_nids.csv",
            mime="text/csv"
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

        attack = result_df[
            result_df["Kode Prediksi"] == 1
        ]

        normal = result_df[
            result_df["Kode Prediksi"] == 0
        ]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Attack")
            st.write(f"Jumlah: {len(attack)}")

        with col2:
            st.subheader("Normal")
            st.write(f"Jumlah: {len(normal)}")

        st.subheader("Detail Prediksi")

        st.dataframe(
            result_df,
            use_container_width=True
        )


# ==========================================
# HALAMAN EVALUASI
# ==========================================

elif menu == "Evaluasi Model":

    st.header("Evaluasi Model")

    result_df = st.session_state.results
    actual = st.session_state.actual

    if result_df is None:

        st.info(
            "Jalankan analisis dataset terlebih dahulu."
        )

    elif actual is None:

        st.warning(
            "Data tidak memiliki label aktual. "
            "Metrik evaluasi dan confusion matrix "
            "tidak dapat dihitung."
        )

    else:

        y_true = actual

        y_pred = result_df[
            "Kode Prediksi"
        ].values

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred, zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, zero_division=0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")
        col4.metric("F1-Score", f"{f1:.4f}")

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        st.subheader("Confusion Matrix")

        cm_df = pd.DataFrame(
            cm,
            index=["Aktual Normal", "Aktual Attack"],
            columns=["Prediksi Normal", "Prediksi Attack"]
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

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "78.84%")
    col2.metric("Precision", "96.87%")
    col3.metric("Recall", "64.93%")
    col4.metric("F1-Score", "77.75%")

    st.info(
        "Nilai evaluasi di atas merupakan hasil "
        "pengujian final pada KDDTest+, bukan "
        "hasil prediksi dataset upload."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AI-NIDS | Network Intrusion Detection System "
    "| Random Forest"
)
