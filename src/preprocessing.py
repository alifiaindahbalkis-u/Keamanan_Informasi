
import pandas as pd
from pathlib import Path

# Lokasi dataset
DATASET_DIR = Path("dataset")

# Nama kolom NSL-KDD
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

# Fitur kategorikal
CATEGORICAL_FEATURES = [
    "protocol_type",
    "service",
    "flag"
]


def load_dataset():
    train = pd.read_csv(
        DATASET_DIR / "KDDTrain+.txt",
        header=None,
        names=COLUMNS
    )

    test = pd.read_csv(
        DATASET_DIR / "KDDTest+.txt",
        header=None,
        names=COLUMNS
    )

    return train, test


def prepare_features(data):
    # Mengubah label menjadi klasifikasi biner
    y = (data["label"] != "normal").astype(int)

    # Menghapus label dan difficulty dari fitur
    X = data.drop(columns=["label", "difficulty"])

    return X, y


if __name__ == "__main__":
    train, test = load_dataset()

    X_train, y_train = prepare_features(train)
    X_test, y_test = prepare_features(test)

    print("=== HASIL PREPROCESSING AWAL ===")

    print("\nData training:", X_train.shape)
    print("Data testing:", X_test.shape)

    print("\nDistribusi label training:")
    print(y_train.value_counts().rename(index={
        0: "Normal",
        1: "Attack"
    }))

    print("\nDistribusi label testing:")
    print(y_test.value_counts().rename(index={
        0: "Normal",
        1: "Attack"
    }))

    print("\nFitur kategorikal:")
    print(CATEGORICAL_FEATURES)

    print("\nJumlah fitur:", X_train.shape[1])

    print("\nPREPROCESSING AWAL BERHASIL")
