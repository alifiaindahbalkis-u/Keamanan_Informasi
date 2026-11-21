
import pandas as pd
from pathlib import Path

# Lokasi dataset
DATASET_DIR = Path("dataset")

# Nama kolom NSL-KDD (41 fitur + 1 label + 1 difficulty)
columns = [
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

# Membaca dataset
train = pd.read_csv(
    DATASET_DIR / "KDDTrain+.txt",
    header=None,
    names=columns
)

test = pd.read_csv(
    DATASET_DIR / "KDDTest+.txt",
    header=None,
    names=columns
)

# Menampilkan informasi dataset
print("=== INFORMASI DATASET NSL-KDD ===")

print("\nJumlah data training:", train.shape)
print("Jumlah data testing:", test.shape)

print("\nNama kolom:")
print(train.columns.tolist())

print("\nDistribusi label training:")
print(train["label"].value_counts())

print("\nDistribusi label testing:")
print(test["label"].value_counts())

print("\nContoh data training:")
print(train.head())
