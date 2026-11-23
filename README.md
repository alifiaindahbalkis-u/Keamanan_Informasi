# AI-Based Network Intrusion Detection System (AI-NIDS)

## 1. Deskripsi Proyek

AI-Based Network Intrusion Detection System (AI-NIDS) merupakan sistem deteksi intrusi jaringan berbasis Machine Learning yang dikembangkan untuk mengidentifikasi aktivitas jaringan ke dalam dua kategori, yaitu **Normal** dan **Attack**.

Proyek ini menggunakan algoritma **Random Forest** dengan dataset **NSL-KDD** sebagai data pelatihan dan pengujian. Model yang telah dilatih diintegrasikan ke dalam aplikasi berbasis web menggunakan Streamlit sehingga pengguna dapat melakukan analisis dataset, melihat hasil klasifikasi, mengevaluasi performa model, dan mengunduh hasil prediksi.

Sistem ini berfokus pada klasifikasi aktivitas jaringan berdasarkan 41 fitur yang terdapat dalam dataset NSL-KDD.

## 2. Tujuan Proyek

Tujuan pengembangan sistem AI-NIDS adalah:

1. Mengimplementasikan algoritma Machine Learning untuk mendeteksi potensi intrusi jaringan.
2. Mengklasifikasikan aktivitas jaringan menjadi Normal dan Attack.
3. Mengintegrasikan model Random Forest ke dalam dashboard berbasis web.
4. Menyediakan visualisasi hasil deteksi secara interaktif.
5. Menyediakan fitur unggah dataset dan unduh hasil prediksi dalam format CSV.
6. Menampilkan hasil evaluasi model berdasarkan data pengujian.

## 3. Teknologi yang Digunakan

Proyek ini dikembangkan menggunakan beberapa teknologi berikut:

| Teknologi | Kegunaan |
|---|---|
| Python 3.11 | Bahasa pemrograman utama |
| Pandas | Pengolahan dan manipulasi dataset |
| NumPy | Operasi numerik |
| Scikit-learn | Implementasi dan evaluasi Machine Learning |
| Matplotlib | Visualisasi data |
| Seaborn | Visualisasi statistik |
| Streamlit | Pengembangan dashboard web |
| Joblib | Penyimpanan dan pemuatan model |

**Algoritma:** Random Forest Classifier

**Dataset:** NSL-KDD

## 4. Dataset

Dataset yang digunakan dalam proyek ini adalah NSL-KDD, yang terdiri dari dua bagian utama:

- `KDDTrain+`: Dataset untuk pelatihan model.
- `KDDTest+`: Dataset untuk pengujian model.

Dataset memiliki 43 kolom, yang terdiri dari:

- 41 fitur aktivitas jaringan.
- 1 kolom label.
- 1 kolom difficulty.

Dalam proses preprocessing, kolom `label` dikonversi menjadi klasifikasi biner, sedangkan kolom `difficulty` tidak digunakan sebagai fitur model.

### Kategori Klasifikasi

| Kode | Kategori |
|---|---|
| 0 | Normal |
| 1 | Attack |

### Fitur Kategorikal

Terdapat tiga fitur kategorikal utama:

- `protocol_type`
- `service`
- `flag`

Ketiga fitur tersebut diproses sebagai bagian dari pipeline preprocessing model.

## 5. Struktur Direktori

Struktur direktori proyek AI-NIDS adalah sebagai berikut:

```text
AI-NIDS/
│
├── dataset/
│   ├── KDDTrain+.txt
│   ├── KDDTest+.txt
│   └── KDDTest+.csv
│
├── models/
│   ├── final_model.pkl
│   ├── random_forest_pipeline.pkl
│   ├── final_evaluation.csv
│   ├── optimization_results.csv
│   └── file hasil analisis model lainnya
│
├── src/
│   ├── attack_analysis.py
│   ├── check_dataset.py
│   ├── error_analysis.py
│   ├── feature_analysis.py
│   ├── final_train.py
│   ├── inspect_dataset.py
│   ├── model_comparison.py
│   ├── optimize.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── test_predict.py
│   ├── train.py
│   └── validate_final_model.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 6. Instalasi dan Persiapan Lingkungan

### Persyaratan

Sebelum menjalankan proyek, pastikan perangkat telah memiliki:

- Python 3.11
- pip
- Git
- Visual Studio Code (opsional)

### Langkah Instalasi

**1. Clone Repository**

```bash
git clone https://github.com/alifiaindahbalkis-u/Keamanan_Informasi.git
```

Masuk ke direktori proyek:

```bash
cd Keamanan_Informasi
```

**2. Membuat Virtual Environment**

```bash
python -m venv venv
```

**3. Mengaktifkan Virtual Environment**

Untuk Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

**4. Menginstal Dependencies**

```bash
pip install -r requirements.txt
```

**5. Memastikan Dataset dan Model Tersedia**

Pastikan file dataset berada di folder `dataset/` dan model final tersedia di:

```text
models/final_model.pkl
```

## 7. Menjalankan Aplikasi

Setelah seluruh dependensi terinstal dan file proyek tersedia, jalankan aplikasi menggunakan perintah:

```bash
python -m streamlit run app.py
```

Streamlit akan menjalankan server lokal dan memberikan alamat dashboard yang dapat dibuka melalui browser.

## 8. Fitur Dashboard

Aplikasi AI-NIDS menyediakan empat halaman utama.

### 8.1 Dashboard Overview

Halaman utama yang menampilkan ringkasan hasil analisis dataset.

Informasi yang ditampilkan meliputi:

- Total data yang dianalisis.
- Jumlah aktivitas Normal.
- Jumlah aktivitas Attack.
- Grafik distribusi hasil deteksi.
- Tabel hasil prediksi.
- Tombol unduh hasil prediksi.

### 8.2 Deteksi Intrusi

Halaman ini menampilkan hasil klasifikasi aktivitas jaringan berdasarkan kategori Normal dan Attack.

Pengguna dapat melihat jumlah data yang diklasifikasikan ke dalam setiap kategori serta detail hasil prediksi.

### 8.3 Evaluasi Model

Halaman evaluasi digunakan untuk melihat performa model berdasarkan data yang memiliki label aktual.

Metrik yang ditampilkan meliputi:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Evaluasi hanya dapat dilakukan apabila dataset yang digunakan memiliki label aktual.

### 8.4 Informasi Model

Halaman ini menampilkan informasi mengenai model yang digunakan, meliputi:

- Algoritma: Random Forest.
- Dataset: NSL-KDD.
- Jumlah fitur: 41.
- Threshold: 0.5.
- Kategori klasifikasi: Normal dan Attack.
- Hasil evaluasi final model.

## 9. Fitur Upload dan Download Dataset

### Upload Dataset

Dashboard menyediakan fitur unggah dataset dalam format CSV.

Format input yang didukung:

- CSV dengan header dan 41 fitur.
- CSV dengan 43 kolom sesuai format NSL-KDD, termasuk label dan difficulty.
- CSV tanpa header dengan 41 atau 43 kolom.

Jika dataset memiliki label aktual, sistem dapat menggunakan label tersebut untuk menghitung metrik evaluasi.

Jika dataset tidak memiliki label aktual, sistem tetap dapat melakukan prediksi, tetapi metrik evaluasi tidak ditampilkan.

Selain fitur upload, tersedia opsi untuk menggunakan dataset pengujian KDDTest+ yang telah disediakan di dalam proyek.

### Download Hasil Prediksi

Setelah proses analisis selesai, pengguna dapat mengunduh hasil deteksi melalui tombol **Download Hasil Prediksi**.

Hasil unduhan disimpan dalam format CSV dengan nama:

```text
hasil_deteksi_ai_nids.csv
```

Kolom hasil prediksi meliputi:

| Kolom | Keterangan |
|---|---|
| Kode Prediksi | Kode klasifikasi 0 atau 1 |
| Prediksi | Hasil klasifikasi Normal atau Attack |
| Probabilitas Attack | Nilai probabilitas prediksi Attack |
| Status | Kategori hasil deteksi |
| Aktual | Label sebenarnya jika tersedia |

## 10. Evaluasi Model

Berdasarkan hasil pengujian final menggunakan dataset KDDTest+, model Random Forest memperoleh hasil evaluasi sebagai berikut:

| Metrik | Hasil |
|---|---:|
| Accuracy | 78.84% |
| Precision | 96.87% |
| Recall | 64.93% |
| F1-Score | 77.75% |

Threshold klasifikasi yang digunakan adalah **0.5**.

### Confusion Matrix

Hasil evaluasi model menghasilkan confusion matrix berikut:

| Aktual / Prediksi | Normal | Attack |
|---|---:|---:|
| Normal | 9.442 | 269 |
| Attack | 4.501 | 8.332 |

Keterangan:

- True Negative (TN): 9.442
- False Positive (FP): 269
- False Negative (FN): 4.501
- True Positive (TP): 8.332

Hasil tersebut menunjukkan bahwa model memiliki precision yang tinggi dalam mengidentifikasi aktivitas Attack, tetapi masih terdapat aktivitas Attack yang diklasifikasikan sebagai Normal.

## 11. Hasil Pengujian Dashboard

Pengujian integrasi dilakukan untuk memastikan model dapat digunakan melalui dashboard Streamlit.

Pengujian menggunakan dataset KDDTest+ dengan total 22.544 data dan 41 fitur.

Hasil prediksi yang diperoleh:

| Kategori | Jumlah |
|---|---:|
| Normal | 13.943 |
| Attack | 8.601 |
| Total | 22.544 |

Pengujian yang telah dilakukan:

1. Pengujian pemuatan model final.
2. Pengujian preprocessing dataset.
3. Pengujian prediksi seluruh data KDDTest+.
4. Pengujian penggunaan dataset bawaan.
5. Pengujian upload dataset CSV.
6. Pengujian visualisasi hasil deteksi.
7. Pengujian download hasil prediksi.
8. Pemeriksaan isi file CSV hasil ekspor.

Hasil pengujian menunjukkan bahwa dashboard berhasil memproses dataset dan menghasilkan file prediksi dalam format CSV.

## 12. Pengembang

Proyek AI-Based Network Intrusion Detection System (AI-NIDS) dikembangkan secara berkelompok sebagai bagian dari tugas mata kuliah **Keamanan Informasi**.

Universitas Siliwangi  
Fakultas Teknik  
Program Studi Informatika

---

**AI-Based Network Intrusion Detection System (AI-NIDS)**  
*Machine Learning-Based Network Intrusion Detection Using Random Forest and NSL-KDD Dataset.*