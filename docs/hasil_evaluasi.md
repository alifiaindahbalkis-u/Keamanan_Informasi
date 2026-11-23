# Hasil Evaluasi Model AI-NIDS

## 1. Dataset Pengujian

Evaluasi model dilakukan menggunakan dataset NSL-KDD, yaitu `KDDTest+`.

- Jumlah data pengujian: 22.544 baris
- Jumlah fitur: 41
- Algoritma: Random Forest
- Threshold klasifikasi: 0,5
- Klasifikasi: Normal (0) dan Attack (1)

## 2. Hasil Evaluasi

| Metrik | Nilai |
|---|---:|
| Accuracy | 78,84% |
| Precision | 96,87% |
| Recall | 64,93% |
| F1-Score | 77,75% |

## 3. Confusion Matrix

| Aktual / Prediksi | Normal | Attack |
|---|---:|---:|
| Normal | 9.442 | 269 |
| Attack | 4.501 | 8.332 |

Keterangan:
- True Negative (TN): 9.442
- False Positive (FP): 269
- False Negative (FN): 4.501
- True Positive (TP): 8.332

## 4. Analisis Kesalahan Prediksi

### False Positive (FP)

Sebanyak 269 data Normal salah diklasifikasikan sebagai Attack. Kondisi ini dapat menyebabkan aktivitas jaringan normal dianggap sebagai ancaman.

### False Negative (FN)

Sebanyak 4.501 data Attack salah diklasifikasikan sebagai Normal. Kondisi ini perlu menjadi perhatian karena serangan jaringan berpotensi tidak terdeteksi.

## 5. Kesimpulan

Model Random Forest menghasilkan precision sebesar 96,87%, yang menunjukkan bahwa sebagian besar prediksi Attack yang dihasilkan model memang merupakan data Attack.

Namun, recall sebesar 64,93% menunjukkan bahwa model masih belum berhasil mendeteksi seluruh data Attack. Jumlah False Negative yang relatif tinggi menjadi aspek utama yang perlu diperhatikan untuk pengembangan model selanjutnya.

## 6. Analisis False Negative Berdasarkan Jenis Serangan

Analisis lebih lanjut dilakukan untuk mengetahui jenis serangan yang paling banyak gagal dikenali oleh model.

Beberapa hasil dengan jumlah False Negative tertinggi adalah:

| Jenis Serangan | Total Data | False Negative | Miss Rate |
|---|---:|---:|---:|
| guess_passwd | 1.231 | 1.231 | 100,00% |
| warezmaster | 944 | 782 | 82,84% |
| processtable | 685 | 676 | 98,69% |
| mscan | 996 | 438 | 43,98% |
| snmpguess | 331 | 331 | 100,00% |
| apache2 | 737 | 296 | 40,16% |
| mailbomb | 293 | 293 | 100,00% |
| snmpgetattack | 178 | 178 | 100,00% |

Jenis serangan `guess_passwd`, `snmpguess`, `mailbomb`, dan `snmpgetattack` memiliki miss rate sebesar 100% pada dataset pengujian ini. Artinya, seluruh data dari masing-masing kategori tersebut salah diklasifikasikan sebagai Normal.

Sebaliknya, jenis serangan `neptune` memiliki 2 False Negative dari 4.657 data atau miss rate sebesar 0,04%.

Hasil ini menunjukkan bahwa kemampuan deteksi model berbeda-beda pada setiap jenis serangan. Oleh karena itu, evaluasi berdasarkan kelas serangan diperlukan selain menggunakan metrik evaluasi keseluruhan.

## 7. File Hasil Analisis

Script `src/error_analysis.py` menghasilkan tiga file:

| File | Isi |
|---|---|
| `models/false_positive.csv` | Data Normal yang diprediksi sebagai Attack |
| `models/false_negative.csv` | Data Attack yang diprediksi sebagai Normal |
| `models/false_negative_by_attack.csv` | Rekap False Negative dan miss rate setiap jenis serangan |

## 8. Kesimpulan Analisis Kesalahan

Hasil evaluasi menunjukkan bahwa model memiliki precision tinggi, tetapi masih terdapat sejumlah serangan yang tidak terdeteksi. False Negative menjadi perhatian utama, khususnya pada kategori serangan yang memiliki miss rate tinggi.

Analisis per jenis serangan dapat digunakan sebagai dasar pengembangan model berikutnya, misalnya melalui evaluasi fitur, penanganan ketidakseimbangan kelas, dan optimasi model.

