# Skrip Visualisasi Hasil Eksperimen

Skrip `visualize_results.py` digunakan untuk memvisualisasikan dan membandingkan hasil dari ketiga metode yang digunakan dalam penelitian:
- **TomBERT** (Multimodal BERT)
- **TF-IDF + SVM** (Classical Method)
- **GNN** (Graph Neural Network untuk Rumor Detection)

## Cara Penggunaan

### 1. Penggunaan Default (Auto-detect paths)

```bash
python run_program/visualize_results.py
```

Skrip akan otomatis mencari file hasil di lokasi default:
- TomBERT: `output/tombert_ultra_optimized_20251129_225613/ultra_results.json`
- TF-IDF+SVM: `output/tfidf_svm_ultra_optimized_20251209_111442/ultra_results.json`
- GNN: `output/gnn_optimized_twitter2015_rumor/results.json`

### 2. Custom Paths

Jika file hasil berada di lokasi berbeda:

```bash
python run_program/visualize_results.py \
  --tombert output/tombert_ultra_optimized_*/ultra_results.json \
  --tfidf output/tfidf_svm_ultra_optimized_*/ultra_results.json \
  --gnn output/gnn_optimized_twitter2015_rumor/results.json \
  --output output/visualizations
```

### 3. Hanya Visualisasi Metode Tertentu

Jika hanya ingin memvisualisasikan beberapa metode:

```bash
# Hanya TomBERT dan GNN
python run_program/visualize_results.py \
  --tombert output/tombert_ultra_optimized_*/ultra_results.json \
  --gnn output/gnn_optimized_twitter2015_rumor/results.json
```

## Output yang Dihasilkan

Skrip akan menghasilkan 5 file visualisasi di folder `output/visualizations/`:

1. **comparison_metrics.png**
   - Perbandingan 4 metrik utama: Accuracy, F1-Score, Training Time, Precision vs Recall
   - Format: 2x2 subplot dengan bar charts

2. **radar_chart.png**
   - Radar chart multi-dimensi untuk perbandingan komprehensif
   - Metrik: Accuracy, F1-Score, Precision, Recall, Efficiency

3. **training_time_comparison.png**
   - Perbandingan waktu training dalam format horizontal bar chart
   - Menampilkan waktu dalam jam atau menit

4. **summary_table.png**
   - Tabel ringkasan semua metrik dalam format tabel
   - Mudah dibaca untuk dokumentasi

5. **gnn_detailed_metrics.png** (jika GNN tersedia)
   - Detail metrik GNN untuk rumor detection
   - Precision, Recall, F1-Score untuk kelas Rumor
   - Optimal threshold yang digunakan

## Format File Input

### TomBERT Results (ultra_results.json)
```json
{
  "final_test_accuracy": 0.8608,
  "final_test_f1": 0.8149,
  "final_test_precision": 0.8226,
  "final_test_recall": 0.8081,
  "training_time_hours": 4.71
}
```

### TF-IDF+SVM Results (ultra_results.json)
```json
{
  "test_accuracy": 0.6413,
  "dev_accuracy": 0.6275,
  "total_training_time": 28594.59
}
```

### GNN Results (results.json)
```json
{
  "test_accuracy": 0.8804,
  "test_f1_macro": 0.5702,
  "test_f1_rumor": 0.2051,
  "test_precision_rumor": 0.1270,
  "test_recall_rumor": 0.5333,
  "training_time": 29.70,
  "optimal_threshold": 0.75
}
```

## Dependencies

Pastikan dependencies berikut sudah terinstall:

```bash
pip install matplotlib seaborn numpy
```

## Contoh Output

Setelah menjalankan skrip, Anda akan mendapatkan visualisasi seperti:

- **Bar charts** dengan nilai metrik yang jelas
- **Radar chart** untuk perbandingan multi-dimensi
- **Tabel ringkasan** yang mudah dibaca
- **Detail metrik** untuk analisis mendalam

Semua file disimpan dalam format PNG dengan resolusi tinggi (300 DPI) untuk kualitas cetak yang baik.

## Troubleshooting

### Error: File tidak ditemukan
- Pastikan path ke file hasil benar
- Gunakan `--tombert`, `--tfidf`, `--gnn` untuk specify custom paths

### Error: No module named 'matplotlib'
```bash
pip install matplotlib seaborn numpy
```

### Visualisasi kosong
- Pastikan file JSON berisi data yang valid
- Check log untuk melihat metode mana yang berhasil di-load

## Integrasi dengan Workflow

Skrip ini dapat dijalankan setelah semua eksperimen selesai:

```bash
# 1. Run semua eksperimen
python run_program/run_tombert_only.py
python run_program/run_tfidf_svm_only.py
python run_program/run_gnn_only.py

# 2. Generate visualizations
python run_program/visualize_results.py

# 3. Visualisasi tersimpan di output/visualizations/
```

