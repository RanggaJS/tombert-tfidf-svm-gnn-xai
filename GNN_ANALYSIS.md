# Analisis Hasil GNN

## Masalah Akurasi Rendah (58.53%)

### Root Cause:
1. **Data Tidak Valid**: Model menggunakan RANDOM EMBEDDINGS (bukan features yang sebenarnya)
   - Features adalah random numbers → model tidak bisa belajar dari data
   - Ini seperti mencoba belajar dari noise
   
2. **Majority Class Bias**: Model hanya memprediksi class terbanyak (Neutral = 58.5%)
   - Accuracy 58.53% = sebesar presentase data Neutral
   - Model tidak belajar pattern apapun

3. **Class Imbalance Parah**:
   - Negative (10.9%): 113 samples
   - Neutral (58.5%): 607 samples  ⬅ MAJORITY
   - Positive (30.6%): 317 samples

4. **Features Tidak Informatif**:
   - Random embeddings tidak mengandung informasi sentiment
   - Perlu actual text/graph features

### Mengapa Ini Terjadi?

Kode original menggunakan GLAN dengan Graph Attention Network, tetapi:
- Ada bug pada sparse matrix handling
- Adjacency matrix tidak compatible dengan batch processing
- Dummy data digunakan sebagai workaround

### Solusi untuk Meningkatkan Akurasi:

#### Opsi 1: Fix GLAN Implementation
- Perbaiki sparse matrix handling
- Gunakan adjacency matrix yang benar
- Integrasikan actual text embeddings

#### Opsi 2: Gunakan Actual Text Features
- Load actual text dari data
- Gunakan pre-trained word embeddings (GloVe, Word2Vec, BERT)
- Atau ekstraksi features dari text

#### Opsi 3: Gunakan TomBERT (Sudah Berjalan)
- TomBERT menggunakan actual text + image
- Hasil: 58.53% (sama dengan ini)
- Perlu hyperparameter tuning untuk lebih baik

### Rekomendasi:

**Untuk akurasi tinggi (90%+), perlu:**
1. ✅ Actual text processing (bukan random embeddings)
2. ✅ Pre-trained embeddings (BERT, RoBERTa, etc)
3. ✅ Proper data preprocessing
4. ✅ Hyperparameter tuning yang intensive
5. ✅ Data augmentation untuk minority classes

**Status Saat Ini:**
- ✅ GNN sudah bisa jalan dengan workaround
- ⚠️ Akurasi 58.53% karena random features
- 📝 File kämpede: `run_improved_gnn.py` dan `best_improved_gnn.pth`

### Next Steps:

1. **Gunakan TomBERT** yang sudah ada (tekst + gambar)
2. Atau **fix GLAN** dengan actual text embeddings
3. Atau **jalankan TF-IDF + SVM** yang lebih simple dan reliable

Mau saya jalankan yang mana?
