# Penjelasan Diagram Alir untuk BAB 4 Perancangan

## 1. Diagram Alir TomBERT (Target-Oriented Multimodal BERT)

### Alur Proses:

1. **Input**: Sistem menerima tiga jenis input:
   - Teks tweet (input_ids)
   - Gambar terkait (visual_embeds_att)
   - Entity/target (s2_input_ids)

2. **Tokenisasi**:
   - Teks di-tokenize menjadi sequence tokens
   - Entity di-tokenize secara terpisah

3. **Encoding**:
   - **BERT Encoder**: Meng-encode teks utama menjadi H^s (hidden states)
   - **s2_BERT Encoder**: Meng-encode entity menjadi H^e (entity embeddings)
   - **ResNet-152**: Mengekstrak fitur visual dari gambar (2048×49 feature map)

4. **Cross-Attention Entity-Image**:
   - Entity embeddings (H^e) melakukan cross-attention dengan visual features
   - Menghasilkan H^v (visual-aware entity representation)

5. **Multimodal Fusion**:
   - Menggabungkan H^v dan H^s menjadi [H^v; H^s]
   - Representasi multimodal yang menggabungkan informasi teks dan gambar

6. **Combined Attention Encoder**:
   - Multi-layer transformer encoder dengan attention mechanism
   - Memproses representasi multimodal untuk menghasilkan context-aware features

7. **Pooling**:
   - Pooling strategi (concat/cls/first) untuk menghasilkan fixed-size representation

8. **Classification**:
   - Linear classifier menghasilkan logits untuk 3 kelas sentimen (Negative, Neutral, Positive)

### Keunggulan:
- Target-oriented: Fokus pada entity spesifik dalam teks
- Multimodal: Menggabungkan informasi teks dan gambar
- Attention mechanism: Memungkinkan model fokus pada bagian penting

---

## 2. Diagram Alir TF-IDF + SVM

### Alur Proses:

1. **Input**: Teks tweet dan gambar terkait

2. **Text Preprocessing**:
   - Cleaning: Menghapus noise, URL, mention
   - Tokenization: Memecah teks menjadi tokens
   - Stemming: Mengurangi kata ke bentuk dasar
   - Lemmatization: Mengubah kata ke bentuk lemma

3. **Feature Extraction**:
   - **TF-IDF Vectorization**:
     - N-gram range: 1-4 (unigram, bigram, trigram, quadgram)
     - Max features: 200,000
     - Menghasilkan sparse matrix representasi
   - **Image Feature Extraction**:
     - Histogram features
     - Texture features (GLCM)
     - Shape features
     - Total: 20-50 features per gambar
   - **Sentiment Features**:
     - Polarity score (TextBlob)
     - Subjectivity score
     - VADER sentiment scores

4. **Feature Combination**:
   - Menggabungkan semua fitur: TF-IDF + Image + Sentiment
   - Menghasilkan feature vector lengkap

5. **Optional Processing**:
   - PCA: Dimensionality reduction
   - Feature Selection: Memilih fitur paling informatif
   - Scaling: Normalisasi fitur (StandardScaler, RobustScaler)

6. **SVM Classification**:
   - Support Vector Machine dengan hyperparameter tuning
   - Output: 3 kelas sentimen

### Keunggulan:
- Interpretable: Fitur dapat dijelaskan secara manual
- Efisien: Training cepat dibanding deep learning
- Robust: Tidak memerlukan GPU untuk training

---

## 3. Diagram Alir GNN (Graph Neural Network) untuk Deteksi Rumor

### Alur Proses:

1. **Input**: Teks tweet dan metadata (user, timestamp, dll)

2. **Graph Construction**:
   - **Nodes**: Setiap tweet/user menjadi node dalam graph
   - **Edges**: Hubungan antar node berdasarkan:
     - Retweet relationship
     - Reply relationship
     - Follow relationship
   - Menghasilkan graph structure G = (V, E)

3. **Node Feature Extraction**:
   - TF-IDF vectorization dari teks tweet
   - Setiap node memiliki feature vector berdasarkan konten teksnya

4. **Adjacency Matrix**:
   - Representasi matriks dari struktur graph
   - Menunjukkan hubungan (edges) antar node

5. **GAT Layers (Graph Attention Network)**:
   - **Multi-Head Attention (Layer 1)**:
     - Setiap node mengaggregate informasi dari neighbor nodes
     - Attention mechanism menentukan pentingnya setiap neighbor
     - Multiple attention heads untuk menangkap berbagai aspek relasi
   - **Output Attention (Layer 2)**:
     - Final attention layer untuk menghasilkan node embeddings
     - Menggabungkan informasi dari semua layers sebelumnya

6. **Node Embeddings**:
   - Setiap node memiliki representasi yang memperhitungkan:
     - Fitur node itu sendiri
     - Fitur dari neighbor nodes
     - Struktur graph secara keseluruhan

7. **Classification**:
   - Linear classifier menggunakan node embeddings
   - Output: Binary classification (Rumor / Non-Rumor)

### Keunggulan:
- Graph structure: Memanfaatkan hubungan sosial antar tweet
- Attention mechanism: Fokus pada node/edge yang relevan
- Propagation: Informasi menyebar melalui graph structure

---

## 4. Diagram Alir XAI (Explainable AI) dengan GPT + BLIP

### Alur Proses:

1. **Input**: Prediksi dari model (TomBERT/TF-IDF+SVM/GNN)
   - Teks original
   - Label prediksi
   - Probabilitas untuk setiap kelas
   - Image ID (jika tersedia)

2. **BLIP Image Captioning** (jika ada gambar):
   - Model BLIP (Bootstrapping Language-Image Pre-training) digunakan
   - Menggenerate caption/deskripsi dari gambar
   - Menghasilkan deskripsi tekstual dari konten visual

3. **Text Processing**:
   - Mengambil teks original tweet
   - Mempertahankan format asli untuk konteks

4. **Context Assembly**:
   - Menggabungkan semua informasi:
     - Teks original
     - Image caption (dari BLIP)
     - Label prediksi
     - Probabilitas prediksi
     - Ground truth label (jika tersedia)

5. **GPT Prompt Construction**:
   - **System Prompt**: Instruksi untuk GPT sebagai asisten XAI
   - **User Prompt**: Berisi semua context yang telah di-assemble
   - Format prompt yang jelas dan terstruktur

6. **GPT-4o-mini API Call**:
   - Mengirim prompt ke OpenAI API
   - Model: gpt-4o-mini (cost-effective)
   - Temperature: 0.2 (deterministic)
   - Max tokens: 320 (penjelasan singkat)

7. **Output**: Penjelasan XAI dalam Bahasa Indonesia
   - Menjelaskan alasan model memilih label tertentu
   - Menyebutkan kata/frasa kunci yang mempengaruhi prediksi
   - Mempertimbangkan konteks gambar (jika ada)

### Keunggulan:
- Multimodal explanation: Mempertimbangkan teks dan gambar
- Natural language: Penjelasan mudah dipahami
- Context-aware: Menggunakan informasi lengkap dari prediksi

---

## Cara Menggunakan Diagram untuk BAB 4

1. **Sisipkan gambar** ke dalam dokumen Word BAB 4 Perancangan
2. **Tambahkan caption** di bawah setiap diagram:
   - "Gambar X.X: Diagram Alir TomBERT"
   - "Gambar X.X: Diagram Alir TF-IDF + SVM"
   - "Gambar X.X: Diagram Alir GNN"
   - "Gambar X.X: Diagram Alir XAI"
3. **Tambahkan penjelasan** menggunakan teks di atas sebagai referensi
4. **Sesuaikan nomor gambar** sesuai dengan struktur dokumen

## Lokasi File Diagram

Semua diagram tersimpan di: `output/diagrams/`
- `diagram_alir_tombert.png`
- `diagram_alir_tfidf_svm.png`
- `diagram_alir_gnn.png`
- `diagram_alir_xai.png`

Semua file dalam format PNG dengan resolusi tinggi (300 DPI) untuk kualitas cetak yang baik.

