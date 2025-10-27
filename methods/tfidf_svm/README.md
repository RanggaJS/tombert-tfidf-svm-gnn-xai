# TF-IDF + SVM - Metode Klasik

## Deskripsi
Implementasi metode klasik TF-IDF + SVM untuk analisis sentimen multimodal dengan ekstraksi fitur teks dan gambar.

## File Utama
- `classical_methods.py` - Implementasi TF-IDF + SVM dengan ResNet untuk fitur gambar
- `run_experiment_without_images.py` - Script untuk eksperimen tanpa gambar (text-only)

## Fitur
- **Text Features**: TF-IDF vectorization dengan n-gram (1,2)
- **Image Features**: Ekstraksi fitur menggunakan ResNet-152
- **Classifier**: SVM dengan kernel RBF
- **Multimodal Fusion**: Kombinasi fitur teks dan gambar

## Cara Menjalankan
```bash
# Dengan gambar (multimodal)
python classical_methods.py

```

## Konfigurasi
- Max features: 5000
- N-gram range: (1, 2)
- SVM kernel: RBF
- Image size: 224x224
- ResNet model: resnet152.pth

## Output
- Model tersimpan sebagai pickle file
- Hasil evaluasi: accuracy, precision, recall, f1-score
- Classification report detail
