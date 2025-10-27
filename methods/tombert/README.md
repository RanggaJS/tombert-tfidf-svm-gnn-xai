# TomBERT - Target-oriented Multimodal BERT

## Deskripsi
Implementasi TomBERT untuk analisis sentimen multimodal pada dataset Twitter15.

## File Utama
- `run_multimodal_classifier.py` - Script utama untuk training dan evaluasi TomBERT
- `run_tombert_gpu.py` - Script khusus untuk GPU
- `run_multimodal_classifier_test.sh` - Script shell untuk menjalankan eksperimen

## Cara Menjalankan
```bash
# Training TomBERT
python run_multimodal_classifier.py \
    --data_dir ./absa_data/twitter2015 \
    --task_name twitter2015 \
    --output_dir ./output/tombert_test \
    --bert_model bert-base-uncased \
    --do_train \
    --do_eval \
    --mm_model TomBert \
    --pooling first \
    --max_seq_length 64 \
    --max_entity_length 16 \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 1

# Atau menggunakan GPU
python run_tombert_gpu.py
```

## Hasil
- Model tersimpan di: `./output/tombert_test/`
- Hasil evaluasi: `eval_results.txt`
- Prediksi: `pred.txt`
- Label sebenarnya: `true.txt`

## Dependencies
- PyTorch
- Transformers
- BERT-base-uncased
- ResNet-152 untuk ekstraksi fitur gambar
