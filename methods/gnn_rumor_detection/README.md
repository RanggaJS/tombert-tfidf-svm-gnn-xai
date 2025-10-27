# GNN Rumor Detection - Graph Neural Network

## Deskripsi
Implementasi Graph Neural Network (GAT/GLAN) untuk deteksi rumor pada dataset Twitter15.

## File Utama
- `run_gnn_gpu.py` - Script untuk menjalankan GNN dengan GPU support

## Model GNN
- **GAT (Graph Attention Network)**: Menggunakan attention mechanism
- **GLAN (Graph Learning Attention Network)**: Variasi GAT dengan learning attention

## Fitur
- Graph construction dari data Twitter
- Node features dari teks dan gambar
- Edge features dari hubungan antar tweet
- Attention mechanism untuk fokus pada node penting

## Cara Menjalankan
```bash
# Jalankan GNN dengan GPU
python run_gnn_gpu.py

# Atau dengan konfigurasi khusus
python run_gnn_gpu.py --model GAT --epochs 50 --lr 0.001
```

## Konfigurasi
- Model: GAT atau GLAN
- Hidden dimensions: 64, 128, 256
- Attention heads: 4, 8
- Learning rate: 0.001
- Epochs: 50

## Output
- Model tersimpan di: `./output/gnn_models/`
- Hasil evaluasi: accuracy, f1-score
- Graph visualization
- Attention weights
