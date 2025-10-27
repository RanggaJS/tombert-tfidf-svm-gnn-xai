# XAI - Explainable AI Methods

## Deskripsi
Implementasi metode Explainable AI untuk interpretasi model analisis sentimen multimodal.

## File Utama
- `xai_methods.py` - Implementasi LIME, SHAP, dan attention visualization

## Metode XAI
- **LIME (Local Interpretable Model-agnostic Explanations)**: Interpretasi lokal
- **SHAP (SHapley Additive exPlanations)**: Interpretasi global
- **Attention Visualization**: Visualisasi attention weights dari TomBERT
- **Feature Importance**: Analisis pentingnya fitur

## Fitur
- Text explanation dengan LIME
- Image explanation dengan LIME
- SHAP values untuk feature importance
- Attention heatmaps
- Feature importance plots

## Cara Menjalankan
```bash
# Jalankan analisis XAI
python xai_methods.py

# Atau import sebagai module
from xai_methods import XAIAnalyzer
```

## Output
- LIME explanations: HTML files
- SHAP plots: PNG files
- Attention maps: PNG files
- Feature importance charts: PNG files

## Dependencies
- lime
- shap
- matplotlib
- seaborn
- plotly
