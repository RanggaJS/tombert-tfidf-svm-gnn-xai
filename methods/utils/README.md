# Utils - Utilities dan Shared Code

## Deskripsi
Utility functions dan shared code yang digunakan oleh semua metode.

## File Utama
- `gpu_config.py` - Konfigurasi GPU dan device management
- `comparison_framework.py` - Framework untuk perbandingan hasil eksperimen

## Fitur

### GPU Configuration
- Deteksi GPU availability
- Device management
- CUDA configuration
- Memory optimization

### Comparison Framework
- Performance comparison
- Visualization generation
- Report creation
- Metrics calculation

## Cara Menggunakan
```python
# GPU Configuration
from utils.gpu_config import get_device, set_cuda_visible_devices

device = get_device()
set_cuda_visible_devices(0)

# Comparison Framework
from utils.comparison_framework import PerformanceComparator

comparator = PerformanceComparator()
comparator.add_results("TomBERT", metrics)
comparator.generate_comparison_table()
comparator.plot_comparison()
```

## Output
- GPU status dan configuration
- Comparison tables (CSV)
- Performance plots (PNG)
- HTML reports
