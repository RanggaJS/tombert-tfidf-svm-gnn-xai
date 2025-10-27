# coding=utf-8
"""
Konfigurasi GPU untuk semua metode dalam eksperimen skripsi
"""

import torch
import os
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUConfig:
    """
    Konfigurasi GPU untuk semua metode
    """
    
    def __init__(self, gpu_id: int = 0, use_mixed_precision: bool = True):
        self.gpu_id = gpu_id
        self.use_mixed_precision = use_mixed_precision
        self.device = self._setup_device()
        self.scaler = None
        
    def _setup_device(self):
        """Setup device (GPU/CPU)"""
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{self.gpu_id}')
            torch.cuda.set_device(self.gpu_id)
            logger.info(f"Using GPU: {torch.cuda.get_device_name(self.gpu_id)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(self.gpu_id).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device('cpu')
            logger.warning("CUDA not available, using CPU")
        
        return device
    
    def get_device(self):
        """Get current device"""
        return self.device
    
    def setup_mixed_precision(self):
        """Setup mixed precision training"""
        if self.use_mixed_precision and torch.cuda.is_available():
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            logger.info("Mixed precision training disabled")
    
    def get_scaler(self):
        """Get gradient scaler for mixed precision"""
        return self.scaler
    
    def clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def get_memory_info(self):
        """Get GPU memory information"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.gpu_id) / 1e9
            cached = torch.cuda.memory_reserved(self.gpu_id) / 1e9
            total = torch.cuda.get_device_properties(self.gpu_id).total_memory / 1e9
            
            return {
                'allocated': allocated,
                'cached': cached,
                'total': total,
                'free': total - allocated
            }
        else:
            return None
    
    def print_memory_info(self):
        """Print GPU memory information"""
        memory_info = self.get_memory_info()
        if memory_info:
            logger.info(f"GPU Memory - Allocated: {memory_info['allocated']:.2f}GB, "
                       f"Cached: {memory_info['cached']:.2f}GB, "
                       f"Free: {memory_info['free']:.2f}GB")
        else:
            logger.info("GPU memory info not available")


def setup_gpu_environment(gpu_id: int = 0, use_mixed_precision: bool = True):
    """
    Setup environment untuk GPU
    """
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Setup GPU config
    gpu_config = GPUConfig(gpu_id=gpu_id, use_mixed_precision=use_mixed_precision)
    
    # Setup mixed precision
    gpu_config.setup_mixed_precision()
    
    return gpu_config


def get_optimal_batch_size(model, input_shape, device, max_batch_size=32):
    """
    Tentukan batch size optimal berdasarkan GPU memory
    """
    if not torch.cuda.is_available():
        return 16  # Default untuk CPU
    
    # Test batch sizes
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        try:
            # Test forward pass
            test_input = torch.randn(batch_size, *input_shape).to(device)
            with torch.no_grad():
                _ = model(test_input)
            
            # Clear cache
            torch.cuda.empty_cache()
            
            logger.info(f"Batch size {batch_size} works")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"Batch size {batch_size} failed: {e}")
                torch.cuda.empty_cache()
                return max(1, batch_size // 2)
            else:
                raise e
    
    return min(max_batch_size, 64)


def optimize_model_for_gpu(model, device):
    """
    Optimasi model untuk GPU
    """
    # Move model to device
    model = model.to(device)
    
    # Enable cuDNN benchmark untuk performa yang lebih baik
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Compile model jika menggunakan PyTorch 2.0+
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                logger.info("Model compiled for better performance")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
    
    return model


def create_gpu_optimized_dataloader(dataset, batch_size, num_workers=4, pin_memory=True):
    """
    Buat DataLoader yang dioptimasi untuk GPU
    """
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )


def save_gpu_checkpoint(model, optimizer, epoch, loss, filepath, gpu_config):
    """
    Simpan checkpoint dengan informasi GPU
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'gpu_config': {
            'gpu_id': gpu_config.gpu_id,
            'use_mixed_precision': gpu_config.use_mixed_precision
        }
    }
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_gpu_checkpoint(filepath, model, optimizer=None, gpu_config=None):
    """
    Load checkpoint dengan informasi GPU
    """
    checkpoint = torch.load(filepath, map_location=gpu_config.get_device() if gpu_config else 'cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if gpu_config and 'gpu_config' in checkpoint:
        logger.info(f"Loaded checkpoint from GPU {checkpoint['gpu_config']['gpu_id']}")
    
    return checkpoint['epoch'], checkpoint['loss']


if __name__ == "__main__":
    # Test GPU configuration
    gpu_config = setup_gpu_environment(gpu_id=0, use_mixed_precision=True)
    
    print(f"Device: {gpu_config.get_device()}")
    gpu_config.print_memory_info()
    
    # Test memory allocation
    if torch.cuda.is_available():
        test_tensor = torch.randn(1000, 1000).to(gpu_config.get_device())
        gpu_config.print_memory_info()
        
        del test_tensor
        gpu_config.clear_cache()
        gpu_config.print_memory_info()
