# coding=utf-8
"""
Implementasi XAI (Explainable AI) methods untuk interpretabilitas model - HIGHLY OPTIMIZED VERSION
Enhanced dengan caching, multiprocessing, memory optimization, dan performa tinggi
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from wordcloud import WordCloud
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import lime
import lime.lime_tabular
import lime.lime_text
import shap
from transformers import BertTokenizer, BertModel
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
import pickle
from pathlib import Path
import multiprocessing as mp
from dataclasses import dataclass, asdict
import time
from contextlib import contextmanager

warnings.filterwarnings('ignore')

# Setup logging dengan optimasi
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xai_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style untuk visualisasi
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration untuk optimasi
@dataclass
class XAIConfig:
    """Configuration class untuk XAI Analyzer"""
    max_workers: int = min(32, (os.cpu_count() or 1) + 4)
    cache_size: int = 128
    batch_size: int = 32
    max_memory_gb: float = 8.0
    enable_gpu: bool = torch.cuda.is_available()
    plot_dpi: int = 150  # Reduced from 300 for performance
    max_tokens: int = 512
    enable_caching: bool = True
    cache_dir: str = './cache'

# Memory management utilities
@contextmanager
def memory_limit(max_memory_gb: float):
    """Context manager untuk memory monitoring"""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024 / 1024
    
    try:
        yield
    finally:
        current_memory = process.memory_info().rss / 1024 / 1024 / 1024
        if current_memory > max_memory_gb:
            logger.warning(f"Memory usage: {current_memory:.2f}GB exceeds limit: {max_memory_gb}GB")
            gc.collect()

def cache_result(cache_dir: str = './cache'):
    """Decorator untuk caching hasil komputasi"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            
            # Create cache key
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            # Load from cache if exists
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except:
                    pass
            
            # Compute and cache result
            result = func(*args, **kwargs)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except:
                pass
            
            return result
        return wrapper
    return decorator

class OptimizedXAIAnalyzer:
    """
    Highly optimized XAI Analyzer dengan caching, multiprocessing, dan memory optimization
    """
    
    def __init__(
        self, 
        model, 
        tokenizer=None, 
        device='auto',
        output_dir='./results/xai_results',
        config: Optional[XAIConfig] = None
    ):
        """Initialize optimized XAI Analyzer"""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or XAIConfig()
        self.output_dir = Path(output_dir)
        self.explanations = {}
        self.results = {}
        
        # Device optimization
        if device == 'auto':
            self.device = 'cuda' if self.config.enable_gpu else 'cpu'
        else:
        self.device = device
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        
        if self.config.enable_caching:
            Path(self.config.cache_dir).mkdir(exist_ok=True)
        
        # Memory pool untuk batch processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        logger.info(f"Optimized XAI Analyzer initialized. Device: {self.device}, Workers: {self.config.max_workers}")

    @lru_cache(maxsize=128)
    def _preprocess_tokens(self, tokens_tuple: Tuple[str, ...]) -> List[str]:
        """Cached token preprocessing"""
        tokens = list(tokens_tuple)
        # Clean special tokens
        return [token.replace('##', '') for token in tokens if not token.startswith('[')]

    @cache_result()
    def analyze_attention_weights_optimized(
        self, 
        input_ids: torch.Tensor, 
        attention_weights: List[torch.Tensor],
        tokens: Optional[List[str]] = None,
        layer_idx: int = -1,
        head_idx: Optional[int] = None,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Optimized attention analysis dengan caching dan batch processing
        """
        logger.info("Analyzing attention weights (optimized)...")
        
        with memory_limit(self.config.max_memory_gb):
            # Move to device efficiently
            if isinstance(attention_weights, (list, tuple)):
                attention_weights = [w.to(self.device) if hasattr(w, 'to') else w 
                                   for w in attention_weights]
            
            # Extract attention efficiently
            layer_attention = attention_weights[layer_idx]
            
            # Optimize tensor operations
            with torch.no_grad():
                if len(layer_attention.shape) == 4:
                    batch_attention = layer_attention[0]
            else:
                    batch_attention = layer_attention
                
                # Use head or average
                if head_idx is not None and head_idx < batch_attention.shape[0]:
                    attention_matrix = batch_attention[head_idx].cpu().numpy()
                    analysis_type = f"Layer {layer_idx}, Head {head_idx}"
        else:
                    attention_matrix = batch_attention.mean(dim=0).cpu().numpy()
                    analysis_type = f"Layer {layer_idx}, Average"
            
            # Process tokens efficiently
            if tokens is None and self.tokenizer is not None:
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
            elif tokens is None:
                tokens = [f"token_{i}" for i in range(attention_matrix.shape[0])]
            
            # Truncate to max_tokens for performance
            max_len = min(len(tokens), attention_matrix.shape[0], self.config.max_tokens)
            tokens = tokens[:max_len]
            attention_matrix = attention_matrix[:max_len, :max_len]
            
            # Parallel plotting
            if save_plots:
                plot_tasks = [
                    self.executor.submit(self._plot_attention_heatmap_optimized, 
                                       attention_matrix, tokens, analysis_type),
                    self.executor.submit(self._plot_attention_flow_optimized, 
                                       attention_matrix, tokens, analysis_type),
                    self.executor.submit(self._plot_token_importance_optimized, 
                                       attention_matrix, tokens, analysis_type)
                ]
                
                # Wait for completion
                for task in plot_tasks:
                    task.result()
            
            # Compute statistics efficiently
            stats = self._compute_attention_statistics_optimized(attention_matrix, tokens)
            
            result = {
                'attention_matrix': attention_matrix.tolist(),  # Convert to list for JSON serialization
            'tokens': tokens,
            'layer': layer_idx,
                'head': head_idx,
                'analysis_type': analysis_type,
                'statistics': stats
            }
            
            self.explanations['attention'] = result
            
            # Clean up GPU memory
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            logger.info(f"Optimized attention analysis completed for {analysis_type}")
            return result

    def _plot_attention_heatmap_optimized(
        self, 
        attention_matrix: np.ndarray, 
        tokens: List[str],
        title: str
    ):
        """Optimized attention heatmap plotting"""
        try:
            # Use smaller figure for performance
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Optimize heatmap rendering
            im = ax.imshow(attention_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
            
            # Limit number of ticks for performance
            max_ticks = 20
            if len(tokens) > max_ticks:
                step = len(tokens) // max_ticks
                tick_indices = range(0, len(tokens), step)
                ax.set_xticks(tick_indices)
                ax.set_yticks(tick_indices)
                ax.set_xticklabels([tokens[i] for i in tick_indices], rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels([tokens[i] for i in tick_indices], fontsize=8)
            else:
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(tokens, fontsize=8)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Attention Weight', rotation=270, labelpad=15)
            
            ax.set_title(f'Attention Heatmap - {title}', fontsize=12, pad=15)
            ax.set_xlabel('Key Tokens', fontsize=10)
            ax.set_ylabel('Query Tokens', fontsize=10)
            
        plt.tight_layout()
            
            # Save with optimized settings
            filename = self.output_dir / 'visualizations' / f'attention_heatmap_{title.replace(" ", "_").replace(",", "")}.png'
            plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to plot attention heatmap: {e}")
            plt.close()

    def _plot_attention_flow_optimized(
        self, 
        attention_matrix: np.ndarray, 
        tokens: List[str],
        title: str
    ):
        """Optimized attention flow plotting"""
        try:
            # Take attention for CLS token or first token
            cls_attention = attention_matrix[0, :]
            
            # Get top N tokens for performance
            top_n = min(15, len(tokens))
            sorted_indices = np.argsort(cls_attention)[::-1][:top_n]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create bar plot
            y_pos = np.arange(len(sorted_indices))
            values = [cls_attention[i] for i in sorted_indices]
            
            bars = ax.barh(y_pos, values)
            
            # Color gradient
            colors = plt.cm.YlOrRd(np.linspace(0.3, 1.0, len(values)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([tokens[i] for i in sorted_indices])
            ax.set_title(f'Top Attended Tokens - {title}', fontsize=12)
            ax.set_xlabel('Attention Weight', fontsize=10)
            
            plt.tight_layout()
            
            filename = self.output_dir / 'visualizations' / f'attention_flow_{title.replace(" ", "_").replace(",", "")}.png'
            plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to plot attention flow: {e}")
            plt.close()

    def _plot_token_importance_optimized(
        self, 
        attention_matrix: np.ndarray, 
        tokens: List[str],
        title: str
    ):
        """Optimized token importance plotting"""
        try:
            # Compute token importance efficiently
            token_importance = np.sum(attention_matrix, axis=0)
            
            # Get top tokens
            top_n = min(15, len(tokens))
            sorted_indices = np.argsort(token_importance)[::-1][:top_n]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            y_pos = np.arange(len(sorted_indices))
            values = [token_importance[i] for i in sorted_indices]
            
            bars = ax.barh(y_pos, values)
            
            # Color bars
            colors = plt.cm.viridis(np.linspace(0.3, 1.0, len(values)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([tokens[i] for i in sorted_indices])
            ax.set_title(f'Token Importance - {title}', fontsize=12)
            ax.set_xlabel('Importance Score', fontsize=10)
            
            plt.tight_layout()
            
            filename = self.output_dir / 'visualizations' / f'token_importance_{title.replace(" ", "_").replace(",", "")}.png'
            plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to plot token importance: {e}")
            plt.close()

    def _compute_attention_statistics_optimized(
        self, 
        attention_matrix: np.ndarray, 
        tokens: List[str]
    ) -> Dict[str, Any]:
        """Optimized attention statistics computation"""
        # Use numpy functions for efficiency
        stats = {
            'max_attention': float(np.max(attention_matrix)),
            'min_attention': float(np.min(attention_matrix)),
            'mean_attention': float(np.mean(attention_matrix)),
            'std_attention': float(np.std(attention_matrix)),
            'attention_entropy': float(-np.sum(attention_matrix * np.log(attention_matrix + 1e-8))),
            'attention_sparsity': float(np.mean(attention_matrix < 0.01)),
            'top_attended_tokens': []
        }
        
        # Efficient top token computation
        token_importance = np.sum(attention_matrix, axis=0)
        top_indices = np.argpartition(token_importance, -5)[-5:]
        top_indices = top_indices[np.argsort(token_importance[top_indices])[::-1]]
        
        for idx in top_indices:
            stats['top_attended_tokens'].append({
                'token': tokens[idx],
                'importance': float(token_importance[idx])
            })
        
        return stats

    def analyze_model_performance_optimized(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Optimized model performance analysis
        """
        logger.info("Analyzing model performance (optimized)...")
        
        with memory_limit(self.config.max_memory_gb):
            # Vectorized computations
            accuracy = np.mean(y_true == y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            # Efficient classification report
            report = classification_report(
                y_true, y_pred, 
                target_names=class_names,
                zero_division=0,
                output_dict=True
            )
            
            # Optimized confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # ROC analysis with optimization
            roc_data = {}
            if y_prob is not None:
                roc_data = self._compute_roc_metrics_optimized(y_true, y_prob, class_names)
            
            # Parallel plotting
            if save_plots:
                plot_tasks = [
                    self.executor.submit(self._plot_confusion_matrix_optimized, cm, class_names),
                    self.executor.submit(self._plot_classification_report_optimized, report, class_names)
                ]
                
                if roc_data:
                    plot_tasks.append(
                        self.executor.submit(self._plot_roc_curves_optimized, roc_data)
                    )
                
                # Wait for completion
                for task in plot_tasks:
                    task.result()
            
            results = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'roc_data': roc_data
            }
            
            self.results['performance'] = results
            
            logger.info("Optimized model performance analysis completed")
            return results

    def _compute_roc_metrics_optimized(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        class_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Optimized ROC metrics computation"""
        roc_data = {}
        
        try:
            if len(y_prob.shape) == 1 or y_prob.shape[1] == 2:
                # Binary classification
                if len(y_prob.shape) == 2:
                    y_prob_binary = y_prob[:, 1]
                else:
                    y_prob_binary = y_prob
                
                fpr, tpr, _ = roc_curve(y_true, y_prob_binary)
                auc_score = auc(fpr, tpr)
                roc_data['binary'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': float(auc_score)
                }
            else:
                # Multiclass - compute only for top classes to save time
                n_classes = min(y_prob.shape[1], 5)  # Limit to top 5 classes
                
                for i in range(n_classes):
                    class_name = class_names[i] if class_names and i < len(class_names) else f'class_{i}'
                    
                    fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
                    auc_score = auc(fpr, tpr)
                    
                    roc_data[f'class_{i}'] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'auc': float(auc_score),
                        'class_name': class_name
                    }
        except Exception as e:
            logger.error(f"ROC computation failed: {e}")
        
        return roc_data

    def _plot_confusion_matrix_optimized(self, cm: np.ndarray, class_names: Optional[List[str]]):
        """Optimized confusion matrix plotting"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Use optimized heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
                xticklabels=class_names or range(cm.shape[1]),
                yticklabels=class_names or range(cm.shape[0]),
                ax=ax,
                cbar_kws={'shrink': 0.8}
            )
            
            ax.set_title('Confusion Matrix', fontsize=12)
            ax.set_xlabel('Predicted Label', fontsize=10)
            ax.set_ylabel('True Label', fontsize=10)
            
        plt.tight_layout()
            
            filename = self.output_dir / 'visualizations' / 'confusion_matrix.png'
            plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}")
            plt.close()

    def _plot_classification_report_optimized(self, report: Dict, class_names: Optional[List[str]]):
        """Optimized classification report plotting"""
        try:
            # Extract metrics efficiently
            classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            
            if not classes:
                return
            
            metrics_data = {
                'precision': [report[cls]['precision'] for cls in classes],
                'recall': [report[cls]['recall'] for cls in classes],
                'f1-score': [report[cls]['f1-score'] for cls in classes]
            }
            
            # Create DataFrame for easier plotting
            df = pd.DataFrame(metrics_data, index=class_names or classes)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            df.plot(kind='bar', ax=ax, alpha=0.8)
            
            ax.set_title('Classification Metrics', fontsize=12)
            ax.set_xlabel('Classes', fontsize=10)
            ax.set_ylabel('Score', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.xticks(rotation=45)
        plt.tight_layout()
            
            filename = self.output_dir / 'visualizations' / 'classification_report.png'
            plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to plot classification report: {e}")
            plt.close()

    def _plot_roc_curves_optimized(self, roc_data: Dict[str, Any]):
        """Optimized ROC curves plotting"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for key, data in roc_data.items():
                if isinstance(data, dict) and 'fpr' in data and 'tpr' in data:
                    label = f"{data.get('class_name', key)} (AUC = {data['auc']:.3f})"
                    ax.plot(data['fpr'], data['tpr'], label=label, linewidth=2)
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
            ax.set_xlabel('False Positive Rate', fontsize=10)
            ax.set_ylabel('True Positive Rate', fontsize=10)
            ax.set_title('ROC Curves', fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
            
            filename = self.output_dir / 'visualizations' / 'roc_curves.png'
            plt.savefig(filename, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to plot ROC curves: {e}")
            plt.close()

    def batch_analyze(
        self, 
        samples: List[Dict[str, Any]], 
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Batch analysis untuk multiple samples
        """
        batch_size = batch_size or self.config.batch_size
        logger.info(f"Starting batch analysis for {len(samples)} samples with batch size {batch_size}")
        
        results = {'batch_results': [], 'summary': {}}
        
        # Process in batches
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size}")
            
            batch_result = self._process_batch(batch)
            results['batch_results'].append(batch_result)
            
            # Memory cleanup
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Generate summary
        results['summary'] = self._generate_batch_summary(results['batch_results'])
        
        logger.info("Batch analysis completed")
        return results

    def _process_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process single batch"""
        batch_results = []
        
        for sample in batch:
            try:
                sample_result = {}
                
                # Process attention if available
                if 'attention_weights' in sample:
                    sample_result['attention'] = self.analyze_attention_weights_optimized(
                        sample.get('input_ids'),
                        sample.get('attention_weights'),
                        tokens=sample.get('tokens'),
                        save_plots=False  # Don't save plots for each sample in batch
                    )
                
                batch_results.append(sample_result)
                
            except Exception as e:
                logger.error(f"Failed to process sample: {e}")
                batch_results.append({'error': str(e)})
        
        return {'results': batch_results, 'batch_size': len(batch)}

    def _generate_batch_summary(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary from batch results"""
        total_samples = sum(br.get('batch_size', 0) for br in batch_results)
        successful_samples = sum(
            len([r for r in br.get('results', []) if 'error' not in r]) 
            for br in batch_results
        )
        
        return {
            'total_samples': total_samples,
            'successful_samples': successful_samples,
            'success_rate': successful_samples / total_samples if total_samples > 0 else 0,
            'total_batches': len(batch_results)
        }

    def generate_optimized_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimized report"""
        logger.info("Generating optimized comprehensive report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': asdict(self.config),
            'system_info': {
                'cpu_count': os.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'gpu_available': torch.cuda.is_available(),
                'device_used': self.device
            },
            'analysis_summary': {
                'explanations_generated': len(self.explanations),
                'explanation_types': list(self.explanations.keys()),
                'results_generated': len(self.results)
            },
            'explanations': self.explanations,
            'results': self.results
        }
        
        # Save report
        report_file = self.output_dir / 'reports' / 'optimized_comprehensive_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Optimized report saved to {report_file}")
        return report

    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        logger.info("Resources cleaned up")

    def __del__(self):
        """Destructor"""
        try:
            self.cleanup()
        except:
            pass

# Convenience functions
def create_optimized_analyzer(
    model, 
    tokenizer=None, 
    device='auto',
    output_dir='./results/xai_results',
    **config_kwargs
) -> OptimizedXAIAnalyzer:
    """Create optimized XAI analyzer with custom configuration"""
    config = XAIConfig(**config_kwargs)
    return OptimizedXAIAnalyzer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
        config=config
    )

def quick_analysis(
    model,
    sample_data: List[Dict[str, Any]],
    output_dir: str = './results/quick_xai',
    max_samples: int = 10
) -> Dict[str, Any]:
    """Quick analysis untuk testing"""
    logger.info(f"Starting quick analysis for {min(len(sample_data), max_samples)} samples")
    
    config = XAIConfig(
        max_workers=4,
        plot_dpi=100,
        max_tokens=256,
        enable_caching=False
    )
    
    analyzer = OptimizedXAIAnalyzer(
        model=model,
        output_dir=output_dir,
        config=config
    )
    
    # Limit samples for quick analysis
    limited_samples = sample_data[:max_samples]
    
    try:
        # Batch analysis
        results = analyzer.batch_analyze(limited_samples, batch_size=5)
        
        # Generate report
        report = analyzer.generate_optimized_report()
        
        return {
            'analysis_results': results,
            'report': report,
            'analyzer': analyzer
        }
        
    finally:
        analyzer.cleanup()

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator untuk monitoring performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            logger.info(f"{func.__name__} completed in {end_time - start_time:.2f}s, "
                       f"memory change: {end_memory - start_memory:.2f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            raise
    
    return wrapper

if __name__ == "__main__":
    # Example usage
    logger.info("XAI Optimization module loaded successfully")
    
    # Print optimization info
    config = XAIConfig()
    logger.info(f"Optimization settings: Workers={config.max_workers}, "
               f"Cache={config.enable_caching}, GPU={config.enable_gpu}")