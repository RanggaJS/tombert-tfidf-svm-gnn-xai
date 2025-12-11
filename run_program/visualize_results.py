#!/usr/bin/env python3
# coding=utf-8
"""
Skrip untuk memvisualisasikan hasil dari ketiga metode:
- TomBERT (Multimodal BERT)
- TF-IDF + SVM (Classical Method)
- GNN (Graph Neural Network untuk Rumor Detection)

Output: Berbagai visualisasi perbandingan metrik, performa, dan analisis
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)

# Colors untuk setiap metode
COLORS = {
    'TomBERT': '#2E86AB',      # Blue
    'TF-IDF + SVM': '#A23B72', # Purple
    'GNN': '#F18F01'           # Orange
}


def load_tombert_results(results_path: Path) -> Dict:
    """Load hasil TomBERT dari JSON"""
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
        return {
            'method': 'TomBERT',
            'test_accuracy': data.get('final_test_accuracy', 0),
            'test_f1': data.get('final_test_f1', 0),
            'test_precision': data.get('final_test_precision', 0),
            'test_recall': data.get('final_test_recall', 0),
            'training_time_hours': data.get('training_time_hours', 0),
            'epochs': data.get('epochs_completed', 0),
            'dev_accuracy': data.get('best_dev_accuracy', 0),
            'dev_f1': data.get('best_dev_f1', 0)
        }
    except Exception as e:
        logger.error(f"Error loading TomBERT results: {e}")
        return None


def load_tfidf_results(results_path: Path) -> Dict:
    """Load hasil TF-IDF + SVM dari JSON"""
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
        return {
            'method': 'TF-IDF + SVM',
            'test_accuracy': data.get('test_accuracy', 0),
            'dev_accuracy': data.get('dev_accuracy', 0),
            'training_time_hours': data.get('total_training_time', 0) / 3600,
            'status': data.get('status', 'unknown')
        }
    except Exception as e:
        logger.error(f"Error loading TF-IDF results: {e}")
        return None


def load_gnn_results(results_path: Path) -> Dict:
    """Load hasil GNN dari JSON"""
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
        return {
            'method': 'GNN',
            'test_accuracy': data.get('test_accuracy', 0),
            'test_f1_macro': data.get('test_f1_macro', 0),
            'test_f1_rumor': data.get('test_f1_rumor', 0),
            'test_precision_rumor': data.get('test_precision_rumor', 0),
            'test_recall_rumor': data.get('test_recall_rumor', 0),
            'training_time_seconds': data.get('training_time', 0),
            'training_time_hours': data.get('training_time', 0) / 3600,
            'optimal_threshold': data.get('optimal_threshold', 0.5),
            'dataset': data.get('dataset', 'unknown')
        }
    except Exception as e:
        logger.error(f"Error loading GNN results: {e}")
        return None


def create_comparison_bar_chart(results: List[Dict], output_dir: Path):
    """Membuat bar chart perbandingan metrik utama"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Perbandingan Hasil Eksperimen: TomBERT vs TF-IDF+SVM vs GNN', 
                 fontsize=16, fontweight='bold')
    
    methods = [r['method'] for r in results if r]
    colors_list = [COLORS.get(m, '#808080') for m in methods]
    
    # 1. Test Accuracy
    ax1 = axes[0, 0]
    accuracies = [r.get('test_accuracy', 0) * 100 for r in results if r]
    bars1 = ax1.bar(methods, accuracies, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. F1-Score (jika tersedia)
    ax2 = axes[0, 1]
    f1_scores = []
    f1_labels = []
    for r in results:
        if r:
            if 'test_f1' in r:
                f1_scores.append(r['test_f1'] * 100)
                f1_labels.append(r['method'])
            elif 'test_f1_macro' in r:
                f1_scores.append(r['test_f1_macro'] * 100)
                f1_labels.append(r['method'])
    
    if f1_scores:
        colors_f1 = [COLORS.get(m, '#808080') for m in f1_labels]
        bars2 = ax2.bar(f1_labels, f1_scores, color=colors_f1, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
        ax2.set_title('F1-Score (Macro)', fontsize=13, fontweight='bold')
        ax2.set_ylim([0, 100])
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        for bar, f1 in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{f1:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Training Time
    ax3 = axes[1, 0]
    times = []
    time_labels = []
    for r in results:
        if r and 'training_time_hours' in r:
            times.append(r['training_time_hours'])
            time_labels.append(r['method'])
    
    if times:
        colors_time = [COLORS.get(m, '#808080') for m in time_labels]
        bars3 = ax3.bar(time_labels, times, color=colors_time, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Waktu Training (jam)', fontsize=12, fontweight='bold')
        ax3.set_title('Training Time', fontsize=13, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        # Format time labels
        for bar, time in zip(bars3, times):
            height = bar.get_height()
            if time < 0.1:
                label = f'{time*60:.1f} menit'
            elif time < 1:
                label = f'{time*60:.1f} menit'
            else:
                label = f'{time:.2f} jam'
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 4. Precision vs Recall (jika tersedia)
    ax4 = axes[1, 1]
    precisions = []
    recalls = []
    pr_labels = []
    for r in results:
        if r:
            if 'test_precision' in r and 'test_recall' in r:
                precisions.append(r['test_precision'] * 100)
                recalls.append(r['test_recall'] * 100)
                pr_labels.append(r['method'])
    
    if precisions and recalls:
        x = np.arange(len(pr_labels))
        width = 0.35
        colors_pr = [COLORS.get(m, '#808080') for m in pr_labels]
        bars4a = ax4.bar(x - width/2, precisions, width, label='Precision', 
                        color=colors_pr, alpha=0.8, edgecolor='black', linewidth=1.5)
        bars4b = ax4.bar(x + width/2, recalls, width, label='Recall',
                        color=[c for c in colors_pr], alpha=0.6, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Precision vs Recall', fontsize=13, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(pr_labels)
        ax4.legend()
        ax4.set_ylim([0, 100])
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        # Add value labels
        for bars in [bars4a, bars4b]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'comparison_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison chart: {output_path}")
    plt.close()


def create_radar_chart(results: List[Dict], output_dir: Path):
    """Membuat radar chart untuk perbandingan multi-dimensi"""
    # Prepare data
    metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Efficiency']
    methods_data = {}
    
    for r in results:
        if not r:
            continue
        method = r['method']
        values = []
        
        # Normalize values to 0-100 scale
        values.append(r.get('test_accuracy', 0) * 100)
        
        if 'test_f1' in r:
            values.append(r['test_f1'] * 100)
        elif 'test_f1_macro' in r:
            values.append(r['test_f1_macro'] * 100)
        else:
            values.append(0)
        
        if 'test_precision' in r:
            values.append(r['test_precision'] * 100)
        else:
            values.append(0)
        
        if 'test_recall' in r:
            values.append(r['test_recall'] * 100)
        else:
            values.append(0)
        
        # Efficiency: inverse of training time (normalized)
        if 'training_time_hours' in r and r['training_time_hours'] > 0:
            # Normalize: faster = higher score (max time = 10 hours = 100 score)
            efficiency = min(100, (10 / r['training_time_hours']) * 10)
        else:
            efficiency = 0
        values.append(efficiency)
        
        methods_data[method] = values
    
    if not methods_data:
        logger.warning("No data for radar chart")
        return
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for method, values in methods_data.items():
        values += values[:1]  # Complete the circle
        color = COLORS.get(method, '#808080')
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title('Perbandingan Multi-Dimensi Metode', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'radar_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved radar chart: {output_path}")
    plt.close()


def create_training_time_comparison(results: List[Dict], output_dir: Path):
    """Membuat visualisasi perbandingan waktu training"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    times = []
    colors_list = []
    
    for r in results:
        if r and 'training_time_hours' in r:
            methods.append(r['method'])
            times.append(r['training_time_hours'])
            colors_list.append(COLORS.get(r['method'], '#808080'))
    
    if not times:
        logger.warning("No training time data available")
        return
    
    bars = ax.barh(methods, times, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Waktu Training (jam)', fontsize=12, fontweight='bold')
    ax.set_title('Perbandingan Waktu Training', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, time in zip(bars, times):
        width = bar.get_width()
        if time < 0.1:
            label = f'{time*60:.1f} menit'
        elif time < 1:
            label = f'{time*60:.0f} menit'
        else:
            label = f'{time:.2f} jam'
        ax.text(width + width*0.05, bar.get_y() + bar.get_height()/2,
                label, ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'training_time_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved training time comparison: {output_path}")
    plt.close()


def create_summary_table(results: List[Dict], output_dir: Path):
    """Membuat tabel ringkasan hasil"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Metode', 'Test Accuracy', 'F1-Score', 'Precision', 'Recall', 'Training Time']
    
    for r in results:
        if not r:
            continue
        row = [r['method']]
        
        # Accuracy
        acc = r.get('test_accuracy', 0)
        row.append(f'{acc*100:.2f}%' if acc > 0 else 'N/A')
        
        # F1-Score
        if 'test_f1' in r:
            f1 = r['test_f1']
            row.append(f'{f1*100:.2f}%')
        elif 'test_f1_macro' in r:
            f1 = r['test_f1_macro']
            row.append(f'{f1*100:.2f}%')
        else:
            row.append('N/A')
        
        # Precision
        if 'test_precision' in r:
            prec = r['test_precision']
            row.append(f'{prec*100:.2f}%')
        else:
            row.append('N/A')
        
        # Recall
        if 'test_recall' in r:
            rec = r['test_recall']
            row.append(f'{rec*100:.2f}%')
        else:
            row.append('N/A')
        
        # Training Time
        if 'training_time_hours' in r:
            time = r['training_time_hours']
            if time < 0.1:
                row.append(f'{time*60:.1f} menit')
            elif time < 1:
                row.append(f'{time*60:.0f} menit')
            else:
                row.append(f'{time:.2f} jam')
        else:
            row.append('N/A')
        
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i, row in enumerate(table_data, 1):
        method = row[0]
        color = COLORS.get(method, '#FFFFFF')
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color if j == 0 else '#F5F5F5')
            table[(i, j)].set_alpha(0.7 if j == 0 else 1.0)
    
    ax.set_title('Ringkasan Hasil Eksperimen', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / 'summary_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved summary table: {output_path}")
    plt.close()


def create_gnn_detailed_metrics(gnn_result: Dict, output_dir: Path):
    """Membuat visualisasi detail metrik GNN untuk rumor detection"""
    if not gnn_result or 'test_f1_rumor' not in gnn_result:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Detail Metrik GNN untuk Rumor Detection', fontsize=14, fontweight='bold')
    
    # 1. Precision, Recall, F1 untuk kelas Rumor
    ax1 = axes[0]
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [
        gnn_result.get('test_precision_rumor', 0) * 100,
        gnn_result.get('test_recall_rumor', 0) * 100,
        gnn_result.get('test_f1_rumor', 0) * 100
    ]
    colors_metrics = ['#E74C3C', '#3498DB', '#2ECC71']
    bars = ax1.bar(metrics, values, color=colors_metrics, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Metrik Kelas Rumor', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Optimal Threshold
    ax2 = axes[1]
    threshold = gnn_result.get('optimal_threshold', 0.5)
    ax2.barh(['Optimal Threshold'], [threshold], color='#F18F01', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Threshold Value', fontsize=11, fontweight='bold')
    ax2.set_title('Threshold Optimal untuk Rumor Detection', fontsize=12, fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.text(threshold, 0, f'{threshold:.3f}', ha='center', va='center', 
            fontweight='bold', fontsize=12, color='white')
    
    plt.tight_layout()
    output_path = output_dir / 'gnn_detailed_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved GNN detailed metrics: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualisasi hasil eksperimen dari ketiga metode')
    parser.add_argument('--tombert', type=str, 
                       default='output/tombert_ultra_optimized_20251129_225613/ultra_results.json',
                       help='Path ke file hasil TomBERT')
    parser.add_argument('--tfidf', type=str,
                       default='output/tfidf_svm_ultra_optimized_20251209_111442/ultra_results.json',
                       help='Path ke file hasil TF-IDF+SVM')
    parser.add_argument('--gnn', type=str,
                       default='output/gnn_optimized_twitter2015_rumor/results.json',
                       help='Path ke file hasil GNN')
    parser.add_argument('--output', type=str, default='output/visualizations',
                       help='Output directory untuk visualisasi')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading results from all methods...")
    
    # Load results
    tombert_result = load_tombert_results(Path(args.tombert)) if Path(args.tombert).exists() else None
    tfidf_result = load_tfidf_results(Path(args.tfidf)) if Path(args.tfidf).exists() else None
    gnn_result = load_gnn_results(Path(args.gnn)) if Path(args.gnn).exists() else None
    
    results = [r for r in [tombert_result, tfidf_result, gnn_result] if r is not None]
    
    if not results:
        logger.error("No results found! Please check file paths.")
        return
    
    logger.info(f"Loaded {len(results)} method results")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    create_comparison_bar_chart(results, output_dir)
    create_radar_chart(results, output_dir)
    create_training_time_comparison(results, output_dir)
    create_summary_table(results, output_dir)
    
    # GNN specific visualizations
    if gnn_result:
        create_gnn_detailed_metrics(gnn_result, output_dir)
    
    logger.info(f"All visualizations saved to: {output_dir}")
    logger.info("Generated files:")
    logger.info("  - comparison_metrics.png: Perbandingan metrik utama")
    logger.info("  - radar_chart.png: Radar chart multi-dimensi")
    logger.info("  - training_time_comparison.png: Perbandingan waktu training")
    logger.info("  - summary_table.png: Tabel ringkasan")
    if gnn_result:
        logger.info("  - gnn_detailed_metrics.png: Detail metrik GNN")


if __name__ == '__main__':
    main()

