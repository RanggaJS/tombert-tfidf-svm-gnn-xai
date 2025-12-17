#!/usr/bin/env python3
"""
Script untuk membuat diagram alir (flowchart) untuk:
1. TomBERT
2. TF-IDF+SVM
3. GNN
4. XAI

Output: 4 file PNG untuk BAB 4 Perancangan
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np
from pathlib import Path

# Setup output directory
OUTPUT_DIR = Path("output/diagrams")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    'input': '#E3F2FD',      # Light blue
    'process': '#FFF3E0',   # Light orange
    'model': '#F3E5F5',      # Light purple
    'output': '#E8F5E9',     # Light green
    'arrow': '#424242',      # Dark gray
    'text': '#212121'        # Almost black
}

def create_tombert_diagram():
    """Membuat diagram alir TomBERT"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Diagram Alir TomBERT (Target-Oriented Multimodal BERT)', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Input layer
    input_box = FancyBboxPatch((3, 9.5), 4, 0.8, boxstyle="round,pad=0.1", 
                               facecolor=COLORS['input'], edgecolor='black', linewidth=1.5)
    ax.add_patch(input_box)
    ax.text(5, 9.9, 'Input: Teks + Gambar + Entity', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Text processing branch
    text_box1 = FancyBboxPatch((1, 8), 2.5, 0.6, boxstyle="round,pad=0.1", 
                               facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(text_box1)
    ax.text(2.25, 8.3, 'Tokenisasi Teks', ha='center', va='center', fontsize=10)
    
    text_box2 = FancyBboxPatch((1, 7), 2.5, 0.6, boxstyle="round,pad=0.1", 
                               facecolor=COLORS['model'], edgecolor='black', linewidth=1.5)
    ax.add_patch(text_box2)
    ax.text(2.25, 7.3, 'BERT Encoder\n(H^s)', ha='center', va='center', fontsize=10)
    
    # Entity processing branch
    entity_box1 = FancyBboxPatch((3.75, 8), 2.5, 0.6, boxstyle="round,pad=0.1", 
                                 facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(entity_box1)
    ax.text(5, 8.3, 'Tokenisasi Entity', ha='center', va='center', fontsize=10)
    
    entity_box2 = FancyBboxPatch((3.75, 7), 2.5, 0.6, boxstyle="round,pad=0.1", 
                                 facecolor=COLORS['model'], edgecolor='black', linewidth=1.5)
    ax.add_patch(entity_box2)
    ax.text(5, 7.3, 's2_BERT Encoder\n(H^e)', ha='center', va='center', fontsize=10)
    
    # Image processing branch
    img_box1 = FancyBboxPatch((6.5, 8), 2.5, 0.6, boxstyle="round,pad=0.1", 
                              facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(img_box1)
    ax.text(7.75, 8.3, 'ResNet-152', ha='center', va='center', fontsize=10)
    
    img_box2 = FancyBboxPatch((6.5, 7), 2.5, 0.6, boxstyle="round,pad=0.1", 
                              facecolor=COLORS['model'], edgecolor='black', linewidth=1.5)
    ax.add_patch(img_box2)
    ax.text(7.75, 7.3, 'Visual Features\n(2048×49)', ha='center', va='center', fontsize=10)
    
    # Cross-attention
    cross_box = FancyBboxPatch((3.75, 5.5), 2.5, 0.6, boxstyle="round,pad=0.1", 
                               facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(cross_box)
    ax.text(5, 5.8, 'Cross-Attention\nEntity-Image', ha='center', va='center', fontsize=10)
    
    # Multimodal fusion
    fusion_box = FancyBboxPatch((2.5, 4), 5, 0.6, boxstyle="round,pad=0.1", 
                                facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(fusion_box)
    ax.text(5, 4.3, 'Multimodal Fusion: [H^v; H^s]', ha='center', va='center', fontsize=10)
    
    # Combined attention
    comb_att_box = FancyBboxPatch((2.5, 2.5), 5, 0.6, boxstyle="round,pad=0.1", 
                                  facecolor=COLORS['model'], edgecolor='black', linewidth=1.5)
    ax.add_patch(comb_att_box)
    ax.text(5, 2.8, 'Combined Attention\nEncoder', ha='center', va='center', fontsize=10)
    
    # Pooling
    pool_box = FancyBboxPatch((2.5, 1), 5, 0.6, boxstyle="round,pad=0.1", 
                             facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(pool_box)
    ax.text(5, 1.3, 'Pooling (concat/cls/first)', ha='center', va='center', fontsize=10)
    
    # Classification
    class_box = FancyBboxPatch((3.5, -0.5), 3, 0.6, boxstyle="round,pad=0.1", 
                              facecolor=COLORS['output'], edgecolor='black', linewidth=1.5)
    ax.add_patch(class_box)
    ax.text(5, -0.2, 'Classifier\n(3 Kelas Sentimen)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    # Input to processing
    ax.arrow(5, 9.5, 0, -0.3, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    # Processing to encoders
    ax.arrow(2.25, 8, 0, -0.3, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(5, 8, 0, -0.3, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(7.75, 8, 0, -0.3, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    # Encoders to cross-attention
    ax.arrow(5, 7, 0, -0.9, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(7.75, 7, -1.5, -0.9, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    # Cross-attention to fusion
    ax.arrow(5, 5.5, 0, -0.9, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(2.25, 7, 0.25, -1.5, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    # Fusion to combined attention
    ax.arrow(5, 4, 0, -0.9, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    # Combined attention to pooling
    ax.arrow(5, 2.5, 0, -0.9, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    # Pooling to classification
    ax.arrow(5, 1, 0, -0.9, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'diagram_alir_tombert.png', dpi=300, bbox_inches='tight')
    print(f"✓ Diagram TomBERT disimpan: {OUTPUT_DIR / 'diagram_alir_tombert.png'}")
    plt.close()

def create_tfidf_svm_diagram():
    """Membuat diagram alir TF-IDF+SVM"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Diagram Alir TF-IDF + SVM', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Input
    input_box = FancyBboxPatch((3, 9.5), 4, 0.8, boxstyle="round,pad=0.1", 
                               facecolor=COLORS['input'], edgecolor='black', linewidth=1.5)
    ax.add_patch(input_box)
    ax.text(5, 9.9, 'Input: Teks + Gambar', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Text preprocessing
    prep_box = FancyBboxPatch((2, 8), 6, 0.6, boxstyle="round,pad=0.1", 
                             facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(prep_box)
    ax.text(5, 8.3, 'Text Preprocessing: Cleaning, Tokenization, Stemming, Lemmatization', 
            ha='center', va='center', fontsize=10)
    
    # Feature extraction branches
    # TF-IDF branch
    tfidf_box1 = FancyBboxPatch((1, 6.5), 3.5, 0.6, boxstyle="round,pad=0.1", 
                               facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(tfidf_box1)
    ax.text(2.75, 6.8, 'TF-IDF Vectorization\n(ngram 1-4, max_features)', 
            ha='center', va='center', fontsize=10)
    
    tfidf_box2 = FancyBboxPatch((1, 5.5), 3.5, 0.6, boxstyle="round,pad=0.1", 
                               facecolor=COLORS['model'], edgecolor='black', linewidth=1.5)
    ax.add_patch(tfidf_box2)
    ax.text(2.75, 5.8, 'TF-IDF Features\n(Sparse Matrix)', ha='center', va='center', fontsize=10)
    
    # Image features branch
    img_box1 = FancyBboxPatch((5.5, 6.5), 3.5, 0.6, boxstyle="round,pad=0.1", 
                             facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(img_box1)
    ax.text(7.25, 6.8, 'Image Feature Extraction\n(Histogram, Texture, Shape)', 
            ha='center', va='center', fontsize=10)
    
    img_box2 = FancyBboxPatch((5.5, 5.5), 3.5, 0.6, boxstyle="round,pad=0.1", 
                             facecolor=COLORS['model'], edgecolor='black', linewidth=1.5)
    ax.add_patch(img_box2)
    ax.text(7.25, 5.8, 'Image Features\n(20-50 features)', ha='center', va='center', fontsize=10)
    
    # Sentiment features
    sent_box = FancyBboxPatch((2, 4), 6, 0.6, boxstyle="round,pad=0.1", 
                            facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(sent_box)
    ax.text(5, 4.3, 'Sentiment Features: Polarity, Subjectivity, VADER', 
            ha='center', va='center', fontsize=10)
    
    # Feature combination
    comb_box = FancyBboxPatch((2, 2.5), 6, 0.6, boxstyle="round,pad=0.1", 
                             facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(comb_box)
    ax.text(5, 2.8, 'Feature Combination: TF-IDF + Image + Sentiment', 
            ha='center', va='center', fontsize=10)
    
    # Optional: PCA/Feature Selection
    pca_box = FancyBboxPatch((2, 1), 6, 0.6, boxstyle="round,pad=0.1", 
                             facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(pca_box)
    ax.text(5, 1.3, 'Optional: PCA / Feature Selection / Scaling', 
            ha='center', va='center', fontsize=10)
    
    # SVM Classification
    svm_box = FancyBboxPatch((3, -0.5), 4, 0.6, boxstyle="round,pad=0.1", 
                            facecolor=COLORS['output'], edgecolor='black', linewidth=1.5)
    ax.add_patch(svm_box)
    ax.text(5, -0.2, 'SVM Classifier\n(3 Kelas Sentimen)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    ax.arrow(5, 9.5, 0, -0.3, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(5, 8, 0, -0.3, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    # Branch to TF-IDF and Image
    ax.arrow(2.75, 8, 0, -0.8, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(7.25, 8, 0, -0.8, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    ax.arrow(2.75, 6.5, 0, -0.3, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(7.25, 6.5, 0, -0.3, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    # To sentiment features
    ax.arrow(2.75, 5.5, 0.25, -0.5, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(7.25, 5.5, -0.25, -0.5, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    ax.arrow(5, 4, 0, -0.9, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(5, 2.5, 0, -0.9, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(5, 1, 0, -0.9, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'diagram_alir_tfidf_svm.png', dpi=300, bbox_inches='tight')
    print(f"✓ Diagram TF-IDF+SVM disimpan: {OUTPUT_DIR / 'diagram_alir_tfidf_svm.png'}")
    plt.close()

def create_gnn_diagram():
    """Membuat diagram alir GNN"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Diagram Alir GNN (Graph Neural Network) untuk Deteksi Rumor', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Input
    input_box = FancyBboxPatch((3, 9.5), 4, 0.8, boxstyle="round,pad=0.1", 
                               facecolor=COLORS['input'], edgecolor='black', linewidth=1.5)
    ax.add_patch(input_box)
    ax.text(5, 9.9, 'Input: Teks Tweet + Metadata', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Graph construction
    graph_box1 = FancyBboxPatch((2, 8), 6, 0.6, boxstyle="round,pad=0.1", 
                               facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(graph_box1)
    ax.text(5, 8.3, 'Graph Construction: Nodes (tweets/users), Edges (retweet/reply/follow)', 
            ha='center', va='center', fontsize=10)
    
    # Node features
    node_box = FancyBboxPatch((2, 6.5), 6, 0.6, boxstyle="round,pad=0.1", 
                             facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(node_box)
    ax.text(5, 6.8, 'Node Feature Extraction: TF-IDF dari teks tweet', 
            ha='center', va='center', fontsize=10)
    
    # Adjacency matrix
    adj_box = FancyBboxPatch((2, 5), 6, 0.6, boxstyle="round,pad=0.1", 
                            facecolor=COLORS['model'], edgecolor='black', linewidth=1.5)
    ax.add_patch(adj_box)
    ax.text(5, 5.3, 'Adjacency Matrix: Representasi hubungan antar node', 
            ha='center', va='center', fontsize=10)
    
    # GAT layers
    gat_box1 = FancyBboxPatch((1, 3), 3.5, 0.6, boxstyle="round,pad=0.1", 
                              facecolor=COLORS['model'], edgecolor='black', linewidth=1.5)
    ax.add_patch(gat_box1)
    ax.text(2.75, 3.3, 'Multi-Head Attention\n(GAT Layer 1)', ha='center', va='center', fontsize=10)
    
    gat_box2 = FancyBboxPatch((5.5, 3), 3.5, 0.6, boxstyle="round,pad=0.1", 
                             facecolor=COLORS['model'], edgecolor='black', linewidth=1.5)
    ax.add_patch(gat_box2)
    ax.text(7.25, 3.3, 'Output Attention\n(GAT Layer 2)', ha='center', va='center', fontsize=10)
    
    # Node embedding
    embed_box = FancyBboxPatch((2, 1.5), 6, 0.6, boxstyle="round,pad=0.1", 
                              facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(embed_box)
    ax.text(5, 1.8, 'Node Embeddings: Representasi node setelah GAT', 
            ha='center', va='center', fontsize=10)
    
    # Classification
    class_box = FancyBboxPatch((3, -0.5), 4, 0.6, boxstyle="round,pad=0.1", 
                              facecolor=COLORS['output'], edgecolor='black', linewidth=1.5)
    ax.add_patch(class_box)
    ax.text(5, -0.2, 'Classifier\n(Rumor / Non-Rumor)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    ax.arrow(5, 9.5, 0, -0.3, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(5, 8, 0, -0.3, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(5, 6.5, 0, -0.3, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(5, 5, 0, -0.3, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    # Branch to GAT layers
    ax.arrow(2.75, 5, 0, -1.7, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(7.25, 5, 0, -1.7, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    # GAT to embedding
    ax.arrow(2.75, 3, 0.25, -0.5, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(7.25, 3, -0.25, -0.5, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    ax.arrow(5, 1.5, 0, -0.9, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'diagram_alir_gnn.png', dpi=300, bbox_inches='tight')
    print(f"✓ Diagram GNN disimpan: {OUTPUT_DIR / 'diagram_alir_gnn.png'}")
    plt.close()

def create_xai_diagram():
    """Membuat diagram alir XAI"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Diagram Alir XAI (Explainable AI) dengan GPT + BLIP', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Input: Predictions
    pred_box = FancyBboxPatch((3, 9.5), 4, 0.8, boxstyle="round,pad=0.1", 
                              facecolor=COLORS['input'], edgecolor='black', linewidth=1.5)
    ax.add_patch(pred_box)
    ax.text(5, 9.9, 'Input: Prediksi Model (Teks + Label + Probabilitas)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Image captioning branch
    blip_box1 = FancyBboxPatch((1, 7.5), 3.5, 0.6, boxstyle="round,pad=0.1", 
                              facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(blip_box1)
    ax.text(2.75, 7.8, 'BLIP Image Captioning', ha='center', va='center', fontsize=10)
    
    blip_box2 = FancyBboxPatch((1, 6.5), 3.5, 0.6, boxstyle="round,pad=0.1", 
                              facecolor=COLORS['model'], edgecolor='black', linewidth=1.5)
    ax.add_patch(blip_box2)
    ax.text(2.75, 6.8, 'Image Caption\n(Deskripsi Gambar)', ha='center', va='center', fontsize=10)
    
    # Text processing
    text_box = FancyBboxPatch((5.5, 7), 3.5, 0.6, boxstyle="round,pad=0.1", 
                             facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(text_box)
    ax.text(7.25, 7.3, 'Text Processing\n(Original Tweet)', ha='center', va='center', fontsize=10)
    
    # Context assembly
    context_box = FancyBboxPatch((2, 5), 6, 0.6, boxstyle="round,pad=0.1", 
                                facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(context_box)
    ax.text(5, 5.3, 'Context Assembly: Teks + Caption + Label + Probabilitas', 
            ha='center', va='center', fontsize=10)
    
    # GPT prompt construction
    prompt_box = FancyBboxPatch((2, 3.5), 6, 0.6, boxstyle="round,pad=0.1", 
                               facecolor=COLORS['process'], edgecolor='black', linewidth=1.5)
    ax.add_patch(prompt_box)
    ax.text(5, 3.8, 'GPT Prompt Construction: System + User Prompt', 
            ha='center', va='center', fontsize=10)
    
    # GPT API call
    gpt_box = FancyBboxPatch((2, 2), 6, 0.6, boxstyle="round,pad=0.1", 
                             facecolor=COLORS['model'], edgecolor='black', linewidth=1.5)
    ax.add_patch(gpt_box)
    ax.text(5, 2.3, 'GPT-4o-mini API Call\n(OpenAI)', ha='center', va='center', fontsize=10)
    
    # Explanation output
    output_box = FancyBboxPatch((3, 0), 4, 0.6, boxstyle="round,pad=0.1", 
                               facecolor=COLORS['output'], edgecolor='black', linewidth=1.5)
    ax.add_patch(output_box)
    ax.text(5, 0.3, 'Output: Penjelasan XAI (Bahasa Indonesia)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    ax.arrow(5, 9.5, 0, -0.3, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    # Branch to BLIP and Text
    ax.arrow(2.75, 9.5, -0.5, -1.2, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(7.25, 9.5, 0.5, -1.2, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    ax.arrow(2.75, 7.5, 0, -0.3, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    # To context assembly
    ax.arrow(2.75, 6.5, 0.25, -0.5, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(7.25, 7, -0.25, -0.5, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    ax.arrow(5, 5, 0, -0.9, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(5, 3.5, 0, -0.9, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    ax.arrow(5, 2, 0, -0.9, head_width=0.15, head_length=0.1, fc=COLORS['arrow'], ec=COLORS['arrow'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'diagram_alir_xai.png', dpi=300, bbox_inches='tight')
    print(f"✓ Diagram XAI disimpan: {OUTPUT_DIR / 'diagram_alir_xai.png'}")
    plt.close()

def main():
    """Membuat semua diagram alir"""
    print("Membuat diagram alir untuk BAB 4 Perancangan...")
    print("=" * 60)
    
    create_tombert_diagram()
    create_tfidf_svm_diagram()
    create_gnn_diagram()
    create_xai_diagram()
    
    print("=" * 60)
    print(f"✓ Semua diagram telah dibuat di: {OUTPUT_DIR}")
    print("\nFile yang dihasilkan:")
    print("  1. diagram_alir_tombert.png")
    print("  2. diagram_alir_tfidf_svm.png")
    print("  3. diagram_alir_gnn.png")
    print("  4. diagram_alir_xai.png")

if __name__ == "__main__":
    main()

