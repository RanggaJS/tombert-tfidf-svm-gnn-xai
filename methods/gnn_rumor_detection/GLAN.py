# coding=utf-8
"""
Graph-based Local Attention Network (GLAN) - OPTIMIZED VERSION
Enhanced with better architecture and training strategies
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from GAT import GAT
from TransformerBlock import TransformerBlock
from NeuralNetwork import NeuralNetwork


class GLAN(NeuralNetwork):
    """
    OPTIMIZED Graph-based Local Attention Network
    
    Improvements:
    - Better fusion strategies
    - Enhanced CNN architecture
    - Improved classifier
    - Better regularization
    """

    def __init__(self, config, adj):
        super(GLAN, self).__init__()
        self.config = config
        self.uV = adj.shape[0]
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        # OPTIMIZED: Enhanced multi-head attention
        self.mh_attention = TransformerBlock(
            input_size=300,
            d_k=64,  # Increased from default
            d_v=64,
            n_heads=8,
            is_layer_norm=True,  # Enable layer norm
            attn_dropout=dropout_rate
        )
        
        # Word embedding
        self.word_embedding = nn.Embedding(V, D, padding_idx=0, 
                                          _weight=torch.from_numpy(embedding_weights))
        
        # OPTIMIZED: Enhanced GAT with better hyperparameters
        self.relation_embedding = GAT(
            nfeat=300,
            uV=self.uV,
            adj=adj,
            hidden=32,  # Increased from 16
            nb_heads=8,
            n_output=300,
            dropout=0.3,  # Decreased from 0.5
            alpha=0.2,  # Decreased from 0.3
            use_residual=True
        )

        # OPTIMIZED: Enhanced CNN with more filters and varied kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(300, 128, kernel_size=K, padding=K//2)  # Increased from 100 to 128
            for K in config['kernel_sizes']
        ])
        
        # OPTIMIZED: Add batch normalization for CNNs
        self.conv_bns = nn.ModuleList([
            nn.BatchNorm1d(128) for _ in config['kernel_sizes']
        ])
        
        self.max_poolings = nn.ModuleList([
            nn.AdaptiveMaxPool1d(1)  # Use adaptive pooling
            for K in config['kernel_sizes']
        ])

        # OPTIMIZED: Better dropout strategy
        self.dropout = nn.Dropout(dropout_rate)
        self.feature_dropout = nn.Dropout(dropout_rate * 0.5)  # Lower dropout for features
        
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()  # Add GELU activation

        # OPTIMIZED: Enhanced classifier with more layers
        cnn_output_size = 128 * len(config['kernel_sizes'])
        total_feature_size = 300 + cnn_output_size  # GAT + CNN features
        
        # Multi-layer classifier
        self.fc1 = nn.Linear(total_feature_size, 512)  # Increased from 300
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.fc_out = nn.Linear(128, config['num_classes'])
        
        # OPTIMIZED: Attention fusion layer
        self.fusion_attention = nn.Sequential(
            nn.Linear(total_feature_size, total_feature_size // 4),
            nn.ReLU(),
            nn.Linear(total_feature_size // 4, total_feature_size),
            nn.Sigmoid()
        )

        self.init_weight()
        print(self)

    def init_weight(self):
        """OPTIMIZED: Better weight initialization"""
        # Initialize linear layers
        for m in [self.fc1, self.fc2, self.fc3, self.fc_out]:
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        # Initialize conv layers
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
        
        # Initialize fusion attention
        for m in self.fusion_attention:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, X_tid, X_text):
        # Text processing with transformer
        X_text = self.word_embedding(X_text)  # (N*C, W, D)
        X_text = self.mh_attention(X_text, X_text, X_text)
        X_text = X_text.permute(0, 2, 1)  # (N*C, D, W)

        # OPTIMIZED: Graph relation embedding
        rembedding = self.relation_embedding(X_tid)  # (N*C, 300)

        # OPTIMIZED: Enhanced CNN processing
        conv_outputs = []
        for conv, bn, max_pool in zip(self.convs, self.conv_bns, self.max_poolings):
            # Convolution
            conv_out = conv(X_text)  # (N*C, 128, W)
            conv_out = bn(conv_out)
            conv_out = self.gelu(conv_out)
            
            # Pooling
            pooled = max_pool(conv_out)  # (N*C, 128, 1)
            pooled = pooled.squeeze(-1)  # (N*C, 128)
            
            conv_outputs.append(pooled)
        
        # Concatenate CNN features
        cnn_features = torch.cat(conv_outputs, dim=1)  # (N*C, 128*num_kernels)
        
        # OPTIMIZED: Combine graph and CNN features
        combined_features = torch.cat([rembedding, cnn_features], dim=1)
        
        # OPTIMIZED: Apply attention fusion
        attention_weights = self.fusion_attention(combined_features)
        combined_features = combined_features * attention_weights
        
        # Feature dropout
        features = self.feature_dropout(combined_features)

        # OPTIMIZED: Multi-layer classification with residual connections
        # Layer 1
        h1 = self.fc1(features)
        h1 = self.bn1(h1)
        h1 = self.gelu(h1)
        h1 = self.dropout(h1)
        
        # Layer 2
        h2 = self.fc2(h1)
        h2 = self.bn2(h2)
        h2 = self.gelu(h2)
        h2 = self.dropout(h2)
        
        # Layer 3
        h3 = self.fc3(h2)
        h3 = self.bn3(h3)
        h3 = self.gelu(h3)
        h3 = self.dropout(h3)
        
        # Output
        output = self.fc_out(h3)
        
        return output

    def get_feature_importance(self, X_tid, X_text):
        """
        OPTIMIZED: Method to get feature importance for interpretability
        """
        with torch.no_grad():
            X_text = self.word_embedding(X_text)
            X_text = self.mh_attention(X_text, X_text, X_text)
            X_text = X_text.permute(0, 2, 1)

            rembedding = self.relation_embedding(X_tid)

            conv_outputs = []
            for conv, bn, max_pool in zip(self.convs, self.conv_bns, self.max_poolings):
                conv_out = conv(X_text)
                conv_out = bn(conv_out)
                conv_out = self.gelu(conv_out)
                pooled = max_pool(conv_out).squeeze(-1)
                conv_outputs.append(pooled)
            
            cnn_features = torch.cat(conv_outputs, dim=1)
            combined_features = torch.cat([rembedding, cnn_features], dim=1)
            
            attention_weights = self.fusion_attention(combined_features)
            
        return attention_weights.cpu().numpy()