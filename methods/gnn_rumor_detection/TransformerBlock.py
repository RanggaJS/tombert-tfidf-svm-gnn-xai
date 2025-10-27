# coding=utf-8
"""
Transformer Block - OPTIMIZED VERSION
Enhanced multi-head attention with better architecture
"""

import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
import math


class TransformerBlock(nn.Module):
    """
    OPTIMIZED Transformer Block with:
    - Better initialization
    - Pre-layer normalization
    - Improved feed-forward network
    - Better dropout strategy
    """

    def __init__(self, input_size, d_k=64, d_v=64, n_heads=8, 
                 is_layer_norm=True, attn_dropout=0.1, ffn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size // n_heads
        self.d_v = d_v if d_v is not None else input_size // n_heads

        self.is_layer_norm = is_layer_norm
        
        # OPTIMIZED: Pre-layer normalization
        if is_layer_norm:
            self.layer_norm1 = nn.LayerNorm(normalized_shape=input_size)
            self.layer_norm2 = nn.LayerNorm(normalized_shape=input_size)

        # OPTIMIZED: Better weight initialization
        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * self.d_v))
        self.W_o = nn.Parameter(torch.Tensor(self.d_v * n_heads, input_size))
        
        # OPTIMIZED: Enhanced FFN with GELU activation
        self.linear1 = nn.Linear(input_size, input_size * 4)  # Increased from 1x to 4x
        self.linear2 = nn.Linear(input_size * 4, input_size)
        
        # OPTIMIZED: Separate dropout for attention and FFN
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.ffn_dropout = nn.Dropout(ffn_dropout)
        self.output_dropout = nn.Dropout(attn_dropout)
        
        # OPTIMIZED: Use GELU instead of ReLU
        self.activation = nn.GELU()
        
        self.__init_weights__()

    def __init_weights__(self):
        """OPTIMIZED: Better weight initialization"""
        # Xavier initialization with gain
        init.xavier_normal_(self.W_q, gain=1.0 / math.sqrt(2.0))
        init.xavier_normal_(self.W_k, gain=1.0 / math.sqrt(2.0))
        init.xavier_normal_(self.W_v, gain=1.0 / math.sqrt(2.0))
        init.xavier_normal_(self.W_o, gain=1.0 / math.sqrt(2.0))

        # Kaiming initialization for FFN
        init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
        
        # Initialize biases to zero
        init.constant_(self.linear1.bias, 0)
        init.constant_(self.linear2.bias, 0)

    def FFN(self, X):
        """
        OPTIMIZED Feed-Forward Network with:
        - Larger hidden dimension (4x)
        - GELU activation
        - Better dropout
        """
        output = self.linear1(X)
        output = self.activation(output)
        output = self.ffn_dropout(output)
        output = self.linear2(output)
        output = self.output_dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, mask=None, epsilon=1e-6):
        '''
        OPTIMIZED Scaled Dot-Product Attention with:
        - Better numerical stability
        - Optional masking support
        - Improved dropout
        
        Args:
            Q: (batch_size, n_heads, max_q_words, d_k)
            K: (batch_size, n_heads, max_k_words, d_k)
            V: (batch_size, n_heads, max_v_words, d_v)
            mask: Optional attention mask
            epsilon: Small value for numerical stability
        
        Returns:
            V_att: (batch_size, n_heads, max_q_words, d_v)
        '''
        temperature = self.d_k ** 0.5
        
        # Compute attention scores
        Q_K = torch.matmul(Q, K.transpose(-2, -1)) / (temperature + epsilon)
        
        # OPTIMIZED: Apply mask if provided
        if mask is not None:
            Q_K = Q_K.masked_fill(mask == 0, -1e9)
        
        # OPTIMIZED: Numerical stability for softmax
        Q_K_max = Q_K.max(dim=-1, keepdim=True)[0]
        Q_K_score = F.softmax(Q_K - Q_K_max, dim=-1)
        Q_K_score = self.attn_dropout(Q_K_score)

        # Apply attention to values
        V_att = torch.matmul(Q_K_score, V)
        return V_att

    def multi_head_attention(self, Q, K, V, mask=None):
        '''
        OPTIMIZED Multi-Head Attention
        
        Args:
            Q: (batch_size, max_q_words, input_size)
            K: (batch_size, max_k_words, input_size)
            V: (batch_size, max_v_words, input_size)
            mask: Optional attention mask
        
        Returns:
            output: (batch_size, max_q_words, input_size)
        '''
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        # Linear projections
        Q_ = torch.matmul(Q, self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = torch.matmul(K, self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = torch.matmul(V, self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        # Transpose for attention computation
        Q_ = Q_.transpose(1, 2)  # (bsz, n_heads, q_len, d_k)
        K_ = K_.transpose(1, 2)  # (bsz, n_heads, k_len, d_k)
        V_ = V_.transpose(1, 2)  # (bsz, n_heads, v_len, d_v)

        # Scaled dot-product attention
        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask)
        
        # Concatenate heads
        V_att = V_att.transpose(1, 2).contiguous().view(bsz, q_len, self.n_heads * self.d_v)

        # Output projection
        output = torch.matmul(V_att, self.W_o)
        output = self.output_dropout(output)
        
        return output

    def forward(self, Q, K, V, mask=None):
        '''
        OPTIMIZED Forward pass with pre-layer normalization
        
        Args:
            Q: (batch_size, max_q_words, input_size)
            K: (batch_size, max_k_words, input_size)
            V: (batch_size, max_v_words, input_size)
            mask: Optional attention mask
        
        Returns:
            output: (batch_size, max_q_words, input_size)
        '''
        # OPTIMIZED: Pre-layer normalization
        if self.is_layer_norm:
            # Multi-head attention with residual
            Q_norm = self.layer_norm1(Q)
            V_att = self.multi_head_attention(Q_norm, K, V, mask)
            X = Q + V_att
            
            # Feed-forward network with residual
            X_norm = self.layer_norm2(X)
            output = X + self.FFN(X_norm)
        else:
            # Post-layer normalization (original)
            V_att = self.multi_head_attention(Q, K, V, mask)
            X = Q + V_att
            output = X + self.FFN(X)
        
        return output

    def get_attention_weights(self, Q, K, V, mask=None):
        """
        OPTIMIZED: Extract attention weights for visualization
        
        Returns:
            attention_weights: (batch_size, n_heads, max_q_words, max_k_words)
        """
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()

        Q_ = torch.matmul(Q, self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = torch.matmul(K, self.W_k).view(bsz, k_len, self.n_heads, self.d_k)

        Q_ = Q_.transpose(1, 2)
        K_ = K_.transpose(1, 2)

        temperature = self.d_k ** 0.5
        Q_K = torch.matmul(Q_, K_.transpose(-2, -1)) / temperature
        
        if mask is not None:
            Q_K = Q_K.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(Q_K, dim=-1)
        
        return attention_weights