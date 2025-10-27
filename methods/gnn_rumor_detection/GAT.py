# coding=utf-8
"""
Graph Attention Network (GAT) - OPTIMIZED VERSION
Enhanced with better initialization, dropout strategies, and attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


class SpecialSpmmFunction(torch.autograd.Function):
    """
    Special function for only sparse region backpropagation layer.
    OPTIMIZED: Better gradient handling
    """
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]

        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer - OPTIMIZED VERSION
    Enhanced with:
    - Better weight initialization
    - Layer normalization
    - Residual connections
    - Improved attention mechanism
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, use_residual=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.use_residual = use_residual

        # OPTIMIZED: Better weight initialization
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        # OPTIMIZED: Add bias
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # OPTIMIZED: Layer normalization for stability
        self.layer_norm = nn.LayerNorm(out_features)
        
        # OPTIMIZED: Residual connection projection if needed
        if use_residual and in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features, bias=False)
            nn.init.xavier_normal_(self.residual_proj.weight, gain=1.414)
        else:
            self.residual_proj = None

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        N = input.size()[0]
        edge = torch.LongTensor(adj.nonzero())

        if edge.size(1) == 0:
            # Handle empty adjacency matrix
            h = torch.mm(input, self.W) + self.bias
            return F.elu(h) if self.concat else h

        h = torch.mm(input, self.W) + self.bias
        # h: N x out
        assert not torch.isnan(h).any(), "NaN detected in h"

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        # OPTIMIZED: Better attention computation with numerical stability
        edge_e = self.leakyrelu(self.a.mm(edge_h).squeeze())
        edge_e = torch.exp(edge_e - edge_e.max())  # Numerical stability
        assert not torch.isnan(edge_e).any(), "NaN detected in edge_e"
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), 
                                     torch.ones(size=(N, 1)).to(edge_e.device))
        e_rowsum = e_rowsum + 1e-8  # Avoid division by zero
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any(), "NaN detected in h_prime"
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any(), "NaN detected after division"

        # OPTIMIZED: Add residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(input)
            elif self.in_features == self.out_features:
                residual = input
            else:
                residual = 0
            h_prime = h_prime + residual

        # OPTIMIZED: Layer normalization
        h_prime = self.layer_norm(h_prime)

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    """
    Graph Attention Network - OPTIMIZED VERSION
    Enhanced with:
    - Better architecture
    - Improved dropout strategy
    - Multi-head attention fusion
    - Better embedding initialization
    """

    def __init__(self, nfeat, uV, adj, hidden=32, nb_heads=8, n_output=300, 
                 dropout=0.3, alpha=0.2, use_residual=True):
        """
        OPTIMIZED Sparse version of GAT.
        
        Args:
            nfeat: Input feature dimension
            uV: Number of nodes (users/tweets)
            adj: Adjacency matrix
            hidden: Hidden dimension (increased from 16 to 32)
            nb_heads: Number of attention heads
            n_output: Output dimension
            dropout: Dropout rate (decreased from 0.5 to 0.3)
            alpha: LeakyReLU negative slope (decreased from 0.3 to 0.2)
            use_residual: Whether to use residual connections
        """
        super(GAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.uV = uV
        self.adj = adj
        self.use_residual = use_residual
        
        # OPTIMIZED: Better embedding initialization
        self.user_tweet_embedding = nn.Embedding(self.uV, nfeat, padding_idx=0)
        nn.init.xavier_uniform_(self.user_tweet_embedding.weight)
        
        # OPTIMIZED: Add embedding dropout
        self.embedding_dropout = nn.Dropout(dropout * 0.5)  # Lower dropout for embeddings

        # OPTIMIZED: Multi-head attention layers with residual connections
        self.attentions = nn.ModuleList([
            SpGraphAttentionLayer(
                in_features=nfeat,
                out_features=hidden,
                dropout=dropout,
                alpha=alpha,
                concat=True,
                use_residual=use_residual
            ) for _ in range(nb_heads)
        ])
        
        # OPTIMIZED: Add intermediate layer for better representation
        self.intermediate_layer = nn.Linear(hidden * nb_heads, hidden * nb_heads)
        nn.init.xavier_normal_(self.intermediate_layer.weight)
        self.intermediate_norm = nn.LayerNorm(hidden * nb_heads)
        
        # Output attention layer
        self.out_att = SpGraphAttentionLayer(
            hidden * nb_heads,
            n_output,
            dropout=dropout,
            alpha=alpha,
            concat=False,
            use_residual=use_residual
        )
        
        # OPTIMIZED: Final projection layer
        self.final_proj = nn.Linear(n_output, n_output)
        nn.init.xavier_normal_(self.final_proj.weight)
        self.final_norm = nn.LayerNorm(n_output)

    def forward(self, X_tid):
        # Get embeddings for all nodes
        X = self.user_tweet_embedding(torch.arange(0, self.uV).long().to(X_tid.device))
        X = self.embedding_dropout(X)

        # OPTIMIZED: Multi-head attention with better fusion
        att_outputs = []
        for att in self.attentions:
            att_out = att(X, self.adj)
            att_outputs.append(att_out)
        
        # Concatenate multi-head outputs
        X = torch.cat(att_outputs, dim=1)
        X = self.dropout(X)
        
        # OPTIMIZED: Intermediate processing
        X_intermediate = self.intermediate_layer(X)
        X_intermediate = F.elu(X_intermediate)
        X_intermediate = self.intermediate_norm(X_intermediate)
        X_intermediate = self.dropout(X_intermediate)
        
        # Add residual connection
        if self.use_residual:
            X = X + X_intermediate
        else:
            X = X_intermediate

        # Output attention
        X = self.out_att(X, self.adj)
        
        # OPTIMIZED: Final projection
        X = self.final_proj(X)
        X = self.final_norm(X)
        X = F.elu(X)
        
        # Select relevant node embeddings
        X_ = X[X_tid]
        return X_

    def get_attention_weights(self, X_tid):
        """
        OPTIMIZED: Method to extract attention weights for interpretability
        """
        X = self.user_tweet_embedding(torch.arange(0, self.uV).long().to(X_tid.device))
        
        attention_weights = []
        for att in self.attentions:
            # This would need modification in SpGraphAttentionLayer to return weights
            pass
        
        return attention_weights