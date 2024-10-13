#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn

from Transformer.utils import scaled_dot_product_attention


class AttentionHead(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int):
        """注意力头"""
        super().__init__()
        # 三个独立的线性层，将 Q, K, V 映射到尺寸为 (batch_size, seq_len, head_dim) 的张量
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        attn_outputs = scaled_dot_product_attention(
            self.q(query), self.k(key), self.v(value), query_mask, key_mask, mask
        )
        return attn_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        """多头注意力"""
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        # 将每个注意力头的输出连接起来
        x = torch.cat(
            [h(query, key, value, query_mask, key_mask, mask) for h in self.heads],
            dim=-1,
        )
        # 然后通过一个线性层来得到最终的输出
        return self.output_linear(x)
