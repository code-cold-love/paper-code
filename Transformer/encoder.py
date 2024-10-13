#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn

from attention import MultiHeadAttention
from layers import FeedForward, PositionalEmbeddings


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        """Transformer编码器层"""
        # 采用 Pre layer normalization，将 Layer Normalization 放在 Skip Connections 的范围内
        # 这种做法通常训练过程会更加稳定，并且不需要任何学习率预热
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        hidden_state = self.layer_norm1(x)
        # Apply attention with a skip connection
        x = x + self.attention(
            hidden_state, hidden_state, hidden_state, mask=attention_mask
        )
        # Apply feed-forward with a skip connection
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = PositionalEmbeddings(config)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x
