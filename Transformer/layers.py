#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig


class FeedForward(nn.Module):
    def __init__(self, config: PretrainedConfig):
        """基于位置的前馈神经网络"""
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class PositionalEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig):
        """使用与位置相关的值模式来增强词向量"""
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids: torch.Tensor):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
