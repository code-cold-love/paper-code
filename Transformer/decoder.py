#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from transformers import PretrainedConfig

from attention import MultiHeadAttention
from layers import PositionalEmbeddings, FeedForward


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig):
        """Transformer解码器层"""
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)

        self.enc_dec_attn = MultiHeadAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

        self.ffn = FeedForward(config)
        self.layer_norm3 = nn.LayerNorm(config.hidden_size)
        self.dropout3 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, dec, enc, trg_mask, src_mask) -> torch.Tensor:
        # compute self attention
        _x = dec
        x = self.self_attn(dec, dec, dec, mask=trg_mask)

        # add and norm
        x = self.dropout1(x)
        x = self.layer_norm1(x + _x)

        # compute encoder-decoder attention
        if enc:
            _x = x
            x = self.enc_dec_attn(query=x, key=enc, value=enc, mask=src_mask)

            # add and norm
            x = self.dropout2(x)
            x = self.layer_norm2(x + _x)

        # position-wise feed forward
        _x = x
        x = self.ffn(x)

        # add and norm
        x = self.dropout3(x)
        x = self.layer_norm3(x + _x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = PositionalEmbeddings(config)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x: torch.Tensor, enc_src: torch.Tensor, trg_mask, src_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, enc_src, trg_mask, src_mask)
        return self.linear(x)
