#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn

from Transformer.decoder import TransformerDecoder
from Transformer.encoder import TransformerEncoder


class Transformer(nn.Module):
    def __init__(self, config, src_pad_idx, trg_pad_idx):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src: torch.Tensor):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg: torch.Tensor):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
