#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch import nn
from transformers import AutoConfig, AutoTokenizer

from attention import MultiHeadAttention
from encoder import TransformerEncoder, TransformerEncoderLayer
from layers import FeedForward

if __name__ == "__main__":
    model_ckpt = "bert-base-uncased"
    BERT_PATH = r"D:\Privacy4\transformers\bert-base-uncased_L-12_H-768_A-12"
    text = "time flies like an arrow"

    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    config = AutoConfig.from_pretrained(BERT_PATH)
    token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
    inputs_embeds = token_emb(inputs.input_ids)

    multihead_attn = MultiHeadAttention(config)
    query = key = value = inputs_embeds
    attn_output = multihead_attn(query, key, value)
    print(attn_output.shape)  # 结果形状为 (batch_size, seq_len_q, dim_v) ([1, 5, 768])

    feed_forward = FeedForward(config)
    ff_outputs = feed_forward(attn_output)
    print(ff_outputs.size())

    encoder_layer = TransformerEncoderLayer(config)
    print(inputs_embeds.shape)
    print(encoder_layer(inputs_embeds).size())

    encoder = TransformerEncoder(config)
    print(encoder(inputs.input_ids).size())
