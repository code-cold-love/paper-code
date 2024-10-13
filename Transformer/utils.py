#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import sqrt

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    query, key, value, query_mask=None, key_mask=None, mask=None
):
    """缩放点积注意力

    Args:
        query (torch.Tensor): 形状为 (batch_size, seq_len_q, dim_k)
        key (torch.Tensor): 形状为 (batch_size, seq_len_k, dim_k)
        value (torch.Tensor): 形状为 (batch_size, seq_len_k, dim_v)
        query_mask (torch.Tensor, optional): query 序列的 mask. Defaults to None.
        key_mask (torch.Tensor, optional): key 序列的 mask. Defaults to None.
        mask (torch.Tensor, optional): value 序列的 mask. Defaults to None.

    Returns:
        torch.Tensor: 形状为 (batch_size, seq_len_q, dim_v)
    """
    # # nn.Embedding 层把输入的词语序列映射到了尺寸为 (batch_size, seq_len, hidden_dim) 的张量
    dim_k = query.size(-1)

    # torch.bmm() 要求第一个输入张量的第 2 维和第二个输入张量的第 1 维的长度相同
    # key.transpose(1, 2) 交换张量的两个维度，把 key 的形状变为 (batch_size, dim_k, seq_len_k)
    # 输出 scores 的形状为 (batch_size, seq_len_q, seq_len_k)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        # 填充 (padding) 字符不应该参与计算，因此将对应的注意力分数设置为负无穷，这样其注意力权重为 0
        scores = scores.masked_fill(mask == 0, -float("inf"))
    # 注意力权重 w_ij 表示第 i 个 query 向量和第 j 个 key 向量的关联程度
    weights = F.softmax(scores, dim=-1)  # 应用 softmax 标准化注意力权重
    return torch.bmm(weights, value)  # 结果形状为 (batch_size, seq_len_q, dim_v)
