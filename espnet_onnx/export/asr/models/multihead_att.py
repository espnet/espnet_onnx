import os
import math

import torch
import torch.nn as nn
import numpy as np

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention


class OnnxMultiHeadedAttention(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.d_k = model.d_k
        self.h = model.h
        self.linear_q = model.linear_q
        self.linear_k = model.linear_k
        self.linear_v = model.linear_v
        self.linear_out = model.linear_out
        self.attn = model.attn
        self.dropout = model.dropout
        self.model = model
        self.all_head_size = self.h * self.d_k
        self.min_value = float(
                np.finfo(torch.tensor(0, dtype=torch.float32).numpy().dtype).min
            )
    
    def forward(self, query, key, value, mask):
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward_qkv(self, query, key, value):
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        return q, k, v
    
    def forward_attention(self, value, scores, mask):
        if mask is not None:
            scores = scores + mask

        self.attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(self.attn)
        context_layer = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return self.linear_out(context_layer)  # (batch, time1, d_model)
