"""Encoder self-attention layer definition."""

import torch
from torch import nn


class OnnxEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(self, model, model_type="espnet"):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = model.self_attn
        if model_type == "espnet":
            self.feed_forward = model.feed_forward
            if hasattr(model, "norm1"):
                self.norm1 = model.norm1
            if hasattr(model, "norm2"):
                self.norm2 = model.norm2
            self.size = model.size
            self.normalize_before = model.normalize_before
            self.concat_after = model.concat_after
            if self.concat_after:
                self.concat_linear = model.concat_linear

        elif model_type == "hubert":
            self.feed_forward = FeedForward(model.fc1, model.activation_fn, model.fc2)
            self.norm1 = model.self_attn_layer_norm
            self.norm2 = model.final_layer_norm
            self.size = model.embedding_dim
            self.normalize_before = model.layer_norm_first
            self.concat_after = False

    def forward(self, x, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if cache is not None:
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
        else:
            x_q = x

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = self.concat_linear(x_concat) + residual
        else:
            x = self.self_attn(x_q, x, x, mask) + residual

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = self.feed_forward(x) + residual
        if not self.normalize_before:
            x = self.norm2(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, mask


class FeedForward(nn.Module):
    def __init__(self, fc1, activation, fc2):
        super().__init__()
        self.fc1 = fc1
        self.activation = activation
        self.fc2 = fc2

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
