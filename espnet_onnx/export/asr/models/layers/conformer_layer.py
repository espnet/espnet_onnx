"""Encoder self-attention layer definition."""

import torch
from torch import nn


class OnnxConformerLayer(nn.Module):
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

    def __init__(self, model):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.model = model
        self.size = model.size
        self.stoch_layer_coeff = 1.0

    def forward(self, x_input, mask, cache=None):
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # whether to use macaron style
        if self.model.feed_forward_macaron is not None:
            residual = x
            if self.model.normalize_before:
                x = self.model.norm_ff_macaron(x)
            x = (
                residual
                + self.stoch_layer_coeff
                * self.model.ff_scale
                * self.model.feed_forward_macaron(x)
            )
            if not self.model.normalize_before:
                x = self.model.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.model.normalize_before:
            x = self.model.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.model.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.model.self_attn(x_q, x, x, mask)

        if self.model.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.stoch_layer_coeff * self.model.concat_linear(x_concat)
        else:
            x = residual + self.stoch_layer_coeff * x_att
        if not self.model.normalize_before:
            x = self.model.norm_mha(x)

        # convolution module
        if self.model.conv_module is not None:
            residual = x
            if self.model.normalize_before:
                x = self.model.norm_conv(x)
            x = residual + self.stoch_layer_coeff * self.model.conv_module(x)
            if not self.model.normalize_before:
                x = self.model.norm_conv(x)

        # feed forward module
        residual = x
        if self.model.normalize_before:
            x = self.model.norm_ff(x)
        x = (
            residual
            + self.stoch_layer_coeff * self.model.ff_scale * self.model.feed_forward(x)
        )
        if not self.model.normalize_before:
            x = self.model.norm_ff(x)

        if self.model.conv_module is not None:
            x = self.model.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask
