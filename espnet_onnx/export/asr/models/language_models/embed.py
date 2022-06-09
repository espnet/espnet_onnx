#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Modified by Masao-Someki
"""Positional Encoding Module."""

import math
import torch
import torch.nn as nn

from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.subsampling_without_posenc import Conv2dSubsamplingWOPosEnc
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,
    ScaledPositionalEncoding,
    RelPositionalEncoding,
    LegacyRelPositionalEncoding,
    StreamPositionalEncoding,
)


def get_pos_emb(pos_emb, max_seq_len=512):
    if isinstance(pos_emb, LegacyRelPositionalEncoding):
        return OnnxLegacyRelPositionalEncoding(pos_emb, max_seq_len)
    elif isinstance(pos_emb, ScaledPositionalEncoding):
        return OnnxScaledPositionalEncoding(pos_emb, max_seq_len)
    elif isinstance(pos_emb, RelPositionalEncoding):
        return OnnxRelPositionalEncoding(pos_emb, max_seq_len)
    elif isinstance(pos_emb, PositionalEncoding):
        return OnnxPositionalEncoding(pos_emb, max_seq_len)
    elif isinstance(pos_emb, StreamPositionalEncoding):
        return OnnxStreamPositionalEncoding(pos_emb, max_seq_len)
    elif (isinstance(pos_emb, nn.Sequential) and len(pos_emb) == 0) \
        or (isinstance(pos_emb, Conv2dSubsamplingWOPosEnc)):
        return pos_emb
    else:
        raise ValueError('Embedding model is not supported.')


class Embedding(nn.Module):
    def __init__(self, model, max_seq_len=512):
        super().__init__()
        self.model = model
        if not isinstance(model, nn.Embedding):
            if (
                isinstance(model, Conv2dSubsampling)
                or isinstance(model, Conv2dSubsampling2)
                or isinstance(model, Conv2dSubsampling6)
                or isinstance(model, Conv2dSubsampling8)
            ):
                self.model.out[-1] = get_pos_emb(model.out[-1], max_seq_len)
            else:
                self.model[-1] = get_pos_emb(model[-1], max_seq_len)

    def forward(self, x, mask=None):
        if mask is None:
            return self.model(x)
        else:
            xs = self.model(x, mask)
            if isinstance(self.model, Conv2dSubsampling):
                return xs, mask[:, :, :-2:2][:, :, :-2:2]
            elif isinstance(self.model, Conv2dSubsampling2):
                return xs, mask[:, :, :-2:2][:, :, :-2:1]
            elif isinstance(self.model, Conv2dSubsampling6):
                return xs, mask[:, :, :-2:2][:, :, :-4:3]
            elif isinstance(self.model, Conv2dSubsampling8):
                return xs, mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Perform pre-hook in load_state_dict for backward compatibility.

    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.

    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class OnnxPositionalEncoding(torch.nn.Module):
    """Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_seq_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position. Only for
        the class LegacyRelPositionalEncoding. We remove it in the current
        class RelPositionalEncoding.
    """

    def __init__(self, model, max_seq_len=512, reverse=False):
        """Construct an PositionalEncoding object."""
        super(OnnxPositionalEncoding, self).__init__()
        self.d_model = model.d_model
        self.reverse = model.reverse
        self.xscale = math.sqrt(self.d_model)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_seq_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None and self.pe.size(1) >= x.size(1):
            if self.pe.dtype != x.dtype or self.pe.device != x.device:
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
            return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(
                0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return x


class OnnxScaledPositionalEncoding(OnnxPositionalEncoding):
    """Scaled positional encoding module.

    See Sec. 3.2  https://arxiv.org/abs/1809.08895

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_seq_len (int): Maximum input length.

    """

    def __init__(self, model, max_seq_len=512):
        """Initialize class."""
        super().__init__(model, max_seq_len)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        x = x + self.alpha * self.pe[:, : x.size(1)]
        return x


class OnnxLegacyRelPositionalEncoding(OnnxPositionalEncoding):
    """Relative positional encoding module (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_seq_len (int): Maximum input length.

    """

    def __init__(self, model, max_seq_len=512):
        """Initialize class."""
        super().__init__(
            model, max_seq_len,
            reverse=True
        )

    def forward(self, x):
        """Compute positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).

        """
        x = x * self.xscale
        pos_emb = self.pe[:, : x.size(1)]
        return x, pos_emb


class OnnxRelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_seq_len (int): Maximum input length.

    """

    def __init__(self, model, max_seq_len=512):
        """Construct an PositionalEncoding object."""
        super(OnnxRelPositionalEncoding, self).__init__()
        self.d_model = model.d_model
        self.xscale = math.sqrt(self.d_model)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_seq_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None and self.pe.size(1) >= x.size(1) * 2 - 1:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.dtype != x.dtype or self.pe.device != x.device:
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
            return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        x = x * self.xscale
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1: self.pe.size(1) // 2 + x.size(1),
        ]
        return x, pos_emb


class OnnxStreamPositionalEncoding(torch.nn.Module):
    """Streaming Positional encoding.

    """
    def __init__(self, model, max_seq_len=5000):
        """Construct an PositionalEncoding object."""
        super(StreamPositionalEncoding, self).__init__()
        
        self.d_model = model.d_model
        self.xscale = model.xscale
        self.pe = None
        # Hold as attribute to export as config parameter,
        # in order to raise an error when start_idx + x.size(1)
        # exceeds the max_seq_len
        tmp = torch.tensor(0.0).expand(1, max_seq_len)
        self.extend_pe(tmp.size(1), tmp.device, tmp.dtype)
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, length, device, dtype):
        """Reset the positional encodings."""
        pe = torch.zeros(length, self.d_model)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, start_idx: int = 0):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        return x * self.xscale + self.pe[:, start_idx : start_idx + x.size(1)]
