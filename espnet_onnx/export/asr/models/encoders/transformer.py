import os

import torch
import torch.nn as nn
from espnet.nets.pytorch_backend.transformer.attention import \
    MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling, Conv2dSubsampling2, Conv2dSubsampling6,
    Conv2dSubsampling8)

from espnet_onnx.export.asr.models.encoder_layer import OnnxEncoderLayer
from espnet_onnx.export.asr.models.language_models.embed import Embedding
from espnet_onnx.export.asr.models.multihead_att import \
    OnnxMultiHeadedAttention
from espnet_onnx.utils.abs_model import AbsExportModel
from espnet_onnx.utils.torch_function import MakePadMask


class TransformerEncoder(nn.Module, AbsExportModel):
    def __init__(
        self, model, preencoder=None, max_seq_len=512, feats_dim=80, **kwargs
    ):
        super().__init__()
        self.embed = Embedding(model.embed, max_seq_len)
        self.model = model
        self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        self.feats_dim = feats_dim
        # replace multihead attention module into customized module.
        for i, d in enumerate(self.model.encoders):
            # d is EncoderLayer
            if isinstance(d.self_attn, MultiHeadedAttention):
                d.self_attn = OnnxMultiHeadedAttention(d.self_attn)
            self.model.encoders[i] = OnnxEncoderLayer(d)

        self.model_name = "xformer_encoder"
        self.num_heads = model.encoders[0].self_attn.h
        self.hidden_size = model.encoders[0].self_attn.linear_out.out_features
        self.preencoder = preencoder

    def prepare_mask(self, mask):
        if len(mask.shape) == 2:
            mask = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask = 1 - mask[:, None, :]

        return mask * -10000.0

    def forward(self, feats, feats_length):
        # compute preencoder
        if self.preencoder is not None:
            feats, feats_length = self.preencoder(feats, feats_length)

        mask = self.make_pad_mask(feats_length)
        if (
            isinstance(self.model.embed, Conv2dSubsampling)
            or isinstance(self.model.embed, Conv2dSubsampling2)
            or isinstance(self.model.embed, Conv2dSubsampling6)
            or isinstance(self.model.embed, Conv2dSubsampling8)
        ):
            xs_pad, mask = self.embed(feats, mask)
        else:
            xs_pad = self.embed(feats)

        mask = self.prepare_mask(mask)

        xs_pad, masks = self.model.encoders(xs_pad, mask)
        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.model.normalize_before:
            xs_pad = self.model.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, None
