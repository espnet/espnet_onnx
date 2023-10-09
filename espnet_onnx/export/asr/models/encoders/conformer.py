import os

import torch
import torch.nn as nn
from espnet.nets.pytorch_backend.transformer.attention import (
    LegacyRelPositionMultiHeadedAttention, MultiHeadedAttention,
    RelPositionMultiHeadedAttention)
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling, Conv2dSubsampling2, Conv2dSubsampling6,
    Conv2dSubsampling8)

from espnet_onnx.export.asr.get_config import (get_frontend_config,
                                               get_norm_config)
from espnet_onnx.export.asr.models.conformer_layer import OnnxConformerLayer
from espnet_onnx.export.asr.models.language_models.embed import Embedding
from espnet_onnx.export.asr.models.multihead_att import (
    OnnxMultiHeadedAttention, OnnxRelPosMultiHeadedAttention)
from espnet_onnx.utils.abs_model import AbsExportModel
from espnet_onnx.utils.torch_function import MakePadMask


class ConformerEncoder(nn.Module, AbsExportModel):
    def __init__(
        self,
        model,
        preencoder=None,
        max_seq_len=512,
        feats_dim=80,
        ctc=None,
        **kwargs,
    ):
        super().__init__()
        self.embed = Embedding(model.embed, max_seq_len)
        self.model = model
        self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        self.feats_dim = feats_dim
        kwargs["max_seq_len"] = max_seq_len

        # replace multihead attention module into customized module.
        for i, d in enumerate(self.model.encoders):
            # d is EncoderLayer
            if isinstance(d.self_attn, LegacyRelPositionMultiHeadedAttention):
                d.self_attn = OnnxRelPosMultiHeadedAttention(
                    d.self_attn, is_legacy=True
                )
            elif isinstance(d.self_attn, RelPositionMultiHeadedAttention):
                d.self_attn = OnnxRelPosMultiHeadedAttention(
                    d.self_attn, is_legacy=False
                )
            elif isinstance(d.self_attn, MultiHeadedAttention):
                d.self_attn = OnnxMultiHeadedAttention(d.self_attn)
            self.model.encoders[i] = OnnxConformerLayer(d)

        self.model_name = "xformer_encoder"
        self.num_heads = model.encoders[0].model.self_attn.h
        self.hidden_size = model.encoders[0].model.self_attn.linear_out.out_features

        if self.model.interctc_use_conditioning:
            assert (
                ctc is not None
            ), 'You have to specify "ctc" in export_config to use interctc'
            self.ctc = ctc
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

        intermediate_outs = []
        if len(self.model.interctc_layer_idx) == 0:
            xs_pad, mask = self.model.encoders(xs_pad, mask)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, mask = encoder_layer(xs_pad, mask)

                if layer_idx + 1 in self.model.interctc_layer_idx:
                    encoder_out = xs_pad
                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]

                    # intermediate outputs are also normalized
                    if self.model.normalize_before:
                        encoder_out = self.model.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.model.interctc_use_conditioning:
                        ctc_out = self.ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            x, pos_emb = xs_pad
                            x = x + self.model.conditioning_layer(ctc_out)
                            xs_pad = (x, pos_emb)
                        else:
                            xs_pad = xs_pad + self.model.conditioning_layer(ctc_out)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.model.normalize_before:
            xs_pad = self.model.after_norm(xs_pad)

        olens = mask.squeeze(1).sum(1)
        return xs_pad, olens
