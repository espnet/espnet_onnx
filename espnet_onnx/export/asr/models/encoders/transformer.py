import os

import torch
import torch.nn as nn
from espnet.nets.pytorch_backend.transformer.attention import \
    MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling, Conv2dSubsampling2, Conv2dSubsampling6,
    Conv2dSubsampling8)

from espnet_onnx.export.asr.get_config import (get_frontend_config,
                                               get_norm_config)
from espnet_onnx.export.asr.models.encoder_layer import OnnxEncoderLayer
from espnet_onnx.export.asr.models.language_models.embed import Embedding
from espnet_onnx.export.asr.models.multihead_att import \
    OnnxMultiHeadedAttention
from espnet_onnx.utils.abs_model import AbsExportModel
from espnet_onnx.utils.torch_function import MakePadMask


class TransformerEncoder(nn.Module, AbsExportModel):
    def __init__(
        self, model, frontend, preencoder=None, max_seq_len=512, feats_dim=80, **kwargs
    ):
        super().__init__()
        self.embed = Embedding(model.embed, max_seq_len)
        self.model = model
        self.frontend = frontend
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
        self.get_frontend(kwargs)
        self.preencoder = preencoder

    def get_frontend(self, kwargs):
        from espnet_onnx.export.asr.models import get_frontend_models

        self.frontend_model = get_frontend_models(self.frontend, kwargs)
        if self.frontend_model is not None:
            self.submodel = []
            self.submodel.append(self.frontend_model)
            self.feats_dim = self.frontend_model.output_dim

    def prepare_mask(self, mask):
        if len(mask.shape) == 2:
            mask = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask = 1 - mask[:, None, :]

        return mask * -10000.0

    def forward(self, feats):
        feats_length = torch.ones(feats[:, :, 0].shape).sum(dim=-1).type(torch.long)
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

    def get_output_size(self):
        return self.model.encoders[0].size

    def is_optimizable(self):
        return True

    def get_dummy_inputs(self):
        feats = torch.randn(1, 100, self.feats_dim)
        return feats

    def get_input_names(self):
        return ["feats"]

    def get_output_names(self):
        return ["encoder_out", "encoder_out_lens"]

    def get_dynamic_axes(self):
        return {"feats": {1: "feats_length"}, "encoder_out": {1: "enc_out_length"}}

    def get_model_config(self, asr_model=None, path=None):
        ret = {}
        ret.update(
            enc_type="XformerEncoder",
            model_path=os.path.join(path, f"{self.model_name}.onnx"),
            is_vggrnn=False,
            frontend=get_frontend_config(
                asr_model.frontend, self.frontend_model, path=path
            ),
            do_normalize=asr_model.normalize is not None,
            do_preencoder=asr_model.preencoder is not None,
            do_postencoder=asr_model.postencoder is not None,
        )
        if ret["do_normalize"]:
            ret.update(normalize=get_norm_config(asr_model.normalize, path))
        # Currently preencoder, postencoder is not supported.
        # if ret['do_postencoder']:
        #     ret.update(postencoder=get_postenc_config(self.model.postencoder))
        return ret
