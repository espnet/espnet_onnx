import os

import numpy as np
import torch
import torch.nn as nn

from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN

from espnet_onnx.utils.function import make_pad_mask
from espnet_onnx.export.asr.get_config import (
    get_frontend_config,
    get_norm_config
)
from ..language_models.lm import Embedding
from ..abs_model import AbsModel


class XformerEncoder(nn.Module, AbsModel):
    def __init__(self, model):
        super().__init__()
        self.embed = Embedding(model.embed)
        self.model = model

    def forward(self, feats, mask):
        if (
            isinstance(self.model.embed, Conv2dSubsampling)
            or isinstance(self.model.embed, Conv2dSubsampling2)
            or isinstance(self.model.embed, Conv2dSubsampling6)
            or isinstance(self.model.embed, Conv2dSubsampling8)
        ):
            xs_pad, mask = self.embed(feats, mask)
            xs_pad = xs_pad[0]
        else:
            xs_pad = self.embed(feats)

        xs_pad, masks = self.model.encoders(xs_pad, mask)
        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.model.normalize_before:
            xs_pad = self.model.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, None

    def get_output_size(self):
        return self.model.encoders[0].size

    def get_dummy_inputs(self):
        feats = torch.randn(1, 100, 80)
        mask = torch.from_numpy(make_pad_mask(
            np.array([feats.shape[1]]))[:, None, :])
        return (feats, mask)

    def get_input_names(self):
        return ['feats', 'mask']

    def get_output_names(self):
        return ['encoder_out', 'encoder_out_lens']

    def get_dynamic_axes(self):
        return {
            'feats': {
                1: 'feats_length'
            },
            'mask': {
                2: 'mask_length'
            }
        }

    def get_model_config(self, asr_model=None, path=None):
        ret = {}
        ret.update(
            enc_type='XformerEncoder',
            model_path=os.path.join(path, 'encoder.onnx'),
            is_vggrnn=False,
            frontend=get_frontend_config(asr_model.frontend),
            do_normalize=asr_model.normalize is not None,
            do_preencoder=asr_model.preencoder is not None,
            do_postencoder=asr_model.postencoder is not None
        )
        if ret['do_normalize']:
            ret.update(normalize=get_norm_config(
                asr_model.normalize, path))
        # Currently preencoder, postencoder is not supported.
        # if ret['do_preencoder']:
        #     ret.update(preencoder=get_preenc_config(self.model.preencoder))
        # if ret['do_postencoder']:
        #     ret.update(postencoder=get_postenc_config(self.model.postencoder))
        return ret
