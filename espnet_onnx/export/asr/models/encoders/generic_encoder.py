# This script is acuired from https://github.com/karthik19967829/espnet_onnx/blob/wrapper-class/espnet_onnx/export/asr/models/encoder_wrapper.py

import os

import numpy as np
import torch
import torch.nn as nn

from espnet_onnx.utils.abs_model import AbsExportModel
from espnet_onnx.export.asr.get_config import (
    get_frontend_config,
    get_norm_config
)
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder


class GenericEncoder(nn.Module, AbsExportModel):
    def __init__(
        self,
        model,
        frontend,
        preencoder=None,
        feats_dim=80,
        calculate_feats_len=True,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.model_name = 'encoder'
        self.frontend = frontend
        self.preencoder = preencoder
        self.feats_dim = feats_dim
        self.calculate_feats_len = calculate_feats_len
        # kwargs['max_seq_len'] = max_seq_len
        self.get_frontend(kwargs)

    def forward(self, feats, feats_length=None):
        if self.calculate_feats_len:
            feats_length = torch.ones(feats[:, :, 0].shape).sum(dim=-1).type(torch.long)
        else:
            assert feats_length is not None

        if self.preencoder is not None:
            feats, feats_length = self.preencoder(feats, feats_length)

        return self.model(feats, feats_length)

    def get_output_size(self):
        return self.model.encoders[0].size

    def get_frontend(self, kwargs):
        from espnet_onnx.export.asr.models import get_frontend_models
        self.frontend_model = get_frontend_models(self.frontend, kwargs)
        if self.frontend_model is not None:
            self.submodel = []
            self.submodel.append(self.frontend_model)
            self.feats_dim = self.frontend_model.output_dim

    def get_dummy_inputs(self):
        feats = torch.randn(1, 100, self.feats_dim)
        feats_length = torch.LongTensor([100])
        if self.calculate_feats_len:
            return (feats,)
        else:
            return (feats, feats_length)

    def get_input_names(self):
        if self.calculate_feats_len:
            return ['feats']
        else:
            return ['feats', 'feats_lens']

    def get_output_names(self):
        return ['encoder_out', 'encoder_out_lens']

    def get_dynamic_axes(self):
        return {
            'feats': {
                1: 'feats_length'
            }
        }

    def get_model_config(self, asr_model=None, path=None):
        ret = {}
        ret.update(
            enc_type='GenericEncoder',
            model_path=os.path.join(path, f'{self.model_name}.onnx'),
            is_vggrnn=isinstance(self.model, VGGRNNEncoder),
            frontend=get_frontend_config(asr_model.frontend, self.frontend_model, path=path),
            do_normalize=asr_model.normalize is not None,
            do_preencoder=asr_model.preencoder is not None,
            do_postencoder=asr_model.postencoder is not None
        )
        if ret['do_normalize']:
            ret.update(normalize=get_norm_config(asr_model.normalize, path))
        # Currently postencoder is not supported.
        # if ret['do_postencoder']:
        #     ret.update(postencoder=get_postenc_config(self.model.postencoder))
        return ret