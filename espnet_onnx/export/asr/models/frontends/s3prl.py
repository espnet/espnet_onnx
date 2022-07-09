import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8
)
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention

from espnet_onnx.utils.torch_function import MakePadMask
from ..language_models.embed import Embedding
from ..conformer_layer import OnnxConformerLayer
from ..multihead_att import OnnxMultiHeadedAttention
from espnet_onnx.utils.abs_model import AbsExportModel


class Featurizer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.feature_selection = model.feature_selection
        self.normalize = model.normalize
        self.layer_num = model.layer_num
        self.weights = model.weights
        self.downsample_rate = model.downsample_rate
    
    def forward(self, feats, wav):
        # wav: (B, T)
        if self.normalize:
            feats = F.layer_norm(
                feats, (feats.shape[-1],))

        # _, *origin_shape = feats.shape
        # feats = feats.view(self.layer_num, -1)
        # norm_weights = F.softmax(self.weights, dim=-1)
        # weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        # weighted_feature = weighted_feature.view(*origin_shape)

        feat_len = round(len(wav[0]) / self.downsample_rate)
        return feats[:feat_len]


class UpstreamExpert(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.task = model.task
        self.extract_features = model.model.extract_features
        
    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wav, wav_length):
        if self.task.cfg.normalize:
            wav = F.layer_norm(wav, wav.shape)

        wav_padding_mask = ~torch.lt(
            torch.arange(wav_length[0]).unsqueeze(0),
            wav_length.unsqueeze(1),
        )
        features, feat_padding_mask = self.extract_features(
            wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )
        return features


class S3PRLModel(nn.Module, AbsExportModel):
    def __init__(
        self,
        model,
        **kwargs
    ):
        super().__init__()
        self.upstream = UpstreamExpert(model.upstream)
        # self.upstream = model.upstream
        self.featurizer = Featurizer(model.featurizer)
        self.model_name = 'hubert_frontend'

    def forward(self, input, input_length):
        feats = self.upstream(input, input_length)
        # feats = self.upstream(input)
        feats = self.featurizer(feats, input)
        return feats, input_length

    def is_optimizable(self):
        return False

    def get_dummy_inputs(self):
        wav = torch.randn(1, 1000)
        wav_length = torch.LongTensor([1000])
        return (wav, wav_length)

    def get_input_names(self):
        return ['wav', 'wav_length']

    def get_output_names(self):
        return ['feats', 'feats_lens']

    def get_dynamic_axes(self):
        return {
            'feats': {
                1: 'feats_length'
            }
        }

    def get_model_config(self, path=None):
        ret = {}
        ret.update(
            frontend_type='hubert',
            model_path=os.path.join(path, f'{self.model_name}.onnx'),
        )
        return ret
