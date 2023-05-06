import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet_onnx.export.asr.models.encoder_layer import OnnxEncoderLayer
from espnet_onnx.export.asr.models.multihead_att import \
    OnnxMultiHeadedAttention
from espnet_onnx.utils.abs_model import AbsExportModel
from espnet_onnx.utils.torch_function import MakePadMask


class Featurizer(nn.Module):
    def __init__(self, model):
        super().__init__()
        if hasattr(model, "feature_selection"):
            self.feature_selection = model.feature_selection
        else:
            self.feature_selection = "hidden_states"

        self.normalize = model.normalize
        if hasattr(model, "output_dim"):
            self.output_dim = model.output_dim
        else:
            self.output_dim = model._output_size
        self.downsample_rate = model.downsample_rate

        if self.feature_selection == "hidden_states":
            self.weights = model.weights
            if hasattr(model, "layer_num"):
                self.layer_num = model.layer_num
            else:
                self.layer_num = len(model.layer_selections)

    def _select_feature(self, feats):
        # if self.feature_selection == 'hidden_states':
        if self.feature_selection == "last_hidden_state":
            feats = feats[-1]
        elif "hidden_state_" in self.feature_selection:
            fs = self.feature_selection.split("_")
            assert len(fs) == 3
            feats = feats[int(fs[2])]
        return feats

    def forward(self, feats):
        feats = self._select_feature(feats)

        if len(feats) > 1:
            # wav: (B, T)
            if self.normalize:
                feats = F.layer_norm(feats, (feats.shape[-1],))

            _, *origin_shape = feats.shape
            feats = feats.view(self.layer_num, -1)
            norm_weights = F.softmax(self.weights, dim=-1)
            weighted_feature = (norm_weights.unsqueeze(-1) * feats).sum(dim=0)
            weighted_feature = weighted_feature.view(*origin_shape)
            return weighted_feature

        return feats


class HubertModel(nn.Module):
    def __init__(self, model, max_seq_len=512, **kwargs):
        super().__init__()
        if hasattr(model, "upstream") is not None:
            self.model = model.upstream
        else:
            self.model = model.model

        if hasattr(self.model, "task"):
            self.task_cfg = self.model.task.cfg
        elif hasattr(self.model, "task_cfg"):
            self.task_cfg = self.model.task_cfg
        else:
            raise RuntimeError(
                "S3PRL version is not supported. Please install v0.3 or v0.4"
            )

        self.encoder = self.model.model.encoder
        self.layers = nn.ModuleList([])
        self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        self.downsample_rate = self.model.get_downsample_rates("")
        for l in self.encoder.layers:
            _l = OnnxEncoderLayer(l, model_type="hubert")
            _l.self_attn = OnnxMultiHeadedAttention(_l.self_attn, model_type="hubert")
            self.layers.append(_l)
        self.eps = 1e-5

    def prepare_mask(self, mask):
        if len(mask.shape) == 2:
            mask = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask = 1 - mask[:, None, :]
        return mask * -10000.0

    def layer_norm(self, wav):
        m = torch.mean(wav, dim=-1, keepdim=True)
        v = torch.std(wav, dim=-1, keepdim=True)
        return (wav - m) / torch.sqrt(v + self.eps)

    def forward(self, wav):
        if self.task_cfg.normalize:
            wav = self.layer_norm(wav)

        # compute extract feature
        features = self.model.model.forward_features(wav)
        features = features.transpose(1, 2)
        features = self.model.model.layer_norm(features)
        if self.model.model.post_extract_proj is not None:
            features = self.model.model.post_extract_proj(features)

        # residual pos_conv
        res = features
        features = self.encoder.pos_conv(features.transpose(1, 2))
        features = features.transpose(1, 2)
        features = features + res

        if not self.encoder.layer_norm_first:
            features = self.encoder.layer_norm(features)

        feats_length = torch.ones(features[:, :, 0].shape).sum(dim=-1).type(torch.long)
        mask = self.make_pad_mask(feats_length)
        mask = self.prepare_mask(mask)
        hidden_states = []

        for l in self.layers:
            hidden_states.append(features)
            features, mask = l(features, mask)

        if self.encoder.layer_norm_first:
            features = self.encoder.layer_norm(features)

        hidden_states.append(features)
        return torch.stack(hidden_states, dim=0), feats_length


class S3PRLModel(nn.Module, AbsExportModel):
    def __init__(self, model, max_seq_len=512, **kwargs):
        super().__init__()
        # Currently only HuBERT is supported
        self.upstream = HubertModel(model.upstream, max_seq_len)
        self.featurizer = Featurizer(model.featurizer)
        self.model_name = "hubert_frontend"
        self.output_dim = self.featurizer.output_dim
        self.num_heads = self.upstream.layers[0].self_attn.h
        self.hidden_size = self.upstream.layers[0].self_attn.all_head_size

    def forward(self, inputs):
        feats, feat_length = self.upstream(inputs)
        feats = self.featurizer(feats)
        return feats, feat_length

    def is_optimizable(self):
        return True

    def require_onnx(self):
        return True

    def get_dummy_inputs(self):
        wav = torch.randn(1, 16000)
        return wav

    def get_input_names(self):
        return ["wav"]

    def get_output_names(self):
        return ["feats", "feats_lens"]

    def get_dynamic_axes(self):
        return {"wav": {0: "wav_batch", 1: "wav_length"}}

    def get_model_config(self, path=None):
        ret = {}
        ret.update(
            frontend_type="hubert",
            output_dim=self.output_dim,
            model_path=os.path.join(path, f"{self.model_name}.onnx"),
        )
        return ret
