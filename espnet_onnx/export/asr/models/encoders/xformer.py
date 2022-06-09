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

from espnet_onnx.utils.torch_function import MakePadMask
from ..language_models.embed import Embedding
from espnet_onnx.utils.abs_model import AbsExportModel


class XformerEncoder(nn.Module, AbsExportModel):
    def __init__(
        self,
        model,
        max_seq_len=512,
        feats_dim=80, 
        **kwargs
    ):
        super().__init__()
        self.embed = Embedding(model.embed, max_seq_len)
        self.model = model
        self.make_pad_mask = MakePadMask(max_seq_len)
        self.feats_dim = feats_dim
        self.model_name = 'xformer_encoder'

    def forward(self, feats, feats_length):
        mask = 1 - self.make_pad_mask(feats_length).unsqueeze(1)
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
        feats = torch.randn(1, 100, self.feats_dim)
        feats_lengths = torch.LongTensor([feats.size(1)])
        return (feats, feats_lengths)

    def get_input_names(self):
        return ['feats', 'feats_length']

    def get_output_names(self):
        return ['encoder_out', 'encoder_out_lens']

    def get_dynamic_axes(self):
        return {
            'feats': {
                1: 'feats_length'
            },
            'encoder_out': {
                1: 'enc_out_length'
            }
        }

    def get_model_config(self, asr_model=None, path=None):
        ret = {}
        ret.update(
            enc_type='XformerEncoder',
            model_path=os.path.join(path, f'{self.model_name}.onnx'),
            is_vggrnn=False,
            frontend=self.get_frontend_config(asr_model.frontend),
            do_normalize=asr_model.normalize is not None,
            do_preencoder=asr_model.preencoder is not None,
            do_postencoder=asr_model.postencoder is not None
        )
        if ret['do_normalize']:
            ret.update(normalize=self.get_norm_config(
                asr_model.normalize, path))
        # Currently preencoder, postencoder is not supported.
        # if ret['do_preencoder']:
        #     ret.update(preencoder=get_preenc_config(self.model.preencoder))
        # if ret['do_postencoder']:
        #     ret.update(postencoder=get_postenc_config(self.model.postencoder))
        return ret

    def get_frontend_config(self, frontend):
        # currently only default config is supported.
        assert isinstance(
            frontend, DefaultFrontend), 'Currently only DefaultFrontend is supported.'

        stft_config = dict(
            n_fft=frontend.stft.n_fft,
            win_length=frontend.stft.win_length,
            hop_length=frontend.stft.hop_length,
            window=frontend.stft.window,
            center=frontend.stft.center,
            onesided=frontend.stft.onesided,
            normalized=frontend.stft.normalized,
        )
        logmel_config = frontend.logmel.mel_options
        logmel_config.update(log_base=frontend.logmel.log_base)
        return {
            "stft": stft_config,
            "logmel": logmel_config
        }

    def get_norm_config(self, normalize, path):
        if isinstance(normalize, GlobalMVN):
            return {
                "type": "gmvn",
                "norm_means": normalize.norm_means,
                "norm_vars": normalize.norm_vars,
                "eps": normalize.eps,
                "stats_file": str(path.parent / 'feats_stats.npz')
            }
        elif isinstance(normalize, UtteranceMVN):
            return {
                "type": "utterance_mvn",
                "norm_means": normalize.norm_means,
                "norm_vars": normalize.norm_vars,
                "eps": normalize.eps,
            }
