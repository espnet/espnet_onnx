import os

import numpy as np
import torch
import torch.nn as nn

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


class ConformerEncoder(nn.Module, AbsExportModel):
    def __init__(
        self,
        model,
        max_seq_len=512,
        feats_dim=80, 
        ctc=None, 
        **kwargs
    ):
        super().__init__()
        self.embed = Embedding(model.embed, max_seq_len)
        self.model = model
        self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        self.feats_dim = feats_dim
        
        # replace multihead attention module into customized module.
        for i, d in enumerate(self.model.encoders):
            # d is EncoderLayer
            # Conformer optimization is currently not supported.
            # if isinstance(d.self_attn, MultiHeadedAttention):
            #     d.self_attn = OnnxMultiHeadedAttention(d.self_attn)
            self.model.encoders[i] = OnnxConformerLayer(d)
        
        self.model_name = 'xformer_encoder'
        self.num_heads = model.encoders[0].model.self_attn.h
        self.hidden_size = model.encoders[0].model.self_attn.linear_out.out_features
        
        if self.model.interctc_use_conditioning:
            assert ctc is not None, 'You have to specify "ctc" in export_config to use interctc'
            self.ctc = ctc
    
    def prepare_mask(self, mask):
        if len(mask.shape) == 2:
            mask = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask = 1 - mask[:, None, :]
        
        return mask * -10000.0

    def forward(self, feats):
        feats_length = torch.ones(feats[:, :, 0].shape).sum(dim=-1).type(torch.long)
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

        # mask = self.prepare_mask(mask)
        
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

    def get_output_size(self):
        return self.model.encoders[0].size

    def is_optimizable(self):
        return True

    def get_dummy_inputs(self):
        feats = torch.randn(1, 100, self.feats_dim)
        return (feats)

    def get_input_names(self):
        return ['feats']

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
        if isinstance(frontend, DefaultFrontend):
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
        else:
            return {}

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
