# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 17:27:16 2021.
@author: Keqi Deng (UCAS)

Modified by Masao Someki
"""

from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.contextual_block_encoder_layer import (
    ContextualBlockEncoderLayer,  # noqa: H301
)
from espnet.nets.pytorch_backend.nets_utils import (
    make_pad_mask,  # noqa: H301
    get_activation,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import StreamPositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling_without_posenc import (
    Conv2dSubsamplingWOPosEnc,  # noqa: H301
)
from espnet2.asr.encoder.abs_encoder import AbsEncoder
import math
import torch
import torch.nn as nn
from typeguard import check_argument_types
from typing import (
    Optional,  # noqa: H301
    Tuple,  # noqa: H301
)
from ..abs_model import AbsModel


class ContextualBlockXformerEncoder(nn.Module, AbsModel):
    """Contextual Block Conformer encoder module.
    """
    def __init__(
        self,
        model
    ):
        super().__init__()
        self.model = model
        self._output_size = model._output_size
        self.pos_enc = model.pos_enc
        
        self.embed = model.embed
        self.subsample = model.subsample
        
        self.normalize_before = model.normalize_before
        self.encoders = model.encoders
        
        if self.normalize_before:
            self.after_norm = model.after_norm

        # for block processing
        self.block_size = model.block_size
        self.hop_size = model.hop_size
        self.look_ahead = model.look_ahead
        self.init_average = model.init_average
        self.ctx_pos_enc = model.ctx_pos_enc
        self.overlap_size = self.block_size - self.hop_size
        self.offset = self.block_size - self.look_ahead - self.hop_size
        
        self.mask_online = torch.zeros(1, 1, self.block_size + 2, self.block_size + 2)
        self.mask_online[:, :, 1:, :-1] = 1

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        buffer_before_downsampling: torch.Tensor,
        buffer_after_downsampling: torch.Tensor,
        prev_addin: torch.Tensor,
        n_processed_blocks: torch.Tensor,
        past_encoder_ctx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
                    L = subsample * (hop_size + 1)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        xs_pad = torch.cat([buffer_before_downsampling, xs_pad], dim=1)
        
        # note that n_samples is equal to hop_size
        n_samples = xs_pad.size(1) // self.subsample - 1
        n_res_samples = xs_pad.size(1) % self.subsample + self.subsample * 2
        
        buffer_before_downsampling = xs_pad[:, -n_res_samples:]
        xs_pad = xs_pad[:, :n_samples * self.subsample]

        xs_pad = self.compute_embed(xs_pad)

        # create empty output container
        # assume that buffer_after_downsampling.shape = (B, L, D),
        # where L = overlap_size
        xs_pad = torch.cat([buffer_after_downsampling, xs_pad], dim=1)

        # since xs_pad contains 1 block of frames, block_num should be always 1.
        block_num = 1
        # since xs_pad.size(1) = hop_size + overlap_size, and block_num = 1,
        # buffer_after_downsampling.size(1) = overlap_size
        buffer_after_downsampling = xs_pad[:, -self.overlap_size:]
        xs_pad = xs_pad[:, :block_num * self.hop_size + self.overlap_size]
        
        if self.init_average:
            addin = xs_pad.mean(1, keepdim=True)
        else:
            addin = xs_pad.max(1, keepdim=True)
        if self.ctx_pos_enc:
            addin = self.pos_enc(addin, n_processed_blocks)
        
        xs_pad = self.pos_enc(xs_pad, self.hop_size * n_processed_blocks)
        xs_chunk = torch.cat([prev_addin, xs_pad, addin], dim=1)

        ys_chunk, _, _, _, past_encoder_ctx, _, _ = self.encoders(
            xs_chunk.unsqueeze(1), self.mask_online, True, past_encoder_ctx
        )

        # remove addin
        ys_chunk = ys_chunk[:, 0, 1:-1]
        if n_processed_blocks == 0:
            ys_pad = ys_chunk[:, 0:self.offset + self.hop_size]
        else:
            ys_pad = ys_chunk[:, self.offet : self.offset + self.hop_size]

        if self.normalize_before:
            ys_pad = self.after_norm(ys_pad)

        return ys_pad, buffer_before_downsampling, buffer_after_downsampling, \
            addin, n_processed_blocks + 1, past_encoder_ctx
    
    def compute_embed(self, xs_pad):
        if isinstance(self.embed, Conv2dSubsamplingWOPosEnc):
            xs_pad, _ = self.embed(xs_pad, None)
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)
        return xs_pad

    def get_output_size(self):
        return self._output_size

    def get_dummy_inputs(self):
        # L = subsample * (hop_size + 1)
        n_feats = 80
        xs_pad = torch.randn(1, self.subsample * (self.hop_size + 1), n_feats)
        n_res_samples = xs_pad.size(1) % self.subsample + self.subsample * 2
        buffer_before_downsampling = torch.randn(1, n_res_samples, n_feats)
        o = self.compute_embed(xs_pad)
        buffer_after_downsampling = torch.randn(1, self.overlap_size, o.shape[-1])
        prev_addin = torch.randn(1, 1, o.shape[-1])
        n_processed_blocks = torch.LongTensor([0])
        past_encoder_ctx = torch.randn(1, len(self.encoders), self.encoders[0].size)
        return (xs_pad, buffer_before_downsampling, buffer_after_downsampling, prev_addin,
                n_processed_blocks, past_encoder_ctx)

    def get_input_names(self):
        return ['xs_pad', 'buffer_before_downsampling', 'buffer_after_downsampling',
                'prev_addin', 'n_processed_blocks', 'past_encoder_ctx']

    def get_output_names(self):
        return ['ys_pad', 'next_buffer_before_downsampling', 'next_buffer_after_downsampling',
                'next_addin', 'next_n_processed_blocks', 'next_encoder_ctx']

    def get_dynamic_axes(self):
        return None

    def get_model_config(self, asr_model=None, path=None):
        ret = {}
        ret.update(
            enc_type='ContextualXformerEncoder',
            model_path=os.path.join(path, 'encoder.onnx'),
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