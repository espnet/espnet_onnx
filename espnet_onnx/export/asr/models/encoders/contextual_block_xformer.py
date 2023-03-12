import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from espnet.nets.pytorch_backend.transformer.attention import \
    MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.subsampling_without_posenc import \
    Conv2dSubsamplingWOPosEnc

from espnet_onnx.export.asr.get_config import (get_frontend_config,
                                               get_norm_config)
from espnet_onnx.export.asr.models.multihead_att import \
    OnnxMultiHeadedAttention
from espnet_onnx.utils.abs_model import AbsExportModel


class ContextualBlockXformerEncoder(nn.Module, AbsExportModel):
    """Contextual Block Conformer encoder module."""

    def __init__(self, model, feats_dim=80, **kwargs):
        super().__init__()
        self.model = model
        self.model_name = "xformer_encoder"
        self._output_size = model._output_size
        self.pos_enc = model.pos_enc

        self.embed = model.embed
        self.subsample = model.subsample

        self.normalize_before = model.normalize_before
        self.encoders = model.encoders
        for i, d in enumerate(self.encoders):
            # d is EncoderLayer
            if isinstance(d.self_attn, MultiHeadedAttention):
                d.self_attn = OnnxMultiHeadedAttention(d.self_attn)

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
        self.xscale = model.pos_enc.xscale

        # for export configuration
        self.feats_dim = feats_dim

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        mask: torch.Tensor,
        buffer_before_downsampling: torch.Tensor,
        buffer_after_downsampling: torch.Tensor,
        prev_addin: torch.Tensor,
        pos_enc_xs: torch.Tensor,
        pos_enc_addin: torch.Tensor,
        past_encoder_ctx: torch.Tensor,
        indicies: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        The sum of the length xs_pad:L_1 and of buffer_before_downsampling: L_2
        should be the multiple of subsample.
        L_1 + L_2 = alpha * self.subsample

        Args:
            xs_pad: (1, hop_size*subsample, D)
            buffer_after_downsampling: (1, overlap_size=block_size-hop_size, D)
            indicies: torch.Tensor. [offset, offset + hop_length]
            mask: zeros(1, 1, self.block_size + 2, self.block_size + 2)
            pos_enc_xs: (B, L, D) L = block_size
        """
        # compute preencoder
        xs_pad = torch.cat([buffer_before_downsampling, xs_pad], dim=1)
        n_samples = xs_pad.size(1) // self.subsample - 1

        # [ TODO ] Need to add slice operation to handle the first iteration
        # which does not require buffer_before_downsampling

        n_res_samples = xs_pad.size(1) % self.subsample + self.subsample * 2
        buffer_before_downsampling = xs_pad[:, -n_res_samples:]  # (B, L, overlap)
        xs_pad = xs_pad[:, : n_samples * self.subsample]  # (B, L, overlap)

        xs_pad = self.compute_embed(xs_pad)
        xs_pad = torch.cat([buffer_after_downsampling, xs_pad], dim=1)

        overlap_size = self.block_size - self.hop_size
        block_num = max(0, xs_pad.size(1) - overlap_size) // self.hop_size
        res_frame_num = xs_pad.size(1) - self.hop_size * block_num
        buffer_after_downsampling = xs_pad[:, -res_frame_num:]

        if self.init_average:
            addin = xs_pad.mean(1, keepdim=True)
        else:
            addin = xs_pad.max(1, keepdim=True)

        if self.ctx_pos_enc:
            addin = addin * self.xscale + pos_enc_addin

        xs_pad = xs_pad[:, : self.block_size] * self.xscale + pos_enc_xs
        xs_chunk = torch.cat([prev_addin, xs_pad, addin], dim=1)

        ys_chunk, _, _, _, past_encoder_ctx, _, _ = self.encoders(
            xs_chunk.unsqueeze(1), mask, True, past_encoder_ctx
        )

        # remove addin
        ys_chunk = ys_chunk.squeeze(1)[:, 1:-1]
        ys_pad = ys_chunk[:, indicies[0] : indicies[1]]

        if self.normalize_before:
            ys_pad = self.after_norm(ys_pad)

        return (
            ys_pad,
            buffer_before_downsampling,
            buffer_after_downsampling,
            addin,
            past_encoder_ctx,
        )

    def compute_embed(self, xs_pad):
        if isinstance(self.embed, Conv2dSubsamplingWOPosEnc):
            xs_pad, _ = self.embed(xs_pad, None)
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)
        return xs_pad

    def get_output_size(self):
        return self._output_size

    def get_dummy_inputs(self):
        n_feats = self.feats_dim
        xs_pad = torch.randn(1, self.hop_size * self.subsample, n_feats)
        mask = torch.ones(1, 1, self.block_size + 2, self.block_size + 2)
        o = self.compute_embed(xs_pad)
        buffer_before_downsampling = torch.randn(1, self.subsample * 2, n_feats)
        buffer_after_downsampling = torch.randn(1, self.overlap_size, o.shape[-1])
        prev_addin = torch.randn(1, 1, o.shape[-1])
        pos_enc_xs = torch.randn(1, self.block_size, o.shape[-1])
        pos_enc_addin = torch.randn(1, 1, o.shape[-1])
        past_encoder_ctx = torch.randn(1, len(self.encoders), self.encoders[0].size)
        indicies = torch.LongTensor([8, 24])
        return (
            xs_pad,
            mask,
            buffer_before_downsampling,
            buffer_after_downsampling,
            prev_addin,
            pos_enc_xs,
            pos_enc_addin,
            past_encoder_ctx,
            indicies,
        )

    def get_input_names(self):
        return [
            "xs_pad",
            "mask",
            "buffer_before_downsampling",
            "buffer_after_downsampling",
            "prev_addin",
            "pos_enc_xs",
            "pos_enc_addin",
            "past_encoder_ctx",
            "indicies",
        ]

    def get_output_names(self):
        return [
            "ys_pad",
            "next_buffer_before_downsampling",
            "next_buffer_after_downsampling",
            "next_addin",
            "next_encoder_ctx",
        ]

    def get_dynamic_axes(self):
        return {
            "xs_pad": {1: "xs_pad_length"},
            "mask": {2: "block_height", 3: "block_width"},
            "buffer_before_downsampling": {1: "bbd_length"},
            "buffer_after_downsampling": {1: "bad_length"},
            "pos_enc_xs": {1: "pex_length"},
            "ys_pad": {1: "ys_pad_length"},
            "next_buffer_before_downsampling": {1: "nbbd_length"},
            "next_buffer_after_downsampling": {1: "nbad_length"},
        }

    def get_model_config(self, asr_model=None, path=None):
        ret = {}
        ret.update(
            enc_type="ContextualXformerEncoder",
            model_path=os.path.join(path, f"{self.model_name}.onnx"),
            frontend=get_frontend_config(asr_model.frontend),
            do_normalize=asr_model.normalize is not None,
            do_postencoder=asr_model.postencoder is not None,
        )
        if ret["do_normalize"]:
            ret.update(normalize=get_norm_config(asr_model.normalize, path))
        # streaming config
        ret.update(
            pe_path=str(path.parent / "pe.npy"),
            n_layers=len(asr_model.encoder.encoders),
            subsample=self.subsample,
        )
        # Currently preencoder, postencoder is not supported.
        # if ret['do_postencoder']:
        #     ret.update(postencoder=get_postenc_config(self.model.postencoder))
        return ret
