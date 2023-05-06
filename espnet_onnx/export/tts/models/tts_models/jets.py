import logging
from typing import Optional

import torch
import torch.nn as nn

from espnet_onnx.export.tts.models.tts_models.fastspeech2 import \
    OnnxStyleEncoder
from espnet_onnx.export.tts.models.vocoders.hifigan import OnnxHiFiGANVocoder
from espnet_onnx.utils.abs_model import AbsExportModel
from espnet_onnx.utils.torch_function import MakePadMask, normalize


class OnnxGaussianUpsampling(nn.Module):
    def __init__(self, model):
        super(OnnxGaussianUpsampling, self).__init__()
        self.model = model
        self.delta = model.delta

    def __call__(self, hs, ds, d_masks=None):
        B = ds.size(0)

        if ds.sum() == 0:
            logging.warning(
                "predicted durations includes all 0 sequences. "
                "fill the first element with 1."
            )
            # NOTE(kan-bayashi): This case must not be happened in teacher forcing.
            #   It will be happened in inference with a bad duration predictor.
            #   So we do not need to care the padded sequence case here.
            ds[ds.sum(dim=1).eq(0)] = 1

        T_feats = ds.sum()
        t = torch.arange(0, T_feats).unsqueeze(0).repeat(B, 1)
        c = ds.cumsum(dim=-1) - ds / 2
        energy = -1 * self.delta * (t.unsqueeze(-1) - c.unsqueeze(1)) ** 2
        if d_masks is not None:
            energy = energy.masked_fill(
                1 - d_masks.unsqueeze(1).repeat(1, T_feats, 1), -10000.0
            )

        p_attn = torch.softmax(energy, dim=2)  # (B, T_feats, T_text)
        hs = torch.matmul(p_attn, hs)
        return hs


class OnnxJETSGenerator(nn.Module):
    def __init__(
        self,
        model,
        max_seq_len: int = None,
    ):
        super().__init__()
        # HPs
        self.use_gst = model.use_gst
        self.spks = model.spks
        self.langs = model.langs
        self.spk_embed_dim = model.spk_embed_dim
        self.max_seq_len = max_seq_len

        # models
        self.make_pad_mask = MakePadMask(max_seq_len)
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.pitch_predictor = model.pitch_predictor
        self.energy_predictor = model.energy_predictor
        self.duration_predictor = model.duration_predictor
        self.pitch_embed = model.pitch_embed
        self.energy_embed = model.energy_embed
        self.length_regulator = OnnxGaussianUpsampling(model.length_regulator)
        self.generator = OnnxHiFiGANVocoder(model.generator)
        if self.use_gst:
            self.gst = OnnxStyleEncoder(model.gst)
        if self.spks:
            self.sid_emb = model.sid_emb
        if self.langs:
            self.lid_emb = model.lid_emb
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = model.spk_embed_integration_type
            if self.spk_embed_integration_type == "add":
                self.projection = model.projection

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ):
        x_masks = self._source_mask(text_lengths)
        hs, _ = self.encoder(text, x_masks)  # (B, T_text, adim)

        # integrate with GST
        if self.use_gst:
            style_embs = self.gst(feats)
            hs = hs + style_embs.unsqueeze(1)

        # integrate with SID and LID embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        h_masks = self.make_pad_mask(text_lengths)
        # forward duration predictor and variance predictors
        p_outs = self.pitch_predictor(hs, h_masks.unsqueeze(-1))
        e_outs = self.energy_predictor(hs, h_masks.unsqueeze(-1))
        d_outs = self.duration_predictor.inference(hs, h_masks)

        p_embs = self.pitch_embed(p_outs.transpose(1, 2)).transpose(1, 2)
        e_embs = self.energy_embed(e_outs.transpose(1, 2)).transpose(1, 2)
        hs = hs + e_embs + p_embs

        # upsampling
        d_masks = 1 - self.make_pad_mask(text_lengths)
        hs = self.length_regulator(hs, d_outs, d_masks)  # (B, T_feats, adim)

        # forward decoder
        zs, _ = self.decoder(hs, None)  # (B, T_feats, adim)

        # forward generator
        wav = self.generator(zs)
        return wav.squeeze(1), d_outs

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        x_masks = 1 - self.make_pad_mask(ilens)
        return x_masks.unsqueeze(-2)

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.projection(torch.cat([hs, spembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs


class OnnxJETSModel(nn.Module, AbsExportModel):
    def __init__(self, model, max_seq_len: int = 512, **kwargs):
        super().__init__()
        self.model = model
        self.generator = OnnxJETSGenerator(model.generator, max_seq_len)
        self.model_name = "jets"

    def forward(
        self,
        text: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ):
        # setup
        text = text[None]
        text_lengths = torch.ones(text.shape).sum(dim=-1).type(torch.long)

        wav, dur = self.generator(
            text=text,
            text_lengths=text_lengths,
            sids=sids,
            spembs=spembs,
            lids=lids,
        )
        return dict(wav=wav.view(-1), duration=dur[0])

    def get_dummy_inputs(self):
        text = torch.LongTensor([0, 1])

        sids = torch.LongTensor([0]) if self.model.generator.spks is not None else None

        spembs = (
            torch.randn(self.model.generator.spk_embed_dim)
            if self.model.generator.spk_embed_dim is not None
            else None
        )

        lids = torch.LongTensor([0]) if self.model.generator.langs is not None else None

        return (text, sids, spembs, lids)

    def get_input_names(self):
        ret = ["text"]
        if self.model.generator.spks is not None:
            ret.append("sids")
        if self.model.generator.spk_embed_dim is not None:
            ret.append("spembs")
        if self.model.generator.langs is not None:
            ret.append("lids")
        return ret

    def get_output_names(self):
        return ["wav", "dur"]

    def get_dynamic_axes(self):
        return {"text": {0: "text_length"}}

    def get_model_config(self, path):
        return {
            "model_type": "JETS",
            "model_path": str(path / f"{self.model_name}.onnx"),
        }
