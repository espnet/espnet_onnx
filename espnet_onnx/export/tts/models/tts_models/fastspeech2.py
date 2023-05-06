from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet_onnx.export.asr.models.language_models.embed import Embedding
from espnet_onnx.utils.abs_model import AbsExportModel
from espnet_onnx.utils.torch_function import MakePadMask, normalize


class OnnxLengthRegurator(nn.Module):
    def __init__(self, alpha=1.0, max_seq_len=512):
        super().__init__()
        self.alpha = alpha
        # The maximum length of the make_pad_mask is the
        # maximum value of the duration.
        self.make_pad_mask = MakePadMask(max_seq_len)

    def forward(self, x, dur):
        # This class assumes that the batch size of x
        # should be 1.
        if self.alpha != 1.0:
            dur = torch.round(ds.float() * alpha).long()
        dur = dur[0]
        dm = 1 - self.make_pad_mask(dur)
        nz = torch.nonzero(dm)[:, 0]
        return x[:, nz]


class OnnxReferenceEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.convs = model.convs
        self.gru = model.gru

    def forward(self, speech):
        xs = speech.unsqueeze(1)
        hs = self.convs(xs).transpose(1, 2)
        new_shape = (hs.size(0), hs.size(1), -1)
        hs = hs.contiguous().view(new_shape)
        _, ref_embs = self.gru(hs)
        return ref_embs[-1]


class OnnxStyleEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ref_enc = OnnxReferenceEncoder(model.ref_enc)
        self.stl = model.stl

    def forward(self, speech):
        ref_embs = self.ref_enc(speech)
        style_embs = self.stl(ref_embs)
        return style_embs


class OnnxFastSpeech2(nn.Module, AbsExportModel):
    def __init__(
        self,
        model,
        max_seq_len: int = 512,
        alpha: float = 1.0,
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__()
        # HPs
        self.odim = model.odim
        self.use_gst = model.use_gst
        self.spks = model.spks
        self.langs = model.langs
        self.spk_embed_dim = model.spk_embed_dim
        self.eos = model.eos
        self.model_name = "fast_speech2"
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = model.spk_embed_integration_type
            if self.spk_embed_integration_type == "add":
                self.projection = model.projection

        # models
        self.make_pad_mask = MakePadMask(max_seq_len)
        self.encoder = model.encoder
        self.length_regulator = OnnxLengthRegurator(alpha, max_seq_len)
        self.pitch_predictor = model.pitch_predictor
        self.energy_predictor = model.energy_predictor
        self.pitch_embed = model.pitch_embed
        self.energy_embed = model.energy_embed
        self.duration_predictor = model.duration_predictor
        self.decoder = model.decoder
        self.feat_out = model.feat_out
        self.postnet = model.postnet
        if self.use_gst:
            self.gst = OnnxStyleEncoder(model.gst)
        if self.spks:
            self.sid_emb = model.sid_emb
        if self.langs:
            self.lid_emb = model.lid_emb
        self.encoder.embed = Embedding(self.encoder.embed, max_seq_len=max_seq_len)
        self.decoder.embed = Embedding(
            self.decoder.embed, max_seq_len=max_seq_len, use_cache=use_cache
        )

    def _source_mask(self, ilens):
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

    def forward(
        self,
        text: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        spembs: torch.Tensor = None,
        lids: Optional[torch.Tensor] = None,
    ):
        x, y = text, feats
        spemb = spembs

        # add eos at the last of sequence
        x = F.pad(x, [0, 1], "constant", self.eos)
        text_lengths = torch.ones(x.shape).sum(dim=-1).type(torch.long)

        # setup batch axis
        xs, ys = x.unsqueeze(0), None
        if y is not None:
            ys = y.unsqueeze(0)
        if spemb is not None:
            spembs = spemb.unsqueeze(0)

        _, outs, d_outs, p_outs, e_outs = self._forward(
            xs,
            text_lengths,
            ys,
            spembs=spembs,
            sids=sids,
            lids=lids,
        )  # (1, T_feats, odim)

        return dict(
            feat_gen=outs[0],
            duration=d_outs[0],
            pitch=p_outs[0],
            energy=e_outs[0],
        )

    def _forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        ys: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ):
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)  # (B, T_text, adim)

        # integrate with GST
        if self.use_gst:
            style_embs = self.gst(ys)
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

        # forward duration predictor and variance predictors
        d_masks = self.make_pad_mask(ilens)

        p_outs = self.pitch_predictor(hs, d_masks.unsqueeze(-1))
        e_outs = self.energy_predictor(hs, d_masks.unsqueeze(-1))

        d_outs = self.duration_predictor.inference(hs, d_masks)  # (B, T_text)
        # use prediction in inference
        p_embs = self.pitch_embed(p_outs.transpose(1, 2)).transpose(1, 2)
        e_embs = self.energy_embed(e_outs.transpose(1, 2)).transpose(1, 2)
        hs = hs + e_embs + p_embs
        hs = self.length_regulator(hs, d_outs)  # (B, T_feats, adim)

        # forward decoder
        h_masks = None
        zs, _ = self.decoder(hs, h_masks)  # (B, T_feats, adim)
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )  # (B, T_feats, odim)

        # postnet -> (B, T_feats//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        return before_outs, after_outs, d_outs, p_outs, e_outs

    def get_dummy_inputs(self):
        text = torch.LongTensor([0, 1])
        feats = torch.randn(10, self.odim) if self.use_gst else None

        sids = torch.LongTensor([0]) if self.spks is not None else None

        spembs = (
            torch.randn(self.spk_embed_dim) if self.spk_embed_dim is not None else None
        )

        lids = torch.LongTensor([0]) if self.langs is not None else None

        return (text, feats, sids, spembs, lids)

    def get_input_names(self):
        ret = ["text"]
        if self.use_gst:
            ret.append("feats")
        if self.spks is not None:
            ret.append("sids")
        if self.spk_embed_dim is not None:
            ret.append("spembs")
        if self.langs is not None:
            ret.append("lids")
        return ret

    def get_output_names(self):
        return ["feat_gen", "out_duration", "out_pitch", "out_energy"]

    def get_dynamic_axes(self):
        ret = {
            "text": {0: "text_length"},
        }
        if self.use_gst:
            ret.update({"feats": {0: "feats_length"}})
        return ret

    def get_model_config(self, path):
        return {
            "model_type": "FastSpeech2",
            "model_path": str(path / f"{self.model_name}.onnx"),
            "eos": self.eos,
        }
