import os
from typing import Optional

import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet.nets.pytorch_backend.rnn.attentions import AttForward, AttForwardTA

from espnet_onnx.export.layers.attention import get_attention
from espnet_onnx.export.layers.predecoder import PreDecoder
from espnet_onnx.utils.abs_model import AbsExportModel
from espnet_onnx.utils.torch_function import MakePadMask, normalize


class OnnxEncoderLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, xs):
        """Inference.
        Args:
            x (Tensor): The sequeunce of character ids (T,)
                    or acoustic feature (T, idim * encoder_reduction_factor).
        Returns:
            Tensor: The sequences of encoder states(T, eunits).
        """
        xs = self.model.embed(xs).transpose(1, 2)
        if self.model.convs is not None:
            for i in six.moves.range(len(self.model.convs)):
                if self.model.use_residual:
                    xs = xs + self.model.convs[i](xs)
                else:
                    xs = self.model.convs[i](xs)

        if self.model.blstm is None:
            return xs.transpose(1, 2)

        xs, _ = self.model.blstm(xs.transpose(1, 2))  # (B, Tmax, C)
        return xs[0]


class OnnxTacotron2Encoder(nn.Module, AbsExportModel):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model_name = "tts_model_encoder"

        # HPs
        self.odim = model.odim
        self.use_gst = model.use_gst
        self.spks = model.spks
        self.langs = model.langs
        self.spk_embed_dim = model.spk_embed_dim
        self.eos = model.eos

        # models
        self.enc = OnnxEncoderLayer(model.enc)
        if self.use_gst:
            self.gst = model.gst
        if self.spks is not None:
            self.sid_emb = model.sid_emb
        if self.langs is not None:
            self.lid_emb = model.lid_emb
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = model.spk_embed_integration_type
            if self.spk_embed_integration_type == "add":
                self.projection = model.projection

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds
            spembs = normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = torch.cat([hs, spembs], dim=-1)
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
        x = text
        y = feats
        spemb = spembs

        # add eos at the last of sequence
        x = F.pad(x, [0, 1], "constant", self.eos)

        # inference
        h = self.enc(x.unsqueeze(0))
        if self.use_gst:
            style_emb = self.gst(y.unsqueeze(0))
            h = h + style_emb
        if self.spks is not None:
            sid_emb = self.sid_emb(sids.view(-1))
            h = h + sid_emb
        if self.langs is not None:
            lid_emb = self.lid_emb(lids.view(-1))
            h = h + lid_emb
        if self.spk_embed_dim is not None:
            hs, spembs = h.unsqueeze(0), spemb.unsqueeze(0)
            h = self._integrate_with_spk_embed(hs, spembs)[0]

        return h

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
        return ["h"]

    def get_dynamic_axes(self):
        ret = {
            "text": {0: "text_length"},
        }
        if self.use_gst:
            ret.update({"feats": {0: "feats_length"}})
        return ret

    def get_model_config(self, path):
        return {
            "model_type": "Tacotron2Encoder",
            "model_path": str(path / f"{self.model_name}.onnx"),
            "eos": self.eos,
        }


class PostDecoder(nn.Module, AbsExportModel):
    def __init__(self, model, activation, odim):
        super().__init__()
        self.model = model
        self.odim = odim
        self.activation = activation
        self.model_name = f"postdecoder"

    def forward(self, x):
        if self.model is not None:
            x = x + self.model(x)
        x = x.transpose(2, 1).squeeze(0)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def get_dummy_inputs(self):
        x = torch.randn(1, self.odim, 100)
        return (x,)

    def get_input_names(self):
        return ["x"]

    def get_output_names(self):
        return ["out"]

    def get_dynamic_axes(self):
        return {"x": {2: "x_length"}}

    def get_model_config(self, path):
        return {
            "model_path": os.path.join(path, f"{self.model_name}.onnx"),
        }


class OnnxTacotron2Decoder(nn.Module, AbsExportModel):
    def __init__(
        self,
        model,
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=10.0,
        use_att_constraint=False,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.submodel = []
        self.model_name = "tts_model_decoder"
        self.model.eval()

        # HPs
        self.odim = model.odim
        self.enc_out = model.enc.blstm.hidden_size * 2
        self.threshold = threshold
        self.minlenratio = minlenratio
        self.maxlenratio = maxlenratio
        self.use_concate = model.dec.use_concate
        self.cumulate_att_w = model.cumulate_att_w
        self.use_att_extra_inputs = model.dec.use_att_extra_inputs
        self.use_att_constraint = use_att_constraint
        self.onnx_export = model.dec.postnet is not None
        self.reduction_factor = model.dec.reduction_factor

        self.prenet = model.dec.prenet
        self.lstm = model.dec.lstm
        self.feat_out = model.dec.feat_out
        self.prob_out = model.dec.prob_out
        self.output_activation_fn = model.dec.output_activation_fn
        # get attention model
        self.att = get_attention(model.dec.att)
        if self.att is None:
            # attention model is AttForward or AttforwardTA.
            if isinstance(model.dec.att, AttForward) or isinstance(
                model.dec.att, AttForwardTA
            ):
                raise ValueError(
                    "AttForwardTA or AttForward is not currently supported"
                )

        self.submodel.append(PreDecoder(model.dec.att, 0))
        self.submodel.append(
            PostDecoder(model.dec.postnet, model.dec.output_activation_fn, self.odim)
        )

    def forward(
        self,
        c_prev,
        z_prev,
        a_prev,
        pre_compute_enc_h,
        enc_h,
        mask,
        prev_out,
        last_att_mask,
    ):
        # decoder calculation
        if self.use_att_extra_inputs:
            att_c, att_w = self.att(
                z_prev[0],
                a_prev,
                pre_compute_enc_h,
                enc_h,
                mask,
                prev_out,
                last_att_mask=last_att_mask,
            )
        else:
            att_c, att_w = self.att(
                z_prev[0],
                a_prev,
                pre_compute_enc_h,
                enc_h,
                mask,
                last_att_mask=last_att_mask,
            )

        prenet_out = prev_out
        if self.prenet is not None:
            for i in six.moves.range(len(self.prenet.prenet)):
                prenet_out = self.prenet.prenet[i](prenet_out)

        xs = torch.cat([att_c, prenet_out], dim=1)
        new_z_cache = []
        new_c_cache = []
        _z_list, _c_list = self.lstm[0](xs, (z_prev[0], c_prev[0]))
        new_z_cache.append(_z_list)
        new_c_cache.append(_c_list)

        for i in six.moves.range(1, len(self.lstm)):
            _z_list, _c_list = self.lstm[i](_z_list, (z_prev[i], c_prev[i]))
            new_z_cache.append(_z_list)
            new_c_cache.append(_c_list)

        zcs = torch.cat([_z_list, att_c], dim=1) if self.use_concate else _z_list
        out = self.feat_out(zcs).view(1, self.odim, -1)  # [(1, odim, r), ...]
        prob = torch.sigmoid(self.prob_out(zcs))[0]  # [(r), ...]
        if self.output_activation_fn is not None:
            prev_out = self.output_activation_fn(out[:, :, -1])  # (1, odim)
        else:
            prev_out = out[:, :, -1]  # (1, odim)

        return (
            out,
            prob,
            att_w,
            prev_out,
            new_c_cache,
            new_z_cache,
        )

    def get_a_prev(self, feat_length, att):
        ret = torch.randn(1, feat_length)
        # if att.att_type == 'location2d':
        #     ret = torch.randn(1, att.att_win, feat_length)
        if att.att_type in ("coverage", "coverage_location"):
            ret = torch.randn(1, 1, feat_length)
        return ret

    def get_dummy_inputs(self):
        feat_length = 50
        make_pad_mask = MakePadMask(torch.LongTensor([1024]))
        z_prev = [
            torch.randn(1, self.model.dec.lstm[i].hidden_size)
            for i in range(len(self.model.dec.lstm))
        ]
        a_prev = self.get_a_prev(feat_length, self.att)
        c_prev = [
            torch.randn(1, self.model.dec.lstm[i].hidden_size)
            for i in range(len(self.model.dec.lstm))
        ]
        pre_compute_enc_h = self.get_precompute_enc_h(feat_length)
        enc_h = torch.randn(1, feat_length, self.enc_out)
        mask = torch.from_numpy(
            np.where(
                make_pad_mask(torch.LongTensor([feat_length])) == 1, -float("inf"), 0
            )
        ).type(torch.float32)
        prev_out = torch.zeros(1, self.odim)
        if self.use_att_constraint:
            last_att_mask = torch.zeros(feat_length)
        else:
            last_att_mask = None
        return (
            c_prev,
            z_prev,
            a_prev,
            pre_compute_enc_h,
            enc_h,
            mask,
            prev_out,
            last_att_mask,
        )

    def get_precompute_enc_h(self, feat_length):
        return torch.randn(1, feat_length, self.model.dec.att.mlp_enc.out_features)

    def get_input_names(self):
        ret = ["c_prev_%d" % i for i in range(len(self.model.dec.lstm))]
        ret += ["z_prev_%d" % i for i in range(len(self.model.dec.lstm))]
        ret += ["a_prev", "pceh", "enc_h", "mask", "prev_in", "last_att_mask"]
        return ret

    def get_output_names(self):
        ret = ["out", "prob", "prev_att_w", "prev_out"]
        ret += ["c_list_%d" % i for i in range(len(self.model.dec.lstm))]
        ret += ["z_list_%d" % i for i in range(len(self.model.dec.lstm))]
        return ret

    def get_dynamic_axes(self):
        # input
        ret = {}
        ret.update(
            {
                "a_prev": {
                    self.att.get_dynamic_axes(): "a_prev_length",
                }
            }
        )
        ret.update(
            {
                "pceh": {
                    1: "pceh_length",
                }
            }
        )
        ret.update(
            {
                "enc_h": {
                    1: "enc_h_length",
                }
            }
        )
        ret.update({"mask": {0: "mask_length", 1: "mask_height"}})
        if self.use_att_constraint:
            ret.update(
                {
                    "last_att_mask": {
                        0: "last_att_mask_length",
                    }
                }
            )
        return ret

    def get_model_config(self, path):
        ret = {
            "model_type": "Tacotron2Decoder",
            "model_path": os.path.join(path, f"{self.model_name}.onnx"),
            "dlayers": len(self.model.dec.lstm),
            "odim": self.odim,
            "dunits": self.model.dec.lstm[0].hidden_size,
            "threshold": self.threshold,
            "minlenratio": self.minlenratio,
            "maxlenratio": self.maxlenratio,
            "reduction_factor": self.reduction_factor,
            "cumulate_att_w": self.cumulate_att_w,
            "predecoder": {
                "model_path": os.path.join(path, f"{self.submodel[0].model_name}.onnx"),
                "att_type": self.att.att_type,
            },
            "postdecoder": {
                "model_path": os.path.join(path, f"{self.submodel[1].model_name}.onnx"),
                "onnx_export": self.onnx_export,
            },
            "use_att_constraint": self.use_att_constraint,
        }
        if hasattr(self.model, "att_win"):
            ret.update(att_win=self.model.att_win)
        return ret
