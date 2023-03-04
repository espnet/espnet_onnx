import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet.nets.pytorch_backend.rnn.attentions import NoAtt

from espnet_onnx.export.layers.attention import OnnxNoAtt, get_attention
from espnet_onnx.export.layers.predecoder import PreDecoder
from espnet_onnx.utils.abs_model import AbsExportModel
from espnet_onnx.utils.function import make_pad_mask


class RNNDecoder(nn.Module, AbsExportModel):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.embed = model.embed
        self.submodel = []
        self.model = model
        self.num_encs = model.num_encs
        self.decoder_length = len(model.decoder)
        self.model_name = "rnn_decoder"

        self.att_list = nn.ModuleList()
        for i, a in enumerate(model.att_list):
            self.att_list.append(get_attention(a))
            self.submodel.append(PreDecoder(a, idx=i))

    def forward(self, vy, z_prev, c_prev, a_prev, pre_compute_enc_h, enc_h, mask):
        ey = self.embed(vy)  # utt list (1) x zdim
        if self.num_encs == 1:
            att_c, att_w = self.att_list[0](
                z_prev[0], a_prev[0], pre_compute_enc_h[0], enc_h[0], mask[0]
            )
        else:
            att_w = []
            for idx in range(self.num_encs):
                _, _att_w = self.att_list[idx](
                    z_prev[0],
                    a_prev[idx],
                    pre_compute_enc_h[idx],
                    enc_h[idx],
                    mask[idx],
                )
                att_w.append(_att_w)

            att_c, _att_w = self.att_list[self.num_encs](
                z_prev[0],
                a_prev[self.num_encs],
                pre_compute_enc_h[self.num_encs],
                enc_h[self.num_encs],
                mask[self.num_encs],
            )
            att_w.append(_att_w)
        if att_c is not None:
            ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
        z_list, c_list = self.rnn_forward(ey, z_prev, c_prev)
        if self.model.context_residual:
            logits = self.model.output(torch.cat((z_list[-1], att_c), dim=-1))
        else:
            logits = self.model.output(z_list[-1])
        logp = F.log_softmax(logits, dim=1).squeeze(0)
        return (
            logp,
            c_list,
            z_list,
            att_w,
        )

    def rnn_forward(self, ey, z_prev, c_prev):
        ret_z_list = []
        ret_c_list = []
        if self.model.dtype == "lstm":
            _z_list, _c_list = self.model.decoder[0](ey, (z_prev[0], c_prev[0]))
            ret_z_list.append(_z_list)
            ret_c_list.append(_c_list)
            for i in range(1, self.model.dlayers):
                _z_list, _c_list = self.model.decoder[i](
                    _z_list,
                    (z_prev[i], c_prev[i]),
                )
                ret_z_list.append(_z_list)
                ret_c_list.append(_c_list)
        else:
            _z_list = self.model.decoder[0](ey, z_prev[0])
            ret_z_list.append(_z_list)
            for i in range(1, self.model.dlayers):
                _z_list = self.model.decoder[i](_z_list, z_prev[i])
                ret_z_list.append(_z_list)
        return ret_z_list, ret_c_list

    def get_a_prev(self, feat_length, att):
        ret = torch.randn(1, feat_length)
        # if att.att_type == 'location2d':
        #     ret = torch.randn(1, att.att_win, feat_length)
        if att.att_type in ("coverage", "coverage_location"):
            ret = torch.randn(1, 1, feat_length)
        return ret

    def get_dummy_inputs(self, enc_size):
        feat_length = 50
        vy = torch.LongTensor([1])
        z_prev = [torch.randn(1, self.model.dunits) for _ in range(self.decoder_length)]
        a_prev = [self.get_a_prev(feat_length, att) for att in self.att_list]
        c_prev = [torch.randn(1, self.model.dunits) for _ in range(self.decoder_length)]
        pre_compute_enc_h = [
            self.get_precompute_enc_h(i, feat_length) for i in range(self.num_encs)
        ]
        enc_h = [torch.randn(1, feat_length, enc_size) for _ in range(self.num_encs)]
        _m = torch.from_numpy(
            np.where(make_pad_mask([feat_length]) == 1, -10000.0, 0)
        ).type(torch.float32)
        mask = [_m for _ in range(self.num_encs)]
        return (vy, z_prev, c_prev, a_prev, pre_compute_enc_h, enc_h, mask)

    def get_precompute_enc_h(self, idx, feat_length):
        if isinstance(self.model.att_list[idx], NoAtt):
            return torch.randn(1, 1, 1)
        else:
            return torch.randn(
                1, feat_length, self.model.att_list[idx].mlp_enc.out_features
            )

    def get_input_names(self):
        ret = ["vy"]
        ret += ["z_prev_%d" % i for i in range(self.decoder_length)]
        ret += ["c_prev_%d" % i for i in range(self.decoder_length)]
        ret += ["a_prev_%d" % i for i in range(self.num_encs)]
        ret += ["pceh_%d" % i for i in range(self.num_encs)]
        ret += ["enc_h_%d" % i for i in range(self.num_encs)]
        ret += ["mask_%d" % i for i in range(self.num_encs)]
        return ret

    def get_output_names(self):
        ret = ["logp"]
        ret += ["c_list_%d" % i for i in range(self.decoder_length)]
        ret += ["z_list_%d" % i for i in range(self.decoder_length)]
        if self.num_encs == 1:
            ret += ["att_w"]
        else:
            ret += ["att_w_%d" % i for i in range(self.num_encs + 1)]
        return ret

    def get_dynamic_axes(self):
        # input
        ret = {}
        ret.update(
            {
                "a_prev_%d"
                % i: {
                    a.get_dynamic_axes(): "a_prev_%d_length" % i,
                }
                for i, a in enumerate(self.att_list)
            }
        )
        ret.update(
            {
                "pceh_%d"
                % d: {
                    1: "pceh_%d_length" % d,
                }
                for d in range(self.num_encs)
            }
        )
        ret.update(
            {
                "enc_h_%d"
                % d: {
                    1: "enc_h_%d_length" % d,
                }
                for d in range(self.num_encs)
            }
        )
        ret.update(
            {
                "mask_%d" % d: {0: "mask_%d_length" % d, 1: "mask_%d_height" % d}
                for d in range(self.num_encs)
            }
        )
        # output
        ret.update(
            {"att_w_%d" % d: {1: "att_w_%d_length" % d} for d in range(self.num_encs)}
        )
        return ret

    def get_model_config(self, path):
        ret = {
            "dec_type": "RNNDecoder",
            "model_path": os.path.join(path, f"{self.model_name}.onnx"),
            "dlayers": self.model.dlayers,
            "odim": self.model.odim,
            "dunits": self.model.dunits,
            "decoder_length": self.decoder_length,
            "rnn_type": self.model.dtype,
            "predecoder": [
                {
                    "model_path": os.path.join(path, f"predecoder_{i}.onnx")
                    if not isinstance(a, OnnxNoAtt)
                    else "",
                    "att_type": a.att_type,
                }
                for i, a in enumerate(self.att_list)
                # if not isinstance(a, OnnxNoAtt)
            ],
        }
        if hasattr(self.model, "att_win"):
            ret.update(att_win=self.model.att_win)
        return ret
