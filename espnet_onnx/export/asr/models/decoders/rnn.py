import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from espnet.nets.pytorch_backend.rnn.attentions import AttLoc
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.transducer.transducer_decoder import TransducerDecoder

from espnet_onnx.utils.function import (
    subsequent_mask,
    make_pad_mask
)
from ..abs_model import AbsModel


def _apply_attention_constraint(
    e, last_attended_idx, backward_window=1, forward_window=3
):
    """Apply monotonic attention constraint.
    **Note** This function is copied from espnet.nets.pytorch_backend.rnn.attention.py
    """
    if e.size(0) != 1:
        raise NotImplementedError("Batch attention constraining is not yet supported.")
    backward_idx = last_attended_idx - backward_window
    forward_idx = last_attended_idx + forward_window
    if backward_idx > 0:
        e[:, :backward_idx] = -float("inf")
    if forward_idx < e.size(1):
        e[:, forward_idx:] = -float("inf")
    return e


class PreDecoder(nn.Module, AbsModel):
    def __init__(self, model):
        super().__init__()
        self.model = model.mlp_enc
    
    def forward(self, enc_h):
        return self.model(enc_h)
    
    def get_dummy_inputs(self):
        di = torch.randn(1, 100, self.model.in_features)
        return (di,)
    
    def get_input_names(self):
        return ['enc_h']
    
    def get_output_names(self):
        return ['pre_compute_enc_h']
    
    def get_dynamic_axes(self):
        return {
            'enc_h': {
                1: 'enc_h_length'
            }
        }
    
    def get_model_config(self, path, idx):
        file_name = os.path.join(path, 'pre-decoder_%d.onnx' % idx)
        return {
            "model_path": file_name,
        }


class onnxAttLoc(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dunits = model.dunits
        self.att_dim = model.att_dim
    
    def forward(
        self,
        dec_z,
        att_prev,
        pre_compute_enc_h,
        enc_h,
        mask,
        scaling=2.0,
        last_attended_idx=None,
        backward_window=1,
        forward_window=3,
    ):
        batch = 1
        dec_z = dec_z.view(batch, self.dunits)
        # att_prev: utt x frame -> utt x 1 x 1 x frame
        # -> utt x att_conv_chans x 1 x frame
        frame_length = att_prev.size(1)
        att_conv = self.model.loc_conv(att_prev.view(batch, 1, 1, frame_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.model.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.model.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.model.gvec(
            torch.tanh(att_conv + pre_compute_enc_h + dec_z_tiled)
        ).squeeze(2)
        
        # mask is an array with -inf
        e = e + mask

        # apply monotonic attention constraint (mainly for TTS)
        if last_attended_idx is not None:
            e = _apply_attention_constraint(
                e, last_attended_idx, backward_window, forward_window
            )

        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        c = torch.sum(enc_h * w.view(batch, frame_length, 1), dim=1)

        return c, w


class RNNDecoder(nn.Module, AbsModel):
    def __init__(self, model):
        super().__init__()
        self.embed = model.embed
        self.model = model
        self.num_encs = model.num_encs
        self.decoder_length = len(model.decoder)
        
        if not isinstance(self.model.att_list[0], AttLoc):
            raise ValueError('Currently only espnet.asr.pytorch_backend.rnn.attention.AttLoc is supported.')
        
        self.att_list = nn.ModuleList()
        for a in model.att_list:
            self.att_list.append(onnxAttLoc(a))

    def forward(self, vy, x, z_prev, c_prev, a_prev, pre_compute_enc_h, enc_h, mask):
        ey = self.embed(vy)  # utt list (1) x zdim
        if self.num_encs == 1:
            att_c, att_w = self.att_list[0](
                z_prev[0],
                a_prev[0],
                pre_compute_enc_h[0],
                enc_h[0],
                mask[0]
            )
        else:
            att_w = []
            for idx in range(self.num_encs):
                _att_c_list, _att_w = self.att_list[idx](
                    z_prev[0],
                    a_prev[idx],
                    pre_compute_enc_h[idx],
                    enc_h[idx],
                    mask[idx]
                )
                att_w.append(_att_w)
                
            att_c, _att_w = self.att_list[self.num_encs](
                z_prev[0],
                a_prev[self.num_encs],
                pre_compute_enc_h[self.num_encs],
                enc_h[self.num_encs],
                mask[self.num_encs]
            )
            att_w.append(_att_w)
        ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
        z_list, c_list = self.rnn_forward(
            ey, z_prev, c_prev
        )
        if self.model.context_residual:
            logits = self.model.output(
                torch.cat((z_list[-1], att_c), dim=-1)
            )
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
                _z_list = self.model.decoder[i](
                    _z_list, z_prev[i]
                )
                ret_z_list.append(_z_list)
        return ret_z_list, ret_c_list

    def get_dummy_inputs(self, enc_size):
        feat_length = 50
        vy = torch.LongTensor([1])
        x = torch.randn(feat_length, enc_size)
        z_prev = [torch.randn(1, self.model.dunits) for _ in range(self.decoder_length)]
        a_prev = [torch.randn(1, feat_length) for _ in range(self.num_encs)]
        c_prev = [torch.randn(1, self.model.dunits) for _ in range(self.decoder_length)]
        pre_compute_enc_h = [torch.randn(1, feat_length, self.model.att_list[i].mlp_enc.out_features) for i in range(self.num_encs)]
        enc_h = [torch.randn(1, feat_length, enc_size) for _ in range(self.num_encs)]
        _m = torch.from_numpy(np.where(make_pad_mask([feat_length])==1, -float('inf'), 0)).type(torch.float32)
        mask = [_m for _ in range(self.num_encs)]
        return (
            vy, x, z_prev, c_prev,
            a_prev, pre_compute_enc_h,
            enc_h, mask
        )

    def get_input_names(self):
        ret = ['vy', 'x']
        ret += ['z_prev_%d' % i for i in range(self.decoder_length)]
        ret += ['c_prev_%d' % i for i in range(self.decoder_length)]
        ret += ['a_prev_%d' % i for i in range(self.num_encs)]
        ret += ['pceh_%d' % i for i in range(self.num_encs)]
        ret += ['enc_h_%d' % i for i in range(self.num_encs)]
        ret += ['mask_%d' % i for i in range(self.num_encs)]
        return ret

    def get_output_names(self):
        ret = ['logp']
        ret += ['c_list_%d' % i for i in range(self.decoder_length)]
        ret += ['z_list_%d' % i for i in range(self.decoder_length)]
        if self.num_encs == 1:
            ret += ['att_w']
        else:
            ret += ['att_w_%d' % i for i in range(self.num_encs + 1)]
        return ret

    def get_dynamic_axes(self):
        # input
        ret = {
            'x': {
                0: 'x_length',
            }
        }
        ret.update({
            'a_prev_%d' % d: {
                1: 'a_prev_%d_length' % d,
            }
            for d in range(self.num_encs)
        })
        ret.update({
            'pceh_%d' % d: {
                1: 'pceh_%d_length' % d,
            }
            for d in range(self.num_encs)
        })
        ret.update({
            'enc_h_%d' % d: {
                1: 'enc_h_%d_length' % d,
            }
            for d in range(self.num_encs)
        })
        ret.update({
            'mask_%d' % d:{
                0: 'mask_%d_length' % d,
                1: 'mask_%d_height' % d
            }
            for d in range(self.num_encs)
        })
        #output
        ret.update({
            'att_w_%d' % d:{
                1: 'att_w_%d_length' % d
            }
            for d in range(self.num_encs)
        })
        return ret

    def get_model_config(self, path):
        file_name = os.path.join(path, 'decoder.onnx')
        return {
            "dec_type": "RNNDecoder",
            "model_path": file_name,
            "dlayers": self.model.dlayers,
            "odim": self.model.odim,
            "dunits": self.model.dunits,
            "decoder_length": self.decoder_length,
            "rnn_type": self.model.dtype,
            "predecoder": [
                {
                    "model_path": os.path.join(path, 'predecoder_%d.onnx' % d)
                }
                for d in range(self.num_encs)
            ]
        }
        

