
from typing import (
    Optional,
    Tuple
)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet_onnx.export.asr.models.language_models.embed import Embedding
from espnet_onnx.utils.torch_function import MakePadMask
from espnet_onnx.utils.abs_model import AbsExportModel
from espnet_onnx.utils.export.layers.predecoder import PreDecoder


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
        xs = self.model.embed(xs.unsqueeze(0)).transpose(1, 2)
        if self.model.convs is not None:
            for i in six.moves.range(len(self.model.convs)):
                if self.model.use_residual:
                    xs = xs + self.model.convs[i](xs)
                else:
                    xs = self.model.convs[i](xs)
                    
        if self.model.blstm is None:
            return xs.transpose(1, 2)
        
        xs, _ = self.model.blstm(xs)  # (B, Tmax, C)
        return xs[0]


class OnnxTacotron2Encoder(nn.Module, AbsExportModel):
    def __init__(
        self,
        model,
        **kwargs
    ):
        super().__init__()
        self.model_name = 'tts_model_encoder'
        
        # HPs
        self.odim = model.odim
        self.use_gst = model.use_gst
        self.spks = model.spks
        self.langs = model.langs
        self.spk_embed_dim = model.spk_embed_dim
        self.eos = model.eos
        
        # models
        self.make_pad_mask = MakePadMask(max_seq_len)
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
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
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
        h = self.enc(x)
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
        feats = torch.randn(10, self.odim) \
            if self.use_gst else None

        sids = torch.LongTensor([0]) \
            if self.spks is not None else None

        spembs = torch.randn(self.spk_embed_dim) \
            if self.spk_embed_dim is not None else None

        lids = torch.LongTensor([0]) \
            if self.langs is not None else None

        return (text, text_length, feats, sids, spembs, lids)

    def get_input_names(self):
        return ['text', 'feats', 'sids', 'spembs', 'lids']

    def get_output_names(self):
        return ['h']

    def get_dynamic_axes(self):
        ret = {
            'text': {0: 'text_length'},
            'feats': {0: 'feats_length'},
        }
        return ret

    def get_model_config(self, path):
        return {
            'model_type': 'Tacotron2Encoder',
            'model_path': str(path / f'{self.model_name}.onnx'),
            'eos': self.eos,
        }


class PostDecoder(nn.Module, AbsExportModel):
    def __init__(self, model, activation, odim):
        super().__init__()
        self.model = model
        self.odim = odim
        self.activation = activation
        self.model_name = f'postdecoder_{idx}'
    
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
        return ['x']

    def get_output_names(self):
        return ['out']

    def get_dynamic_axes(self):
        return {
            'x': {
                2: 'x_length'
            }
        }

    def get_model_config(self, path):
        return {
            "model_path": os.path.join(path, f'{self.model_name}.onnx'),
        }


class OnnxTacotron2Decoder(nn.Module, AbsExportModel):
    def __init__(
        self,
        model,
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=10.0,
        backward_window=None,
        forward_window=None,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.submodel = []
        self.model_name = 'tts_model_decoder'
        
        # HPs
        self.odim = model.odim
        self.threshold = threshold
        self.minlenratio = minlenratio
        self.maxlenratio = maxlenratio
        self.backward_window = backward_window
        self.forward_window = forward_window
        self.use_concat = model.use_concat
        self.cumulate_att_w = model.cumulate_att_w
        
        # models
        self.att = model.dec.att
        self.prenet = model.dec.prenet
        self.lstm = model.dec.lstm
        self.feats_out = model.dec.feats_out
        self.prob_out = model.dec.prob_out
        self.output_activation_fn = model.dec.output_activation_fn
        
        self.subnet.append(
            PreDecoder(att, 0)
        )
        self.subnet.append(
            PostDecoder(
                model.dec.postnet,
                model.dec.output_activation_fn,
                self.odim
            )
        )

    def forward(
        self,
        z_prev,
        c_prev,
        a_prev,
        pre_compute_enc_h,
        enc_h,
        mask,
        prev_out,
    ):
        # decoder calculation
        if self.use_att_extra_inputs:
            att_c, att_w = self.att(
                z_prev[0],
                a_prev[idx],
                pre_compute_enc_h[idx],
                enc_h[idx],
                mask[idx]
                prev_out,
                backward_window=self.backward_window,
                forward_window=self.forward_window,
            )
        else:
            att_c, att_w = self.att(
                z_prev[0],
                a_prev[idx],
                pre_compute_enc_h[idx],
                enc_h[idx],
                mask[idx]
                backward_window=self.backward_window,
                forward_window=self.forward_window,
            )

        att_ws += [att_w]
        prenet_out = self.prenet(prev_out) if self.prenet is not None else prev_out
        xs = torch.cat([att_c, prenet_out], dim=1)
        ret_z_list = []
        ret_c_list = []
        _z_list, _c_list = self.lstm[0](xs, (z_prev[0], c_prev[0]))
        ret_z_list.append(_z_list)
        ret_c_list.append(_c_list)
        for i in six.moves.range(1, len(self.lstm)):
            _z_list, _c_list = self.lstm[i](
                z_prev[i - 1], (z_prev[i], c_prev[i])
            )
            ret_z_list.append(_z_list)
            ret_c_list.append(_c_list)
        zcs = (
            torch.cat([z_prev[-1], att_c], dim=1)
            if self.use_concate
            else z_list[-1]
        )
        outs += [self.feat_out(zcs).view(1, self.odim, -1)]  # [(1, odim, r), ...]
        probs += [torch.sigmoid(self.prob_out(zcs))[0]]  # [(r), ...]
        if self.output_activation_fn is not None:
            prev_out = self.output_activation_fn(outs[-1][:, :, -1])  # (1, odim)
        else:
            prev_out = outs[-1][:, :, -1]  # (1, odim)
        if self.cumulate_att_w and prev_att_w is not None:
            prev_att_w = prev_att_w + att_w  # Note: error when use +=
        else:
            prev_att_w = att_w

        return (
            outs,
            ret_c_list,
            ret_z_list,
            prev_att_w,
            prev_out
        )
        
    def get_a_prev(self, feat_length, att):
        ret = torch.randn(1, feat_length)
        # if att.att_type == 'location2d':
        #     ret = torch.randn(1, att.att_win, feat_length)
        if att.att_type in ('coverage', 'coverage_location'):
            ret = torch.randn(1, 1, feat_length)
        return ret

    def get_dummy_inputs(self, enc_size):
        feat_length = 50
        z_prev = [torch.randn(1, self.model.lstm[i].hidden_size)
                  for i in range(len(self.model.lstm))]
        a_prev = self.get_a_prev(feat_length, self.att)
        c_prev = [torch.randn(1, self.model.lstm[i].hidden_size)
                  for i in range(len(self.model.lstm))]
        pre_compute_enc_h = self.get_precompute_enc_h(feat_length)
        enc_h = torch.randn(1, feat_length, enc_size)
        mask = torch.from_numpy(np.where(make_pad_mask(
            [feat_length]) == 1, -float('inf'), 0)).type(torch.float32)
        prev_out = torch.zeros(1, self.odim)
        return (
            z_prev, c_prev,
            a_prev, pre_compute_enc_h,
            enc_h, mask, prev_out
        )
    
    def get_precompute_enc_h(self,feat_length):
        return torch.randn(1, feat_length, self.model.att.mlp_enc.out_features)

    def get_input_names(self):
        ret = ['z_prev_%d' % i for i in range(len(self.model.lstm))]
        ret += ['c_prev_%d' % i for i in range(len(self.model.lstm))]
        ret += 'a_prev'
        ret += 'pceh'
        ret += 'enc_h'
        ret += 'mask'
        ret += 'prev_out'
        return ret

    def get_output_names(self):
        ret = ['outs']
        ret += ['c_list_%d' % i for i in range(len(self.model.lstm))]
        ret += ['z_list_%d' % i for i in range(len(self.model.lstm))]
        ret += ['prev_att_w', 'prev_out']
        return ret

    def get_dynamic_axes(self):
        # input
        ret = {}
        ret.update({
            'prev_att_w': {
                self.att.get_dynamic_axes(): 'a_prev_length',
            }
        })
        ret.update({
            'pceh': {
                1: 'pceh_length',
            }
        })
        ret.update({
            'enc_h': {
                1: 'enc_h_length',
            }
        })
        ret.update({
            'mask': {
                0: 'mask_length',
                1: 'mask_height'
            }
        })
        # output
        ret.update({
            'prev_att_w': {
                1: 'prev_att_w_length'
            }
        })
        return ret

    def get_model_config(self, path):
        ret = {
            "dec_type": "RNNDecoder",
            "model_path": os.path.join(path, f'{self.model_name}.onnx'),
            "dlayers": self.model.dlayers,
            "odim": self.model.odim,
            "dunits": self.model.dunits,
            "decoder_length": self.decoder_length,
            "rnn_type": self.model.dtype,
            "predecoder": [
                {
                    "model_path": os.path.join(path, f'predecoder_{i}.onnx'),
                    "att_type": a.att_type
                }
                for i,a in enumerate(self.att_list)
                if not isinstance(a, OnnxNoAtt)
            ]
        }
        if hasattr(self.model, 'att_win'):
            ret.update(att_win=self.model.att_win)
        return ret