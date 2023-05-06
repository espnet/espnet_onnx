import os

import six
import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet2.asr.encoder.vgg_rnn_encoder import \
    VGGRNNEncoder as espnetVGGRNNEncoder
from espnet.nets.pytorch_backend.rnn.encoders import RNN, RNNP, VGG2L

from espnet_onnx.export.asr.get_config import (get_frontend_config,
                                               get_norm_config)
from espnet_onnx.utils.abs_model import AbsExportModel


class OnnxRNNP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.subsample = model.subsample
        is_bidirectional = 2 if model.bidir else 1
        self.initial_state = torch.zeros(is_bidirectional * 1, 1, model.cdim)

    def forward(self, xs_pad: torch.Tensor, ilens: torch.Tensor):
        """Inference version of RNNEncoder.
        Since batch_size is always 1 in inference with onnx, pad_sequence related
        functions can be removed.
        """
        elayer_states = []
        for layer in six.moves.range(self.model.elayers):
            rnn = getattr(
                self.model, ("birnn" if self.model.bidir else "rnn") + str(layer)
            )
            ys, states = rnn(xs_pad, hx=self.initial_state)
            elayer_states.append(states)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            sub = self.subsample[layer + 1]
            if sub > 1:
                ys = ys[:, ::sub]
                ilens = torch.div(ilens, sub, rounding_mode="trunc")
            # (sum _utt frame_utt) x dim
            projection_layer = getattr(self.model, "bt%d" % layer)
            projected = projection_layer(ys.contiguous().view(-1, ys.size(2)))
            xs_pad = projected.view(ys.size(0), ys.size(1), -1)
            if layer < self.model.elayers - 1:
                xs_pad = torch.tanh(xs_pad)

        return xs_pad, ilens, elayer_states  # x: utt list of frame x dim


class OnnxRNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.nbrnn = model.nbrnn
        self.l_last = model.l_last
        is_bidirectional = 2 if model.nbrnn.bidirectional else 1
        num_layers = model.nbrnn.num_layers
        self.initial_state = torch.zeros(
            is_bidirectional * num_layers, 1, model.nbrnn.hidden_size
        )

    def forward(self, xs_pad: torch.Tensor, ilens: torch.Tensor):
        if not isinstance(ilens, torch.Tensor):
            ilens = torch.tensor(ilens)

        ys, states = self.nbrnn(xs_pad, hx=self.initial_state)
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        # (sum _utt frame_utt) x dim
        projected = torch.tanh(self.l_last(ys.contiguous().view(-1, ys.size(2))))
        xs_pad = projected.view(ys.size(0), ys.size(1), -1)
        return xs_pad, ilens, states  # x: utt list of frame x dim


class OnnxVGG2l(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.in_channel = model.in_channel

    def forward(self, xs_pad, ilens, **kwargs):
        if not isinstance(ilens, torch.Tensor):
            ilens = torch.tensor(ilens)
        # x: utt x frame x dim
        # x: utt x 1 (input channel num) x frame x dim
        xs_pad = xs_pad.view(
            xs_pad.size(0),
            xs_pad.size(1),
            self.in_channel,
            xs_pad.size(2) // self.in_channel,
        ).transpose(1, 2)
        # NOTE: max_pool1d ?
        xs_pad = F.relu(self.model.conv1_1(xs_pad))
        xs_pad = F.relu(self.model.conv1_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        xs_pad = F.relu(self.model.conv2_1(xs_pad))
        xs_pad = F.relu(self.model.conv2_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3)
        )

        ilens = torch.ceil(ilens / 4).type(torch.long)
        return xs_pad, ilens, None  # no state in this layer


class RNNEncoderLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        if isinstance(layer, RNNP):
            self.layer = OnnxRNNP(layer)
        elif isinstance(layer, RNN):
            self.layer = OnnxRNN(layer)
        elif isinstance(layer, VGG2L):
            self.layer = OnnxVGG2l(layer)

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)


class RNNEncoder(nn.Module, AbsExportModel):
    def __init__(self, model, frontend, preencoder, feats_dim=80, **kwargs):
        super().__init__()
        self.model = model
        self.model_name = "rnn_encoder"
        self.enc = nn.ModuleList()
        self.feats_dim = feats_dim
        self.frontend = frontend
        for e in model.enc:
            self.enc.append(RNNEncoderLayer(e))

        self.get_frontend(kwargs)
        self.preencoder = preencoder

    def get_frontend(self, kwargs):
        from espnet_onnx.export.asr.models import get_frontend_models

        self.frontend_model = get_frontend_models(self.frontend, kwargs)
        if self.frontend_model is not None:
            self.submodel = []
            self.submodel.append(self.frontend_model)
            self.feats_dim = self.frontend_model.output_dim

    def forward(self, feats):
        current_states = []
        ilens = torch.ones(feats[:, :, 0].shape).sum(dim=-1).type(torch.long)
        if self.preencoder is not None:
            feats, ilens = self.preencoder(feats, ilens)

        for module in self.enc:
            feats, ilens, states = module(feats, ilens)
            current_states.append(states)
        return feats, ilens, current_states

    def get_output_size(self):
        return self.model._output_size

    def get_dummy_inputs(self):
        feats = torch.randn(1, 100, self.feats_dim)
        return feats

    def get_input_names(self):
        return ["feats"]

    def get_output_names(self):
        return ["encoder_out", "encoder_out_lens"]

    def get_dynamic_axes(self):
        return {"feats": {1: "feats_length"}}

    def get_model_config(self, asr_model=None, path=None):
        ret = {}
        ret.update(
            enc_type="RNNEncoder",
            model_path=os.path.join(path, f"{self.model_name}.onnx"),
            is_vggrnn=isinstance(self.model, espnetVGGRNNEncoder),
            frontend=get_frontend_config(asr_model.frontend),
            do_normalize=asr_model.normalize is not None,
            do_postencoder=asr_model.postencoder is not None,
        )
        if ret["do_normalize"]:
            ret.update(normalize=get_norm_config(asr_model.normalize, path))
        # Currently, postencoder is not supported.
        # if ret['do_postencoder']:
        #     ret.update(postencoder=get_postenc_config(self.model.postencoder))
        return ret
