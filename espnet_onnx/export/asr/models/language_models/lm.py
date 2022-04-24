import os

import torch
import torch.nn as nn

from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,
    ScaledPositionalEncoding,
    RelPositionalEncoding,
    LegacyRelPositionalEncoding,
    StreamPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.subsampling_without_posenc import Conv2dSubsamplingWOPosEnc

from espnet_onnx.utils.function import subsequent_mask
from ..abs_model import AbsModel
from .embed import (
    OnnxPositionalEncoding,
    OnnxScaledPositionalEncoding,
    OnnxRelPositionalEncoding,
    OnnxLegacyRelPositionalEncoding,
    OnnxStreamPositionalEncoding,
)


def get_pos_emb(pos_emb):
    if isinstance(pos_emb, LegacyRelPositionalEncoding):
        return OnnxLegacyRelPositionalEncoding(pos_emb)
    elif isinstance(pos_emb, ScaledPositionalEncoding):
        return OnnxScaledPositionalEncoding(pos_emb)
    elif isinstance(pos_emb, RelPositionalEncoding):
        return OnnxRelPositionalEncoding(pos_emb)
    elif isinstance(pos_emb, PositionalEncoding):
        return OnnxPositionalEncoding(pos_emb)
    elif isinstance(pos_emb, StreamPositionalEncoding):
        return OnnxStreamPositionalEncoding(pos_emb)
    elif (isinstance(pos_emb, nn.Sequential) and len(pos_emb) == 0) \
        or (isinstance(pos_emb, Conv2dSubsamplingWOPosEnc)):
        return pos_emb
    else:
        raise ValueError('Embedding model is not supported.')


class Embedding(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        if not isinstance(model, nn.Embedding):
            if (
                isinstance(model, Conv2dSubsampling)
                or isinstance(model, Conv2dSubsampling2)
                or isinstance(model, Conv2dSubsampling6)
                or isinstance(model, Conv2dSubsampling8)
            ):
                self.model.out[-1] = get_pos_emb(model.out[-1])
            else:
                self.model[-1] = get_pos_emb(model[-1])

    def forward(self, x, mask=None):
        if mask is None:
            return self.model(x)
        else:
            xs = self.model(x, mask)
            if isinstance(self.model, Conv2dSubsampling):
                return xs, mask[:, :, :-2:2][:, :, :-2:2]
            elif isinstance(self.model, Conv2dSubsampling2):
                return xs, mask[:, :, :-2:2][:, :, :-2:1]
            elif isinstance(self.model, Conv2dSubsampling6):
                return xs, mask[:, :, :-2:2][:, :, :-4:3]
            elif isinstance(self.model, Conv2dSubsampling8):
                return xs, mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]


class SequentialRNNLM(nn.Module, AbsModel):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.rnn = model.rnn
        self.rnn_type = model.rnn_type
        self.decoder = model.decoder
        self.nlayers = model.nlayers
        self.nhid = model.nhid

    def forward(self, y, hidden1, hidden2=None):
        # batch_score function.
        emb = self.encoder(y)
        if self.rnn_type == 'LSTM':
            output, (hidden1, hidden2) = self.rnn(emb, (hidden1, hidden2))
        else:
            output, hidden1 = self.rnn(emb, hidden1)

        decoded = self.decoder(
            output.contiguous().view(output.size(0) * output.size(1), output.size(2))
        )
        if self.rnn_type == 'LSTM':
            return (
                decoded.view(output.size(0), output.size(1), decoded.size(1)),
                hidden1,
                hidden2
            )
        else:
            return (
                decoded.view(output.size(0), output.size(1), decoded.size(1)),
                hidden1
            )

    def get_dummy_inputs(self):
        tgt = torch.LongTensor([0, 1]).unsqueeze(0)
        hidden = torch.randn(self.nlayers, 1, self.nhid)
        if self.rnn_type == 'LSTM':
            return (tgt, hidden, hidden)
        else:
            return (tgt, hidden)

    def get_input_names(self):
        if self.rnn_type == 'LSTM':
            return ['x', 'in_hidden1', 'in_hidden2']
        else:
            return ['x', 'in_hidden1']

    def get_output_names(self):
        if self.rnn_type == 'LSTM':
            return ['y', 'out_hidden1', 'out_hidden2']
        else:
            return ['y', 'out_hidden1']

    def get_dynamix_axes(self):
        ret = {
            'x': {
                0: 'x_batch',
                1: 'x_length'
            },
            'y': {
                0: 'y_batch'
            },
            'in_hidden1': {
                1: 'hidden1_batch'
            },
            'out_hidden1': {
                1: 'out_hidden1_batch'
            }
        }
        if self.rnn_type == 'LSTM':
            ret.update({
                'in_hidden2': {
                    1: 'hidden2_batch'
                },
                'out_hidden2': {
                    1: 'out_hidden2_batch'
                }
            })
        return ret

    def get_model_config(self, path):
        return {
            "use_lm": True,
            "model_path": os.path.join(path, "lm.onnx"),
            "lm_type": "SequentialRNNLM",
            "rnn_type": self.rnn_type,
            "nhid": self.nhid,
            "nlayers": self.nlayers
        }


class TransformerLM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed = Embedding(model.embed)
        self.encoder = model.encoder
        self.decoder = model.decoder

    def forward(self, y, mask, cache):
        xs = self.embed(y)
        # forward_one_step of Encoder
        if isinstance(self.encoder.embed, Conv2dSubsampling):
            xs, mask = self.encoder.embed(xs, mask)
        else:
            xs = self.encoder.embed(xs)

        new_cache = []
        for c, e in zip(cache, self.encoder.encoders):
            xs, mask = e(xs, mask, cache=c)
            new_cache.append(xs)

        if self.encoder.normalize_before:
            xs = self.encoder.after_norm(xs)

        h = self.decoder(xs[:, -1])
        return h, new_cache

    def get_dummy_inputs(self):
        tgt = torch.LongTensor([0, 1]).unsqueeze(0)
        ys_mask = tgt != 0
        m = torch.from_numpy(subsequent_mask(ys_mask.shape[-1])[None, :])
        mask = ys_mask[None, :] * m
        cache = [
            torch.zeros((1, 1, self.encoder.encoders[0].size))
            for _ in range(len(self.encoder.encoders))
        ]
        return (tgt, mask, cache)

    def get_input_names(self):
        return ['tgt', 'tgt_mask'] \
            + ['cache_%d' % i for i in range(len(self.encoder.encoders))]

    def get_output_names(self):
        return ['y'] \
            + ['out_cache_%d' % i for i in range(len(self.encoder.encoders))]

    def get_dynamix_axes(self):
        ret = {
            'tgt': {
                0: 'tgt_batch',
                1: 'tgt_length'
            },
            'tgt_mask': {
                0: 'tgt_mask_batch',
                1: 'tgt_mask_length',
                2: 'tgt_mask_height'
            }
        }
        ret.update({
            'cache_%d' % d: {
                0: 'cache_%d_batch' % d,
                1: 'cache_%d_length' % d
            }
            for d in range(len(self.encoder.encoders))
        })
        ret.update({
            'out_cache_%d' % d: {
                0: 'out_cache_%d_batch' % d,
                1: 'out_cache_%d_length' % d
            }
            for d in range(len(self.encoder.encoders))
        })
        return ret

    def get_model_config(self, path):
        return {
            "use_lm": True,
            "model_path": os.path.join(path, "lm.onnx"),
            "lm_type": "TransformerLM",
            "odim": self.encoder.encoders[0].size,
            "nlayers": len(self.encoder.encoders)
        }
