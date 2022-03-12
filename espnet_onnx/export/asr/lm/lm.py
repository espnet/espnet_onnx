import torch
import torch.nn as nn

from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
)
from .embed import (
    OnnxPositionalEncoding,
    OnnxScaledPositionalEncoding,
    OnnxRelPositionalEncoding,
    OnnxLegacyRelPositionalEncoding
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
    elif isinstance(pos_emb, nn.Sequential) and len(pos_emb) == 0:
        return pos_emb
    else:
        raise Error('Embedding model is not supported.')


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


class SequentialRNNLM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.rnn = model.rnn
        self.rnn_type = model.rnn_type
        self.decoder = model.decoder

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
