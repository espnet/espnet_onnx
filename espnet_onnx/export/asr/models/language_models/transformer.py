import os

import torch
import torch.nn as nn

from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling

from espnet_onnx.utils.function import subsequent_mask
from espnet_onnx.utils.abs_model import AbsExportModel
from .embed import Embedding


class TransformerLM(nn.Module, AbsExportModel):
    def __init__(self, model, max_seq_len=512, **kwargs):
        super().__init__()
        self.embed = Embedding(model.embed, max_seq_len)
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.model_name = 'transformer_lm'

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

    def get_dynamic_axes(self):
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
            "model_path": os.path.join(path, f'{self.model_name}.onnx'),
            "lm_type": "TransformerLM",
            "odim": self.encoder.encoders[0].size,
            "nlayers": len(self.encoder.encoders)
        }
