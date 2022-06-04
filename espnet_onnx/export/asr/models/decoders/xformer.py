import os

import torch
import torch.nn as nn

from espnet2.asr.decoder.transformer_decoder import TransformerDecoder

from espnet_onnx.utils.function import subsequent_mask
from ..language_models.embed import Embedding
from espnet_onnx.utils.abs_model import AbsExportModel


class XformerDecoder(nn.Module, AbsExportModel):
    def __init__(self, model, max_seq_len=512, **kwargs):
        super().__init__()
        self.embed = Embedding(model.embed, max_seq_len)
        self.model = model
        self.model_name = 'xformer_decoder'

    def forward(self, tgt, tgt_mask, memory, cache):
        x = self.embed(tgt)
        new_cache = []
        for c, decoder in zip(cache, self.model.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, None, cache=c
            )
            new_cache.append(x)  # (1, L, 512) * n_layer
        y = self.model.after_norm(x[:, -1])
        y = torch.log_softmax(self.model.output_layer(y), dim=-1)
        return y, new_cache

    def get_dummy_inputs(self, enc_size):
        tgt = torch.LongTensor([0, 1]).unsqueeze(0)
        tgt_mask = torch.from_numpy(subsequent_mask(2)[None, :])
        enc_out = torch.randn(1, 100, enc_size)
        cache = [
            torch.zeros((1, 1, self.model.decoders[0].size))
            for _ in range(len(self.model.decoders))
        ]
        return (tgt, tgt_mask, enc_out, cache)

    def get_input_names(self):
        return ['tgt', 'tgt_mask', 'memory'] \
            + ['cache_%d' % i for i in range(len(self.model.decoders))]

    def get_output_names(self):
        return ['y'] \
            + ['out_cache_%d' % i for i in range(len(self.model.decoders))]

    def get_dynamic_axes(self):
        ret = {
            'tgt': {
                0: 'tgt_batch',
                1: 'tgt_length'
            },
            'tgt_mask': {
                1: 'tgt_mask_length',
                2: 'tgt_mask_height'
            },
            'memory': {
                0: 'memory_batch',
                1: 'memory_length'
            }
        }
        ret.update({
            'cache_%d' % d: {
                0: 'cache_%d_batch' % d,
                1: 'cache_%d_length' % d
            }
            for d in range(len(self.model.decoders))
        })
        ret.update({
            'out_cache_%d' % d: {
                0: 'out_cache_%d_batch' % d,
                1: 'out_cache_%d_length' % d
            }
            for d in range(len(self.model.decoders))
        })
        return ret

    def get_model_config(self, path):
        return {
            "dec_type": "XformerDecoder",
            "model_path": os.path.join(path, f'{self.model_name}.onnx'),
            "n_layers": len(self.model.decoders),
            "odim": self.model.decoders[0].size
        }
