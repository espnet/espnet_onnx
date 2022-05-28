import os

import torch
import torch.nn as nn

from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention

from espnet_onnx.utils.function import subsequent_mask
from espnet_onnx.utils.abs_model import AbsExportModel
from .embed import Embedding
from ..encoder_layer import OnnxEncoderLayer
from ..multihead_att import OnnxMultiHeadedAttention
from espnet_onnx.utils.torch_function import MakePadMask


class TransformerLM(nn.Module, AbsExportModel):
    def __init__(self, model, max_seq_len=512, optimize_lm=False, **kwargs):
        super().__init__()
        self.optimize_lm = optimize_lm
        self.embed = Embedding(model.embed, max_seq_len)
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        # replace multihead attention module into customized module.
        for i, d in enumerate(self.encoder.encoders):
            # d is EncoderLayer
            if isinstance(d.self_attn, MultiHeadedAttention):
                d.self_attn = OnnxMultiHeadedAttention(d.self_attn)
            self.encoder.encoders[i] = OnnxEncoderLayer(d)
            
        self.num_heads = self.encoder.encoders[0].self_attn.h
        self.hidden_size = self.encoder.encoders[0].self_attn.linear_out.out_features
    
    def prepare_mask(self, mask):
        if self.optimize_lm:
            if len(mask.shape) == 2:
                mask = mask[:, None, None, :]
            elif len(mask.shape) == 3:
                mask = mask[:, None, :]
        mask = 1 - mask
        return mask * -10000.0

    def forward(self, y, tgt_length_or_mask, cache):
        if self.optimize_lm:
            mask = self.make_pad_mask(tgt_length_or_mask) # (B, T)
        else:
            mask = tgt_length_or_mask
        
        xs = self.embed(y)
        # forward_one_step of Encoder
        if isinstance(self.encoder.embed, Conv2dSubsampling):
            xs, mask = self.encoder.embed(xs, mask)
        else:
            xs = self.encoder.embed(xs)
            
        new_cache = []
        mask = self.prepare_mask(mask)
        for c, e in zip(cache, self.encoder.encoders):
            xs, mask = e(xs, mask)
            new_cache.append(torch.cat([c, xs], dim=1))

        if self.encoder.normalize_before:
            xs = self.encoder.after_norm(xs)

        h = self.decoder(xs[:, -1])
        return h, new_cache

    def get_dummy_inputs(self):
        tgt = torch.LongTensor([0, 1]).unsqueeze(0)
        if self.optimize_lm:
            mask_or_length = torch.LongTensor([tgt.size(1)])
        else:
            ys_mask = tgt != 0
            mask_or_length = torch.from_numpy(subsequent_mask(ys_mask.shape[-1])[None, :])
        cache = [
            torch.zeros((1, 1, self.encoder.encoders[0].size))
            for _ in range(len(self.encoder.encoders))
        ]
        return (tgt, mask_or_length, cache)

    def is_optimizable(self):
        return True

    def get_input_names(self):
        return ['tgt', 'mask_or_length'] \
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
            'mask_or_length': {
                0: 'mol_batch',
            }
        }
        if not self.optimize_lm:
            ret.update({
                'mask_or_length': {
                    1: 'mol_height',
                    2: 'mol_width'
                }
            })
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
            "optimize_lm": self.optimize_lm,
            "model_path": os.path.join(path, "lm.onnx"),
            "lm_type": "TransformerLM",
            "odim": self.encoder.encoders[0].size,
            "nlayers": len(self.encoder.encoders)
        }
