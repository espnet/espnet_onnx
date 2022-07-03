import os

import torch
import torch.nn as nn
import numpy as np

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from ..multihead_att import OnnxMultiHeadedAttention

from espnet_onnx.utils.function import subsequent_mask
from ..language_models.embed import Embedding
from espnet_onnx.utils.abs_model import AbsExportModel
from espnet_onnx.utils.torch_function import MakePadMask
from ..decoder_layer import OnnxDecoderLayer


class XformerDecoder(nn.Module, AbsExportModel):
    def __init__(self, model, max_seq_len=512, optimize=False, **kwargs):
        super().__init__()
        self.embed = Embedding(model.embed, max_seq_len)
        self.model = model
        self.optimize = optimize
        self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        # replace multihead attention module into customized module.
        for i,d in enumerate(self.model.decoders):
            # d is DecoderLayer
            if isinstance(d.self_attn, MultiHeadedAttention):
                d.self_attn = OnnxMultiHeadedAttention(d.self_attn)
            if isinstance(d.src_attn, MultiHeadedAttention):
                d.src_attn = OnnxMultiHeadedAttention(d.src_attn)
            self.model.decoders[i] = OnnxDecoderLayer(d)
                
        self.num_heads = model.decoders[0].self_attn.h
        self.hidden_size = model.decoders[0].self_attn.linear_out.out_features
        self.model_name = 'xformer_decoder'
    
    def prepare_mask(self, mask):
        if self.optimize:
            if len(mask.shape) == 2:
                mask = mask[:, None, None, :]
            elif len(mask.shape) == 3:
                mask = mask[:, None, :]
        mask = 1 - mask
        return mask * -10000.0

    def forward(self, tgt, mask_or_length, memory, cache):
        if self.optimize:
            mask = self.make_pad_mask(mask_or_length) # (B, T)
            mask[:, -1] = 1
        else:
            mask = mask_or_length
        
        x = self.embed(tgt)
        mask = self.prepare_mask(mask)
        new_cache = []
        for c, decoder in zip(cache, self.model.decoders):
            x, mask = decoder(
                x, mask, memory, None, c
            )
            new_cache.append(x)
            
        y = self.model.after_norm(x[:, -1])
        y = torch.log_softmax(self.model.output_layer(y), dim=-1)
        return y, new_cache

    def get_dummy_inputs(self, enc_size):
        tgt = torch.LongTensor([0, 1]).unsqueeze(0)
        if self.optimize:
            mask_or_length = torch.LongTensor([tgt.size(1)])
        else:
            ys_mask = tgt != 0
            mask_or_length = torch.from_numpy(subsequent_mask(ys_mask.shape[-1])[None, :]).type(torch.long)
        enc_out = torch.randn(1, 100, enc_size)
        cache = [
            torch.zeros((1, 1, self.model.decoders[0].size))
            for _ in range(len(self.model.decoders))
        ]
        return (tgt, mask_or_length, enc_out, cache)

    def is_optimizable(self):
        return True

    def get_input_names(self):
        return ['tgt', 'mask_or_length', 'memory'] \
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
            'mask_or_length': {
                0: 'mol_batch',
            },
            'memory': {
                0: 'memory_batch',
                1: 'memory_length'
            }
        }
        if not self.optimize:
            ret.update({
                'mask_or_length': {
                    0: 'mol_batch',
                    1: 'mol_height',
                    2: 'mol_width'
                }
            })
        ret.update({
            'cache_%d' % d: {
                0: 'cache_%d_batch' % d,
                1: 'cache_%d_length' % d
            }
            for d in range(len(self.model.decoders))
        })
        return ret

    def get_model_config(self, path):
        return {
            "dec_type": "XformerDecoder",
            "model_path": os.path.join(path, f'{self.model_name}.onnx'),
            "n_layers": len(self.model.decoders),
            "odim": self.model.decoders[0].size,
            "optimized": self.optimize
        }
