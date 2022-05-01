import os

import torch
import torch.nn as nn

from espnet2.asr.transducer.transducer_decoder import TransducerDecoder

from espnet_onnx.utils.function import subsequent_mask
from ..language_models.lm import Embedding
from ..abs_model import AbsModel


class TransducerDecoder(nn.Module, AbsModel):
    def __init__(self, model):
        super().__init__()
        self.embed = Embedding(model.embed)
        self.decoder = model.decoder
        self.dlayers = model.dlayers
        self.dunits = model.dunits
        self.dtype = model.dtype

    def forward(self, labels, h_cache, c_cache):
        # embed and rnn-forward
        sequence = self.embed(labels)
        h_next_list = []
        c_next_list = []
        if self.dtype == "lstm":
            for i in range(self.dlayers):
                sequence, (_h, _c) = self.decoder[i](
                    sequence,
                    hx=(h_cache[i], c_cache[i]) 
                )
                h_next_list.append(_h)
                c_next_list.append(_c)
        else:
            for i in range(self.dlayers):
                sequence, _h = self.decoder[i](
                    sequence, hx=h_cache[i]
                )
                h_next_list.append(_h)

        return sequence, h_next_list, c_next_list 

    def get_dummy_inputs(self, enc_size):
        labels = torch.LongTensor([0, 1]).unsqueeze(0)
        h_cache = [
            torch.randn(1, 1, self.dunits)
            for _ in range(self.dlayers)
        ]
        c_cache = [
            torch.randn(1, 1, self.dunits)
            for _ in range(self.dlayers)
        ]
        return labels, h_cache, c_cache

    def get_input_names(self):
        return ['labels'] \
            + [f'h_cache_{i}' for i in range(self.dlayers)] \
            + [f'c_cache_{i}' for i in range(self.dlayers)]

    def get_output_names(self):
        return ['sequence'] \
            + [f'out_h_cache_{i}' for i in range(self.dlayers)] \
            + [f'out_c_cache_{i}' for i in range(self.dlayers)]
            
    def get_dynamic_axes(self):
        ret = {
            'labels': {
                0: 'labels_batch',
                1: 'labels_length'
            }
        }
        for i in range(self.dlayers):
            ret.update({
                f'h_cache_{i}': {
                    1: f'h_cache_{i}_length'
                },
                f'c_cache_{i}': {
                    1: f'c_cache_{i}_length'
                },
                f'out_h_cache_{i}': {
                    1: f'out_h_cache_{i}_length'
                },
                f'out_c_cache_{i}': {
                    1: f'out_c_cache_{i}_length'
                }
            })
        return ret

    def get_model_config(self, path):
        file_name = os.path.join(path, 'decoder.onnx')
        return {
            "dec_type": "TransducerDecoder",
            "model_path": file_name,
            "n_layers": self.dlayers,
            "odim": self.dunits,
            "dtype": self.dtype
        }
