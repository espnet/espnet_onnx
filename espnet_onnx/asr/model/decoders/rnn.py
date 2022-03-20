from typing import List
from typing import Any
from typing import Tuple
from typing import Optional
from typing import Dict
from typing import Union

import numpy as np
import onnxruntime

from espnet_onnx.asr.scorer.interface import BatchScorerInterface
from espnet_onnx.asr.beam_search.hyps import TransducerHypothesis
from espnet_onnx.utils.function import (
    subsequent_mask,
    make_pad_mask
)
from espnet_onnx.utils.config import Config


class RNNDecoder(BatchScorerInterface):
    def __init__(
        self,
        config,
        use_quantized
    ):
        """
        Args:
            config (Config):
            use_quantized (bool): Flag to use quantized model
        """
        # predecoder
        self.predecoders = []
        for p in range(config.predecoder):
            if use_quantized:
                model_path = p.quantized_model_path
            else:
                model_path = p.model_path
            self.predecoders.append(
                onnxruntime.InferenceSession(model_path)
            )
            
        # decoder
        if use_quantized:
            self.decoder = onnxruntime.InferenceSession(
                config.quantized_model_path)
        else:
            self.decoder = onnxruntime.InferenceSession(config.model_path)
        
        # HP
        self.num_encs = len(self.predecoders)
        
        # predecoder
        self.decoder_output_names = self.get_decoder_output_names()
        
        
    
    def get_decoder_output_names(self):
        ret = []
        for d in self.decoder.get_inputs():
            ret.append(d.name)
        return ret

    def score(self, yseq, state, x):
        att_idx, z_list, c_list = state["workspace"]
        vy = yseq[-1][None, :]
        
        if self.num_encs == 1:
            x = [x]
            
        # pre compute states of attention.
        pre_compute_enc_h = []
        enc_h = []
        mask = []
        for idx in range(len(self.num_encs)):
            _pceh = self.predecoders[idx].run(
                ['pre_compute_enc_h'], {
                    'enc_h': x[idx]
                }
            )
            pre_compute_enc_h.append(_pceh)
            enc_h.append(x[idx])
            mask.append(np.where(make_pad_mask([x[idx].shape[0]])==1, -float('inf'), 0))
        
        logp, *status_lists = self.decoder(
            self.decoder_output_names,
            {
                'vy': vy,
                'x': x,
                'z_prev': state['z_prev'][0],
                'a_prev': state['a_prev'],
                'c_prev': state['c_prev'],
                'z_list': z_list,
                'c_list': c_list,
                'pre_compute_enc_h': pre_compute_enc_h,
                'enc_h': enc_h,
                'mask': mask,
            }
        )
        c_list, z_list, att_w = self.separate(status_lists)
        return (
            logp,
            dict(
                c_prev=c_list,
                z_prev=z_list,
                a_prev=att_w,
                workspace=(att_idx, z_list, c_list),
            ),
        )
        
        
        
        
        