import os

import torch
import torch.nn as nn
import numpy as np

from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8

from espnet_onnx.utils.function import make_pad_mask
from .abs_model import AbsModel


class CTC(nn.Module, AbsModel):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.log_softmax(self.model(x), dim=2)
    
    def get_dummy_inputs(self, enc_size):
        enc_out = torch.randn(1, 100, enc_size)
        return (enc_out,)

    def get_input_names(self):
        return ['x']

    def get_output_names(self):
        return ['ctc_out']

    def get_dynamic_axes(self):
        return {
            "x": {1: "ctc_in_length" },
            "ctc_out": {1: "ctc_out_length"}
        }

    def get_model_config(self, path):
        return {
            "model_path": os.path.join(path, "ctc.onnx")
        }
