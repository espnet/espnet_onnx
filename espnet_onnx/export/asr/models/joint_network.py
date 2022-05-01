import os

import numpy as np
import torch
import torch.nn as nn

from .abs_model import AbsModel


class JointNetwork(nn.Module, AbsModel):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, enc_out, dec_out):
        return torch.log_softmax(
            self.model(enc_out, dec_out),
            dim=-1
        )

    def get_dummy_inputs(self):
        enc_out = torch.randn(self.model.lin_enc.in_features)
        dec_out = torch.randn(self.model.lin_dec.in_features)
        return (enc_out, dec_out)

    def get_input_names(self):
        return ['enc_out', 'dec_out']

    def get_output_names(self):
        return ['joint_out']

    def get_dynamic_axes(self):
        return {}

    def get_model_config(self, path):
        return {
            "model_path": os.path.join(path, "joint_network.onnx")
        }
