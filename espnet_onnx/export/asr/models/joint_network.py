import os

import numpy as np
import torch
import torch.nn as nn

from .abs_model import AbsModel


class JointNetwork(nn.Module, AbsModel):
    def __init__(self, model, templature):
        super().__init__()
        self.model = model
        self.templature = templature

    def forward(self, enc_out, dec_out):
        return torch.log_softmax(
            self.model(enc_out, dec_out) / self.templature,
            dim=-1
        )

    def get_dummy_inputs(self):
        enc_out = torch.randn(1, 50, 1, self.model.lin_enc.in_feature)
        dec_out = torch.randn(1, 1, 3, self.model.lin_dec.in_feature)
        return (enc_out, dec_out)

    def get_input_names(self):
        return ['enc_out', 'dec_out']

    def get_output_names(self):
        return ['joint_out']

    def get_dynamic_axes(self):
        return {
            "enc_out": {1: "joint_enc_T"},
            "dec_out": {2: "joint_dec_U"},
            "joint_out": {
                1: "joint_out_T",
                2: "joint_out_U"
            }
        }

    def get_model_config(self, path):
        return {
            "model_path": os.path.join(path, "joint_network.onnx")
        }
