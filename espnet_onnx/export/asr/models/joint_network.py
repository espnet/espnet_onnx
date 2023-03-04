import os

import torch
import torch.nn as nn

from espnet_onnx.utils.abs_model import AbsExportModel


class JointNetwork(nn.Module, AbsExportModel):
    def __init__(self, model, search_type):
        super().__init__()
        self.model = model
        self.search_type = search_type
        self.model_name = "joint_network"

    def forward(self, enc_out, dec_out):
        return self.model(enc_out, dec_out)

    def get_dummy_inputs(self):
        if self.search_type in ("default", "greedy"):
            enc_out = torch.randn(self.model.lin_enc.in_features)
            dec_out = torch.randn(self.model.lin_dec.in_features)
        else:
            enc_out = torch.randn(1, self.model.lin_enc.in_features)
            dec_out = torch.randn(1, self.model.lin_dec.in_features)
        return (enc_out, dec_out)

    def get_input_names(self):
        return ["enc_out", "dec_out"]

    def get_output_names(self):
        return ["joint_out"]

    def get_dynamic_axes(self):
        return {"enc_out": {0: "enc_out_length"}, "dec_out": {0: "dec_out_length"}}

    def get_model_config(self, path):
        return {"model_path": os.path.join(path, f"{self.model_name}.onnx")}
