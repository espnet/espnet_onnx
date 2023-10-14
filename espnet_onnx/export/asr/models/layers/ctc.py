import os

import torch
import torch.nn as nn

from espnet_onnx.utils.abs_model import AbsExportModel


class CTC(nn.Module, AbsExportModel):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model_name = "ctc"

    def forward(self, x):
        return torch.log_softmax(self.model(x), dim=2)

    def get_dummy_inputs(self, enc_size):
        enc_out = torch.randn(1, 100, enc_size)
        return (enc_out,)

    def get_input_names(self):
        return ["x"]

    def get_output_names(self):
        return ["ctc_out"]

    def get_dynamic_axes(self):
        return {"x": {1: "ctc_in_length"}, "ctc_out": {1: "ctc_out_length"}}

    def get_model_config(self, path):
        return {"model_path": os.path.join(path, f"{self.model_name}.onnx")}
