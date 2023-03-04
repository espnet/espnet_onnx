import os

import torch
import torch.nn as nn
from espnet.nets.pytorch_backend.rnn.attentions import NoAtt

from espnet_onnx.export.layers.attention import require_tanh
from espnet_onnx.utils.abs_model import AbsExportModel


class PreDecoder(nn.Module, AbsExportModel):
    def __init__(self, model, idx):
        super().__init__()
        if isinstance(model, NoAtt):
            self.model = None
        else:
            self.model = model.mlp_enc
        self.model_name = f"predecoder_{idx}"
        self.apply_tanh = require_tanh(model)

    def require_onnx(self):
        return self.model is not None

    def forward(self, enc_h):
        if self.apply_tanh:
            return torch.tanh(self.model(enc_h))
        else:
            return self.model(enc_h)

    def get_dummy_inputs(self, *args):
        di = torch.randn(1, 100, self.model.in_features)
        return (di,)

    def get_input_names(self):
        return ["enc_h"]

    def get_output_names(self):
        return ["pre_compute_enc_h"]

    def get_dynamic_axes(self):
        return {"enc_h": {1: "enc_h_length"}}

    def get_model_config(self, path):
        return {
            "model_path": os.path.join(path, f"{self.model_name}.onnx"),
        }
