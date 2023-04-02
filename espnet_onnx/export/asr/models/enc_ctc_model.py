import os

import torch
import torch.nn as nn

from espnet_onnx.utils.abs_model import AbsExportModel


class CombinedModel(nn.Module, AbsExportModel):
    def __init__(self, encoder, ctc):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc
        self.ctc.export_ids = True
        self.model_name = "encoder_ctc"

    def forward(self, feats):
        enc_out, _ = self.encoder(feats)
        if isinstance(enc_out, tuple):
            enc_out = enc_out[0]
        return self.ctc(enc_out)

    def get_dummy_inputs(self):
        return self.encoder.get_dummy_inputs()

    def get_input_names(self):
        return ["feats"]

    def get_output_names(self):
        return self.ctc.get_output_names()

    def get_dynamic_axes(self):
        ret = self.ctc.get_dynamic_axes()
        ret.pop("x")
        ret.update(self.encoder.get_dynamic_axes())
        return ret

    def get_model_config(self, path):
        return {
            "model_path": os.path.join(path, f"{self.model_name}.onnx"),
        }

