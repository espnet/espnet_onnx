import os

import torch
import torch.nn as nn

from espnet_onnx.utils.abs_model import AbsExportModel


class CTC(nn.Module, AbsExportModel):
    def __init__(self, model, export_ids = False, **kwargs):
        super().__init__()
        self.model = model
        self.model_name = "ctc"
        self.export_ids = export_ids

    def forward(self, x):
        ctc_lsf = torch.log_softmax(self.model(x), dim=2)
        if self.export_ids:
            ctc_probs, ctc_ids = torch.exp(ctc_lsf).max(dim=-1)
            return ctc_lsf, ctc_probs, ctc_ids
        else:
            return ctc_lsf

    def get_dummy_inputs(self, enc_size):
        enc_out = torch.randn(1, 100, enc_size)
        return (enc_out,)

    def get_input_names(self):
        return ["x"]

    def get_output_names(self):
        output_names = ["ctc_out"]
        if self.export_ids:
            output_names += ["ctc_probs", "ctc_ids"]
        return output_names

    def get_dynamic_axes(self):
        ret = {"x": {1: "ctc_in_length"}, "ctc_out": {1: "ctc_out_length"}}
        if self.export_ids:
            ret["ctc_probs"] = {1: "ctc_probs_length"}
            ret["ctc_ids"] = {1: "ctc_ids_length"}
        return ret

    def get_model_config(self, path):
        return {
            "model_path": os.path.join(path, f"{self.model_name}.onnx"),
            "export_ids": self.export_ids,
        }
