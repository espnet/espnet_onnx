import torch
import torch.nn as nn

from espnet_onnx.utils.abs_model import AbsExportModel


class OnnxMelGANVocoder(nn.Module, AbsExportModel):
    def __init__(self, model, use_z=False, **kwargs):
        super().__init__()
        self.model = model
        self.in_channels = model.melgan[1].in_channels
        self.model_name = "MelGANVocoder"

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        c = self.model.melgan(c.transpose(1, 0).unsqueeze(0))
        return c.squeeze(0).transpose(1, 0)

    def get_dummy_inputs(self):
        c = torch.randn(100, self.in_channels)
        return (c,)

    def get_input_names(self):
        return ["c"]

    def get_output_names(self):
        return ["wav"]

    def get_dynamic_axes(self):
        return {
            "c": {0: "c_length"},
        }

    def get_model_config(self, path):
        return {
            "vocoder_type": "OnnxVocoder",
            "model_path": str(path / f"{self.model_name}.onnx"),
        }
