from typing import Optional

import torch
import torch.nn as nn

from espnet_onnx.utils.abs_model import AbsExportModel


class OnnxHiFiGANVocoder(nn.Module, AbsExportModel):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.model_name = "HiFiGANVocoder"

    def forward(
        self, c: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform inference.
        Args:
            c (torch.Tensor): Input tensor (T, in_channels), or (B, T, in_channels)
            g (Optional[Tensor]): Global conditioning tensor (global_channels, 1).
        Returns:
            Tensor: Output tensor (T ** upsample_factor, out_channels).
        """
        if g is not None:
            g = g.unsqueeze(0)

        if len(c.shape) == 3:
            c = self.model.forward(c.transpose(1, 2), g=g)
        else:
            c = self.model.forward(c.transpose(1, 0).unsqueeze(0), g=g)

        return c.squeeze(0).transpose(1, 0)

    def get_dummy_inputs(self):
        c = torch.randn(100, self.model.input_conv.in_channels)
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
