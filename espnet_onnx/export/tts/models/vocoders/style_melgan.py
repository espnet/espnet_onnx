import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet_onnx.utils.abs_model import AbsExportModel


class OnnxStyleMelGANVocoder(nn.Module, AbsExportModel):
    def __init__(self, model, use_z=False, **kwargs):
        super().__init__()
        self.model = model
        self.in_channels = model.in_channels
        self.aux_channels = model.blocks[0].tade1.aux_conv[0].in_channels
        self.model_name = "StyleMelGANVocoder"

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        c = c.transpose(1, 0).unsqueeze(0)

        # prepare noise input
        noise_size = (
            1,
            self.in_channels,
            math.ceil(c.size(2) / self.model.noise_upsample_factor),
        )
        noise = torch.randn(*noise_size, dtype=torch.float)
        x = self.model.noise_upsample(noise)

        # NOTE(kan-bayashi): To remove pop noise at the end of audio, perform padding
        #    for feature sequence and after generation cut the generated audio. This
        #    requires additional computation but it can prevent pop noise.
        total_length = c.size(2) * self.model.upsample_factor
        c = F.pad(c, (0, x.size(2) - c.size(2)), "replicate")

        # This version causes pop noise.
        # x = x[:, :, :c.size(2)]

        for block in self.model.blocks:
            x, c = block(x, c)
        x = self.model.output_conv(x)[..., :total_length]

        return x.squeeze(0).transpose(1, 0)

    def get_dummy_inputs(self):
        c = torch.randn(100, self.aux_channels)
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
