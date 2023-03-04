from typing import Optional

import torch
import torch.nn as nn

from espnet_onnx.utils.abs_model import AbsExportModel


class OnnxPWGVocoder(nn.Module, AbsExportModel):
    def __init__(self, model, pretrained=False, use_z=False, **kwargs):
        super().__init__()
        self.model = model
        self.use_z = use_z
        self.pretrained = pretrained
        self.aux_channels = model.aux_channels
        self.model_name = "PWGVocoder"

    def forward(
        self, c: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform inference.
        Args:
            c (torch.Tensor): Input tensor (T, in_channels).
            g (Optional[Tensor]): Global conditioning tensor (global_channels, 1).
        Returns:
            Tensor: Output tensor (T ** upsample_factor, out_channels).
        """
        if self.pretrained:
            feat_length = torch.ones(c[:, 0].shape).sum(dim=-1).type(torch.long)
            random_value = torch.randn(
                feat_length * self.model.upsample_factor
            ).unsqueeze(1)
            return self.model.inference(c, x=random_value)
        else:
            if z is not None:
                z = z.transpose(1, 0).unsqueeze(0)
            c = c.transpose(1, 0).unsqueeze(0)
            return self.model.forward(c, z).squeeze(0).transpose(1, 0)

    def get_dummy_inputs(self):
        c = torch.randn(100, self.aux_channels)

        if self.use_z:
            z = torch.randn(100, self.aux_channels)
            return (c, z)
        else:
            return (c,)

    def get_input_names(self):
        if self.use_z:
            return ["c", "z"]
        else:
            return ["c"]

    def get_output_names(self):
        return ["wav"]

    def get_dynamic_axes(self):
        ret = {
            "c": {0: "c_length"},
        }
        if self.use_z:
            ret.update({"z": {0: "z_length"}})
        return ret

    def get_model_config(self, path):
        return {
            "vocoder_type": "OnnxVocoder",
            "model_path": str(path / f"{self.model_name}.onnx"),
        }
