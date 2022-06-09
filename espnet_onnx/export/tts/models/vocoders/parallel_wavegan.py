
from typing import (
    Optional,
    Tuple
)

import math
import torch
import torch.nn as nn

from espnet_onnx.utils.abs_model import AbsExportModel


class OnnxPWGVocoder(nn.Module, AbsExportModel):
    def __init__(
        self,
        model,
        use_z=False,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.use_z = use_z
        self.aux_channels = model.aux_channels
        self.model_name = 'PWGVocoder'

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
            return ['c', 'z']
        else:
            return ['c']

    def get_output_names(self):
        return ['wav']

    def get_dynamic_axes(self):
        ret = {
            'c': {0: 'c_length'},
        }
        if self.use_z:
            ret.update({
                'z': {0: 'z_length'}
            })
        return ret

    def get_model_config(self, path):
        return {
            'vocoder_type': 'OnnxPWGVocoder',
            'model_path': str(path / f'{self.model_name}.onnx')
        }
