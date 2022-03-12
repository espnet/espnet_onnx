from typing import Union
from pathlib import Path
from typeguard import check_argument_types

import numpy as np
import onnxruntime

from .stft import Stft
from .logmel import LogMel

from espnet_onnx.utils.config import Config


class Frontend:
    """Default frontend module.
    This class is based on espnet2.asr.frontend.default.DefaultFrontend

    Args:
        config (Config): configuration for default frontend
    """

    def __init__(
        self,
        config: Config,
    ):
        self.stft = Stft(config.stft)
        self.logmel = LogMel(config.logmel)

    def __call__(self, inputs: np.ndarray, input_length: np.ndarray):
        assert check_argument_types()
        # STFT
        input_stft, feats_lens = self.stft(inputs, input_length)

        # 3. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft[..., 0]**2 + input_stft[..., 1]**2

        # 4. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)
        return input_feats, feats_lens
