from typing import List

import numpy as np
from typeguard import check_argument_types

from espnet_onnx.asr.frontend.default.logmel import LogMel
from espnet_onnx.asr.frontend.default.stft import Stft
from espnet_onnx.utils.config import Config


class DefaultFrontend:
    """Default frontend module.
    This class is based on espnet2.asr.frontend.default.DefaultFrontend

    Args:
        config (Config): configuration for default frontend
    """

    def __init__(
        self,
        config: Config,
        providers: List[str],
        use_quantized: bool = False,
    ):
        self.config = config
        self.stft = Stft(config.stft)
        self.logmel = LogMel(config.logmel)

    def __call__(self, inputs: np.ndarray, input_length: np.ndarray):
        assert check_argument_types()
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        input_stft, feats_lens = self.stft(inputs, input_length)

        # 2. [Option] Speech enhancement
        # Currently this is not supported.

        # 3. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real**2 + input_stft.imag**2

        # 4. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)
        return input_feats, feats_lens
