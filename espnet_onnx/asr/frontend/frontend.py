from typing import List

import numpy as np

from espnet_onnx.asr.frontend.default.default_frontend import DefaultFrontend
from espnet_onnx.asr.frontend.s3prl.hubert import HubertFrontend
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
        providers: List[str],
        use_quantized: bool = False,
        torch_input: bool = False,
    ):
        self.torch_input = torch_input
        if config.frontend_type == "default":
            self.frontend = DefaultFrontend(config, providers, use_quantized)
        elif config.frontend_type == "hubert":
            self.frontend = HubertFrontend(config, providers, use_quantized)
        else:
            raise ValueError("Unknown frontend type")

    def __call__(self, inputs: np.ndarray, input_length: np.ndarray):
        input_feats, feats_lens = self.frontend(inputs, input_length)
        return input_feats, feats_lens
