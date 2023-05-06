from typing import List

import numpy as np
import onnxruntime

from espnet_onnx.utils.config import Config


class HubertFrontend:
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
        if use_quantized:
            self.frontend = onnxruntime.InferenceSession(
                self.config.quantized_model_path, providers=providers
            )
        else:
            self.frontend = onnxruntime.InferenceSession(
                self.config.model_path, providers=providers
            )
        self.output_names = ["feats", "feats_lens"]

    def __call__(self, inputs: np.ndarray, input_length: np.ndarray):
        input_dic = {
            "wav": inputs,
        }
        feats, feat_length = self.frontend.run(self.output_names, input_dic)
        return feats, feat_length
