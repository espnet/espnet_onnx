from typing import List

import numpy as np
import onnxruntime

from espnet_onnx.utils.config import Config


class Vocoder:
    def __init__(
        self,
        config: Config,
        providers: List[str],
        use_quantized: bool = False,
    ):
        self.config = config
        if use_quantized:
            self.model = onnxruntime.InferenceSession(
                self.config.quantized_model_path, providers=providers
            )
        else:
            self.model = onnxruntime.InferenceSession(
                self.config.model_path, providers=providers
            )

    def __call__(self, c: np.ndarray):
        input_dict = {"c": c}
        return self.model.run(["wav"], input_dict)[0]
