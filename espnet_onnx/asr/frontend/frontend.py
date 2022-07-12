from typing import (
    Union,
    List
)
from pathlib import Path
from typeguard import check_argument_types

import numpy as np
import onnxruntime

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
    ):
        if config.frontend_type == 'default':
            self.frontend = DefaultFrontend(config, providers, use_quantized)
        elif config.frontend_type == 'hubert':
            self.frontend = HubertFrontend(config, providers, use_quantized)
        else:
            raise ValueError("Unknown frontend type")

    def __call__(self, inputs: np.ndarray, input_length: np.ndarray):
        assert check_argument_types()
        input_feats, feats_lens = self.frontend(inputs, input_length)
        return input_feats, feats_lens
    
    @staticmethod
    def get_frontend(tag_name, providers: list = ['CPUExecutionProvider'], use_quantized: bool = False):
        from espnet_onnx.utils.config import (
            get_config,
            get_tag_config
        )
        import os
        import glob
        tag_config = get_tag_config()
        if tag_name not in tag_config.keys():
            raise RuntimeError(f'Model path for tag_name "{tag_name}" is not set on tag_config.yaml.'
                                + 'You have to export to onnx format with `espnet_onnx.export.asr.export_asr.ModelExport`,'
                                + 'or have to set exported model path in tag_config.yaml.')
        config_file = glob.glob(os.path.join(tag_config[tag_name], 'config.*'))[0]
        config = get_config(config_file)
        return Frontend(config.encoder.frontend, providers, use_quantized)