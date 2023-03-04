import warnings
from typing import List

import numpy as np
import torch

from espnet_onnx.asr.frontend.s3prl.hubert import HubertFrontend
from espnet_onnx.utils.config import Config


class TorchFrontend:
    """Frontend module that can be used with training.

    Args:
        config (Config): configuration for default frontend
    """

    def __init__(
        self,
        config: Config,
        providers: List[str] = ["CUDAExecutionProvider"],
        use_quantized: bool = False,
    ):
        if providers != ["CUDAExecutionProvider"]:
            warnings.warn(
                "Currently TorchFrontend supports only GPU input."
                + "Please check your device and providers option."
            )
        if use_quantized:
            warnings.warn(
                "Using quantized model on GPU may cause performance degradation."
            )

        if config.frontend_type == "default":
            raise ValueError("Use the original frontend.")
        elif config.frontend_type == "hubert":
            hubert = HubertFrontend(config, providers, use_quantized)
            self.frontend = hubert.frontend
        else:
            raise ValueError("Unknown frontend type")
        self.binding = self.frontend.io_binding()

    def __call__(
        self, inputs: torch.Tensor, input_length: torch.Tensor, output_shape: tuple
    ):
        device = str(inputs.device)
        input_tensor = inputs.contiguous()
        if device == "cpu":
            raise ValueError("The `inputs` tensor must be on GPU.")

        device_id = input_tensor.get_device()
        output_tensor = torch.empty(
            output_shape, dtype=torch.float32, device=f"cuda:{device_id}"
        ).contiguous()
        output_lens_tensor = torch.empty(
            (output_shape[0],), dtype=torch.int64, device=f"cuda:{device_id}"
        ).contiguous()
        self.binding.bind_input(
            name="wav",
            device_type="cuda",
            device_id=device_id,
            element_type=np.float32,
            shape=tuple(input_tensor.shape),
            buffer_ptr=input_tensor.data_ptr(),
        )
        self.binding.bind_output(
            name="feats",
            device_type="cuda",
            device_id=device_id,
            element_type=np.float32,
            shape=output_shape,
            buffer_ptr=output_tensor.data_ptr(),
        )
        self.binding.bind_output(
            name="feats_lens",
            device_type="cuda",
            device_id=device_id,
            element_type=np.int64,
            shape=(output_shape[0],),
            buffer_ptr=output_lens_tensor.data_ptr(),
        )

        self.frontend.run_with_iobinding(self.binding)
        return output_tensor, output_lens_tensor

    @staticmethod
    def get_frontend(
        tag_name,
        providers: list = ["CPUExecutionProvider"],
        use_quantized: bool = False,
        torch_input: bool = False,
    ):
        import glob
        import os

        from espnet_onnx.utils.config import get_config, get_tag_config

        tag_config = get_tag_config()
        if tag_name not in tag_config.keys():
            raise RuntimeError(
                f'Model path for tag_name "{tag_name}" is not set on tag_config.yaml.'
                + "You have to export to onnx format with `espnet_onnx.export.asr.export_asr.ModelExport`,"
                + "or have to set exported model path in tag_config.yaml."
            )
        config_file = glob.glob(os.path.join(tag_config[tag_name], "config.*"))[0]
        config = get_config(config_file)
        return TorchFrontend(config.encoder.frontend, providers, use_quantized)
