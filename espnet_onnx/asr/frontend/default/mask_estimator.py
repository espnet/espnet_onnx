from typing import (
    Tuple,
    List
)

import numpy as np

from espnet_onnx.utils.function import make_pad_mask
from espnet_onnx.utils.config import Config


class MaskEstimator:
    def __init__(
        self,
        config: Config,
        providers: List[str],
        use_quantized: bool = False,
    ):
        if use_quantized:
            self.estimator = onnxruntime.InferenceSession(
                self.config.quantized_model_path,
                providers=providers
            )
        else:
            self.estimator = onnxruntime.InferenceSession(
                self.config.model_path,
                providers=providers
            )
        self.output_names = sorted([d.name for d in self.estimator.get_outputs()])


    def forward(
        self, xs: np.ndarray, ilens: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        """The forward function

        Args:
            xs: (B, F, C, T)
            ilens: (B,)
        Returns:
            hs (np.ndarray): The hidden vector (B, F, C, T)
            masks: A tuple of the masks. (B, F, C, T)
            ilens: (B,)
        """
        assert xs.shape[0] == ilens.shape[0], (xs.shape[0], ilens.shape[0])
        _, _, C, input_length = xs.shape
        # (B, F, C, T) -> (B, C, T, F)
        xs = xs.transpose(0, 2, 3, 1)

        # Calculate amplitude: (B, C, T, F) -> (B, C, T, F)
        xs = (xs.real**2 + xs.imag**2) ** 0.5
        # xs: (B, C, T, F) -> xs: (B * C, T, F)
        xs = xs.contiguous().reshape(-1, xs.shape[-2], xs.shape[-1])
        # ilens: (B,) -> ilens_: (B * C)
        ilens_ = ilens[:, None].expand(-1, C).contiguous().view(-1)

        # Use onnx model to get masks
        _masks = self.estimator(
            self.output_names,
            self.get_input_dict(xs, ilens_)
        )
        masks = []
        
        for mask in _masks:
            m = mask_fill(
                mask,
                make_pad_mask(ilens, mask, dim=2),
                0
            )
            # m: (B*C, T, F), and B = 1
            m = m[None, :].transpose(0, 3, 1, 2)
            masks.append(m)
            
        return tuple(masks), ilens

    def get_input_dict(self, xs, feats_len):
        return {
            'feat': xs,
            'feats_length': feats_len
        }