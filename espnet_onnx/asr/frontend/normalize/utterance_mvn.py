from typing import Tuple

import numpy as np
from typeguard import check_argument_types

from espnet_onnx.utils.config import Config
from espnet_onnx.utils.function import make_pad_mask, mask_fill


class UtteranceMVN:
    def __init__(self, config: Config):
        assert check_argument_types()
        self.norm_means = config.norm_means
        self.norm_vars = config.norm_vars
        self.eps = config.eps

    def extra_repr(self):
        return f"norm_means={self.norm_means}, norm_vars={self.norm_vars}"

    def __call__(
        self, x: np.ndarray, ilens: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward function

        Args:
            x: (B, L, ...)
            ilens: (B,)

        """
        return utterance_mvn(
            x,
            ilens,
            norm_means=self.norm_means,
            norm_vars=self.norm_vars,
            eps=self.eps,
        )


def utterance_mvn(
    x: np.ndarray,
    ilens: np.ndarray = None,
    norm_means: bool = True,
    norm_vars: bool = False,
    eps: float = 1.0e-20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply utterance mean and variance normalization

    Args:
        x: (B, T, D), assumed zero padded
        ilens: (B,)
        norm_means:
        norm_vars:
        eps:

    """
    if ilens is None:
        ilens = np.full([x.shape[0]], x.shape[1])
    ilens_ = ilens.reshape(-1, *[1 for _ in range(len(x.shape) - 1)])
    # Zero padding
    x = mask_fill(x, make_pad_mask(ilens, x, 1), 0.0)
    # mean: (B, 1, D)
    mean = x.sum(axis=1, keepdims=True) / ilens_

    if norm_means:
        x -= mean

        if norm_vars:
            var = np.power(x, 2).sum(axis=1, keepdims=True) / ilens_
            std = np.clip(np.sqrt(var), eps, None)
            x = x / np.sqrt(std)
        return x, ilens
    else:
        if norm_vars:
            y = x - mean
            y = mask_fill(y, make_pad_mask(ilens, y, 1), 0.0)
            var = np.power(y, 2).sum(axis=1, keepdims=True) / ilens_
            std = np.clip(np.sqrt(var), eps, None)
            x /= std
        return x, ilens
