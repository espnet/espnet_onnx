from pathlib import Path
from typing import Tuple

import numpy as np
from typeguard import check_argument_types

from espnet_onnx.utils.function import make_pad_mask, mask_fill


class GlobalMVN:
    """Apply global mean and variance normalization

    Args:
        config.stats_file: npy file
        config.norm_means: Apply mean normalization
        config.norm_vars: Apply var normalization
        config.eps:
    """

    def __init__(self, config):
        assert check_argument_types()
        self.norm_means = config.norm_means
        self.norm_vars = config.norm_vars
        self.eps = config.eps
        stats_file = Path(config.stats_file)
        stats = np.load(stats_file)

        if isinstance(stats, np.ndarray):
            # Kaldi like stats
            count = stats[0].flatten()[-1]
            mean = stats[0, :-1] / count
            var = stats[1, :-1] / count - mean * mean
        else:
            # New style: Npz file
            count = stats["count"]
            sum_v = stats["sum"]
            sum_square_v = stats["sum_square"]
            mean = sum_v / count
            var = sum_square_v / count - mean * mean
        std = np.sqrt(np.maximum(var, config.eps))
        self.mean = mean
        self.std = std

    def extra_repr(self):
        return (
            f"stats_file={self.stats_file}, "
            f"norm_means={self.norm_means}, norm_vars={self.norm_vars}"
        )

    def __call__(
        self, x: np.ndarray, ilens: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward function
        Args:
            x: (B, L, ...)
            ilens: (B,)
        """
        mask = make_pad_mask(ilens, x, 1)

        # feat: (B, T, D)
        if self.norm_means:
            x -= self.mean
        x = mask_fill(x, mask, 0.0)

        if self.norm_vars:
            x /= self.std

        return x, ilens

    def inverse(
        self, x: np.ndarray, ilens: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward function
        Args:
            x: (B, L, ...)
            ilens: (B,)
        """
        mask = make_pad_mask(ilens, x, 1)

        if self.norm_vars:
            x *= self.std

        # feat: (B, T, D)
        if self.norm_means:
            x += self.mean
            x = mask_fill(x, mask, 0.0)

        return x, ilens
