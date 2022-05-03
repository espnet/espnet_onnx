from typing import (
    Tuple,
    List
)

from nara_wpe.wpe import (
    get_filter_matrix_conj_v5,
    perform_filter_operation
)
import numpy as np

from espnet_onnx.asr.frontend.default.mask_estimator import MaskEstimator
from espnet_onnx.utils.config import Config
from espnet_onnx.utils.function import (
    make_pad_mask,
    mask_fill
)


def wpe_one_iteration(Y, power, taps, delay):
    """WPE for one iteration
    Args:
        Y: Complex valued STFT signal with shape (..., C, T)
        power: : (..., T)
        taps: Number of filter taps
        delay: Delay as a guard interval, such that X does not become zero.
        eps:
        inverse_power (bool):
    Returns:
        enhanced: (..., C, T)
    """
    batch_freq_size = Y.shape[:-2]
    Y = Y.reshape(-1, *Y.shape[-2:])
    power = power.reshape(-1, power.shape[-1])
    inverse_power = 1 / np.clip(power, eps, None)

    filter_matrix_conj = get_filter_matrix_conj_v5(
        Y, inverse_power, taps, delay
    )
    enhanced = perform_filter_operation(
        Y, filter_matrix_conj, taps, delay
    )
    enhanced = enhanced.reshape(*batch_freq_size, *Y.shape[-2:])
    return enhanced


class DNN_WPE:
    def __init__(
        self,
        config: Config,
        providers: List[str],
        use_quantized: bool = False,
    ):
        self.iterations = config.iterations
        self.taps = config.taps
        self.delay = config.delay

        self.normalization = config.normalization
        self.use_dnn_mask = config.use_dnn_mask

        if self.use_dnn_mask:
            self.mask_est = MaskEstimator(
                config.mask_estimator, providers, use_quantized
            )

    def __call__(
        self, data: np.ndarray, ilens: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The forward function

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq or Some dimension of the feature vector

        Args:
            data: (B, C, T, F)
            ilens: (B,)
        Returns:
            data: (B, C, T, F)
            ilens: (B,)
        """
        # (B, T, C, F) -> (B, F, C, T)
        enhanced = data = data.transpose(0, 3, 2, 1)
        mask = None

        for _ in range(self.iterations):
            # Calculate power: (..., C, T)
            power = enhanced.real**2 + enhanced.imag**2
            if i == 0 and self.use_dnn_mask:
                # mask: (B, F, C, T)
                (mask,), _ = self.mask_est(enhanced, ilens)
                if self.normalization:
                    # Normalize along T
                    mask = mask / mask.sum(axis=-1)[..., None]
                # (..., C, T) * (..., C, T) -> (..., C, T)
                power = power * mask

            # Averaging along the channel axis: (..., C, T) -> (..., T)
            power = power.mean(axis=-2)

            # enhanced: (..., C, T) -> (..., C, T)
            enhanced = wpe_one_iteration(
                enhanced, power, self.taps, self.delay
            )
            enhanced = masked_fill(enhanced, make_pad_mask(ilens, enhanced.real), 0)

        # (B, F, C, T) -> (B, T, C, F)
        enhanced = enhanced.transpose(0, 3, 2, 1)
        if mask is not None:
            mask = mask.transpose(-1, -3)
        return enhanced, ilens, mask
