from typing import Optional, Tuple

import librosa
import numpy as np
from typeguard import check_argument_types

from espnet_onnx.utils.config import Config
from espnet_onnx.utils.function import make_pad_mask, mask_fill


class Stft:
    """STFT module."""

    def __init__(self, config: Config):
        assert check_argument_types()
        self.config = config

    def __call__(
        self, input: np.ndarray, ilens: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """STFT forward function.
        Args:
            input: (Batch, Nsamples)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2)
        """
        assert check_argument_types()
        stft_kwargs = dict(
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            center=self.config.center,
            window=self.config.window,
            pad_mode="reflect",
        )
        output = []
        # iterate over istances in a batch
        for i, instance in enumerate(input):
            stft = librosa.stft(input[i], **stft_kwargs)
            output.append(np.array(np.stack([stft.real, stft.imag], -1)))
        output = np.vstack(output).reshape(len(output), *output[0].shape)

        if not self.config.onesided:
            len_conj = self.n_fft - output.shape[1]
            conj = output[:, 1 : 1 + len_conj].flip(1)
            conj[:, :, :, -1].data *= -1
            output = np.concatenate([output, conj], 1)
        if self.config.normalized:
            output = output * (stft_kwargs["window"].shape[0] ** (-0.5))

        # output: (Batch, Freq, Frames, 2=real_imag)
        # -> (Batch, Frames, Freq, 2=real_imag)
        output = output.transpose(0, 2, 1, 3)
        if ilens is not None:
            if self.config.center:
                pad = self.config.n_fft // 2
                ilens = ilens + 2 * pad
            olens = (ilens - self.config.n_fft) // self.config.hop_length + 1
            output = mask_fill(output, make_pad_mask(olens, output, dim=1), 0.0)
        else:
            olens = None

        # create complex array
        output = output[..., 0] + output[..., 1] * 1j
        return output, olens
