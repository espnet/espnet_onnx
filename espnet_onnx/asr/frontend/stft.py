from distutils.version import LooseVersion
from typing import Optional
from typing import Tuple
from typing import Union

from typeguard import check_argument_types
import librosa
import numpy as np

from espnet_onnx.utils.function import mask_fill, make_pad_mask
from .window import get_window

class Stft():
    def __init__(
        self,
        config
    ):
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
        bs = input.shape[0]
        window = get_window(self.config.window, self.config.win_length)
        stft_kwargs = dict(
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            center=self.config.center,
            window=window,
        )
        # pad the given window to n_fft
        n_pad_left = (self.config.n_fft - window.shape[0]) // 2
        n_pad_right = self.config.n_fft - window.shape[0] - n_pad_left
        stft_kwargs["window"] = np.hstack(
            [np.zeros(n_pad_left), window, np.zeros(n_pad_right)]
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
            output = mask_fill(output, make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None
        return output, olens
