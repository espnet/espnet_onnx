from typing import Tuple

import librosa
import numpy as np

from espnet_onnx.utils.config import Config
from espnet_onnx.utils.function import make_pad_mask, mask_fill


class LogMel:
    """Convert STFT to fbank feats
    The arguments is same as librosa.filters.mel
    Args:
        config.fs: number > 0 [scalar] sampling rate of the incoming signal
        config.n_fft: int > 0 [scalar] number of FFT components
        config.n_mels: int > 0 [scalar] number of Mel bands to generate
        config.fmin: float >= 0 [scalar] lowest frequency (in Hz)
        config.fmax: float >= 0 [scalar] highest frequency (in Hz).
            If `None`, use `fmax = fs / 2.0`
        config.htk: use HTK formula instead of Slaney
    """

    def __init__(self, config: Config):
        fmin = 0 if config.fmin is None else config.fmin
        fmax = fs / 2 if config.fmax is None else config.fmax
        _mel_options = dict(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=config.htk,
        )
        self.mel_options = _mel_options
        self.log_base = config.log_base
        melmat = librosa.filters.mel(**_mel_options)
        self.melmat = melmat.T

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.mel_options.items())

    def __call__(
        self,
        feat: np.ndarray,
        ilens: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        mel_feat = np.matmul(feat, self.melmat)
        mel_feat = np.clip(mel_feat, 1e-10, float("inf"))

        if self.log_base is None:
            logmel_feat = np.log(mel_feat)
        elif self.log_base == 2.0:
            logmel_feat = np.log2(mel_feat)
        elif self.log_base == 10.0:
            logmel_feat = np.log10(mel_feat)
        else:
            logmel_feat = np.log(mel_feat) / np.log(self.log_base)

        # Zero padding
        logmel_feat = mask_fill(logmel_feat, make_pad_mask(ilens, logmel_feat, 1), 0.0)
        return logmel_feat, ilens
