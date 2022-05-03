"""DNN beamformer module."""
from typing import (
    Tuple,
    List
)

import numpy as np

from espnet_onnx.asr.frontend.default.dnn_wpe import MaskEstimator
from espnet_onnx.utils.function import (
    mask_fill,
    make_pad_mask
)
from espnet_onnx.utils.config import Config


class DNN_Beamformer:
    """DNN mask based Beamformer

    Citation:
        Multichannel End-to-end Speech Recognition; T. Ochiai et al., 2017;
        https://arxiv.org/abs/1703.04783

    """

    def __init__(
        self,
        config: Config,
        providers: List[str],
        use_quantized: bool = False,
    ):
        self.mask = MaskEstimator(
            config.mask_estimator, providers, use_quantized
        )
        self.ref = AttentionReference(
            config.ref, providers, use_quantized)
        self.ref_channel = ref_channel

        self.nmask = config.nmask
        self.beamformer_type = config.beamformer_type

    def forward(
        self, data: np.ndarray, ilens: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The forward function

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq

        Args:
            data (np.ndarray): (B, T, C, F)
            ilens (np.ndarray): (B,)
        Returns:
            enhanced (np.ndarray): (B, T, F)
            ilens (np.ndarray): (B,)

        """

        def apply_beamforming(data, ilens, psd_speech, psd_noise):
            # u: (B, C)
            if self.ref_channel < 0:
                u, _ = self.ref(psd_speech, ilens)
            else:
                # (optional) Create onehot vector for fixed reference microphone
                u = np.zeros(*(data.shape[:-3] + (data.shape[-2],)))
                u[..., self.ref_channel].fill(1)

            ws = get_mvdr_vector(psd_speech, psd_noise, u)
            enhanced = apply_beamforming_vector(ws, data)

            return enhanced, ws

        # data (B, T, C, F) -> (B, F, C, T)
        data = data.transpose(0, 3, 2, 1)

        # mask: (B, F, C, T)
        masks, _ = self.mask(data, ilens)
        assert self.nmask == len(masks)

        if self.nmask == 2:  # (mask_speech, mask_noise)
            mask_speech, mask_noise = masks

            psd_speech = get_power_spectral_density_matrix(data, mask_speech)
            psd_noise = get_power_spectral_density_matrix(data, mask_noise)

            enhanced, ws = apply_beamforming(data, ilens, psd_speech, psd_noise)

            # (..., F, T) -> (..., T, F)
            enhanced = enhanced.transpose(0, 1, 3, 2)
            mask_speech = mask_speech.transpose(0, 3, 2, 1)
        else:  # multi-speaker case: (mask_speech1, ..., mask_noise)
            mask_speech = list(masks[:-1])
            mask_noise = masks[-1]

            psd_speeches = [
                get_power_spectral_density_matrix(data, mask) for mask in mask_speech
            ]
            psd_noise = get_power_spectral_density_matrix(data, mask_noise)

            enhanced = []
            ws = []
            for i in range(self.nmask - 1):
                psd_speech = psd_speeches.pop(i)
                # treat all other speakers' psd_speech as noises
                enh, w = apply_beamforming(
                    data, ilens, psd_speech, sum(psd_speeches) + psd_noise
                )
                psd_speeches.insert(i, psd_speech)

                # (..., F, T) -> (..., T, F)
                enh = enh.transpose(0, 1, 3, 2)
                mask_speech[i] = mask_speech[i].transpose(0, 3, 2, 1)
                enhanced.append(enh)
                ws.append(w)

        return enhanced, ilens, mask_speech


class AttentionReference:
    def __init__(
        self,
        config: Config,
        providers: List[str],
        use_quantized: bool = False,
    ):
        if use_quantized:
            self.session = onnxruntime.InferenceSession(
                self.config.quantized_model_path,
                providers=providers
            )
        else:
            self.session = onnxruntime.InferenceSession(
                self.config.model_path,
                providers=providers
            )

    def forward(
        self, psd_in: np.ndarray, ilens: np.ndarray, scaling: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The forward function

        Args:
            psd_in (np.ndarray): (B, F, C, C)
            ilens (np.ndarray): (B,)
            scaling (float):
        Returns:
            u (np.ndarray): (B, C)
            ilens (np.ndarray): (B,)
        """
        B, _, C = psd_in.shape[:3]
        assert psd_in.shape[2] == psd_in.shape[3], psd_in.shape
        # psd_in: (B, F, C, C)
        psd = mask_fill(psd_in, np.eye(C, dtype=bool), 0)
        
        # psd: (B, F, C, C) -> (B, C, F)
        psd = (psd.sum(axis=-1) / (C - 1)).transpose(0, 2, 1)

        # Calculate amplitude
        psd_feat = (psd.real**2 + psd.imag**2) ** 0.5

        # (B, C, F) -> (B, C, F2)
        u = self.session.run(
            ['u'], {'psd_feat': psd_feat }
        )[0]
        return u, ilens



def get_power_spectral_density_matrix(
    xs: np.ndarray, mask: np.ndarray, normalization=True, eps: float = 1e-15
) -> np.ndarray:
    """Return cross-channel power spectral density (PSD) matrix

    Args:
        xs (np.ndarray): (..., F, C, T)
        mask (np.ndarray): (..., F, C, T)
        normalization (bool):
        eps (float):
    Returns
        psd (np.ndarray): (..., F, C, C)

    """
    # outer product: (..., C_1, T) x (..., C_2, T) -> (..., T, C, C_2)
    psd_Y = np.einsum("...ct,...et->...tce", [xs, xs.conj()])

    # Averaging mask along C: (..., C, T) -> (..., T)
    mask = mask.mean(axis=-2)

    # Normalized mask along T: (..., T)
    if normalization:
        # If assuming the tensor is padded with zero, the summation along
        # the time axis is same regardless of the padding length.
        mask = mask / (mask.sum(axis=-1, keepdim=True) + eps)

    # psd: (..., T, C, C)
    psd = psd_Y * mask[..., None, None]
    # (..., T, C, C) -> (..., C, C)
    psd = psd.sum(axis=-3)

    return psd


def get_mvdr_vector(
    psd_s: np.ndarray,
    psd_n: np.ndarray,
    reference_vector: np.ndarray,
    eps: float = 1e-15,
) -> np.ndarray:
    """Return the MVDR(Minimum Variance Distortionless Response) vector:

        h = (Npsd^-1 @ Spsd) / (Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_s (np.ndarray): (..., F, C, C)
        psd_n (np.ndarray): (..., F, C, C)
        reference_vector (np.ndarray): (..., C)
        eps (float):
    Returns:
        beamform_vector (np.ndarray)r: (..., F, C)
    """
    # Add eps
    C = psd_n.shape[-1]
    eye = np.eye(C, dtype=psd_n.dtype)
    shape = [1 for _ in range(len(psd_n.shape) - 2)] + [C, C]
    eye = eye.reshape(*shape)
    psd_n += eps * eye

    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    numerator = np.einsum("...ec,...cd->...ed", [psd_n.inverse(), psd_s])
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (np.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = np.einsum("...fec,...c->...fe", [ws, reference_vector])
    return beamform_vector


def apply_beamforming_vector(
    beamform_vector: np.ndarray, mix: np.ndarray
) -> np.ndarray:
    # (..., C) x (..., C, T) -> (..., T)
    es = np.einsum("...c,...ct->...t", [beamform_vector.conj(), mix])
    return es
