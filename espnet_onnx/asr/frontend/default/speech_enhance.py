from typing import (
    List,
    Optional,
    Tuple,
    Union
)

import numpy as np

from espnet_onnx.asr.frontend.default.beamformer import DNN_Beamformer
from espnet_onnx.asr.frontend.default.dnn_wpe import DNN_WPE
from espnet_onnx.utils.config import Config

class SpeechEnhance:
    def __init__(
        self,
        config: Config,
        providers: List[str],
        use_quantized: bool = False,
    ):
        self.config = config

        if self.use_wpe:
            if self.use_dnn_mask_for_wpe:
                # Use DNN for power estimation
                # (Not observed significant gains)
                iterations = 1
            else:
                # Performing as conventional WPE, without DNN Estimator
                iterations = 2

            self.wpe = DNN_WPE(config.wpe, providers, use_quantized)
            
        else:
            self.wpe = None

        if self.use_beamformer:
            self.beamformer = DNN_Beamformer(
                btype=btype,
                bidim=idim,
                bunits=bunits,
                bprojs=bprojs,
                blayers=blayers,
                bnmask=bnmask,
                dropout_rate=bdropout_rate,
                badim=badim,
                ref_channel=ref_channel,
            )
        else:
            self.beamformer = None

    def forward(
        self, x: np.ndarray, ilens: Union[np.ndarray, List[int]]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        assert len(x) == len(ilens), (len(x), len(ilens))
        # (B, T, F, complex) or (B, T, C, F, complex)
        if len(x.shape) not in (3, 4):
            raise ValueError(f"Input dim must be 3 or 4: {len(x.shape)}")
        if isinstance(ilens, list):
            ilens = np.array(ilens)

        mask = None
        h = x
        if len(h.shape) == 4:

            # 1. WPE
            if self.config.use_wpe:
                # h: (B, T, C, F) -> h: (B, T, C, F)
                h, ilens, mask = self.wpe(h, ilens)

            # 2. Beamformer
            if use_beamformer:
                # h: (B, T, C, F) -> h: (B, T, F)
                h, ilens, mask = self.beamformer(h, ilens)

        return h, ilens, mask

