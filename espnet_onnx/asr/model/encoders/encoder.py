from typing import (
    Tuple,
    List
)

import onnxruntime
import numpy as np

from espnet_onnx.asr.frontend.frontend import Frontend
from espnet_onnx.asr.frontend.global_mvn import GlobalMVN
from espnet_onnx.asr.frontend.utterance_mvn import UtteranceMVN
from espnet_onnx.utils.function import (
    make_pad_mask,
    mask_fill
)
from espnet_onnx.utils.config import Config


class Encoder:
    def __init__(
        self,
        encoder_config: Config,
        providers: List[str],
        use_quantized: bool = False,
    ):
        self.config = encoder_config
        if use_quantized:
            self.encoder = onnxruntime.InferenceSession(
                self.config.quantized_model_path,
                providers=providers
            )
        else:
            self.encoder = onnxruntime.InferenceSession(
                self.config.model_path,
                providers=providers
            )

        self.frontend = Frontend(self.config.frontend)
        if self.config.do_normalize:
            if self.config.normalize.type == 'gmvn':
                self.normalize = GlobalMVN(self.config.normalize)
            elif self.config.normalize.type == 'utterance_mvn':
                self.normalize = UtteranceMVN(self.config.normalize)

        # if self.config.do_preencoder:
        #     self.preencoder = Preencoder(self.config.preencoder)

        # if self.config.do_postencoder:
        #     self.postencoder = Postencoder(self.config.postencoder)

    def __call__(
        self, speech: np.ndarray, speech_length: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        # 1. Extract feature
        feats, feat_length = self.frontend(speech, speech_length)

        # 2. normalize with global MVN
        if self.config.do_normalize:
            feats, feat_length = self.normalize(feats, feat_length)

        # if self.config.do_preencoder:
        #     feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 3. forward encoder
        encoder_out, encoder_out_lens = \
            self.forward_encoder(feats, feat_length)
        encoder_out = self.mask_output(encoder_out, encoder_out_lens)

        # if self.config.do_postencoder:
        #     encoder_out, encoder_out_lens = self.postencoder(
        #         encoder_out, encoder_out_lens
        #     )

        return encoder_out, encoder_out_lens

    def mask_output(self, feats, feat_length):
        if self.config.is_vggrnn:
            feats = mask_fill(feats, make_pad_mask(feat_length, feats, 1), 0.0)
        return feats, feat_length

    def forward_encoder(self, feats, feat_length):
        encoder_out, encoder_out_lens = \
            self.encoder.run(["encoder_out", "encoder_out_lens"], {
                "feats": feats,
                "feats_length": feat_length
            })
        
        if self.config.enc_type == 'RNNEncoder':
            encoder_out = mask_fill(encoder_out, make_pad_mask(
                feat_length, encoder_out, 1), 0.0)

        return encoder_out, encoder_out_lens
