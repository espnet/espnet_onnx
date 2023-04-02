from typing import List, Tuple

import onnxruntime
import numpy as np

from espnet_onnx.asr.frontend.frontend import Frontend
from espnet_onnx.asr.frontend.normalize.global_mvn import GlobalMVN
from espnet_onnx.asr.frontend.normalize.utterance_mvn import UtteranceMVN


class CombinedModel:
    def __init__(self, config, encoder_config, providers: List[str], use_quantized=False):
        if use_quantized:
            self.session = onnxruntime.InferenceSession(
                config.quantized_model_path, providers=providers
            )
        else:
            self.session = onnxruntime.InferenceSession(
                config.model_path, providers=providers
            )
        
        # check output
        output_names = [o.name for o in self.session.get_outputs()]
        if "ctc_ids" not in output_names or "ctc_probs" not in output_names:
            raise RuntimeError("Combined model does not have ctc_ids or ctc_probs output. \n" \
                               "You should export your model with export_config['combine_ctc']=True or export_config['export_ids']=True")

        self.frontend = Frontend(encoder_config.frontend, providers, use_quantized)
        if encoder_config.do_normalize:
            if encoder_config.normalize.type == "gmvn":
                self.normalize = GlobalMVN(encoder_config.normalize)
            elif encoder_config.normalize.type == "utterance_mvn":
                self.normalize = UtteranceMVN(encoder_config.normalize)
        
        self.encoder_config = encoder_config

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
        if self.encoder_config.do_normalize:
            feats, feat_length = self.normalize(feats, feat_length)

        # 3. forward encoder and ctc
        ctc_probs, ctc_ids = self.forward_encoder_ctc(feats)
        return ctc_probs, ctc_ids 

    def forward_encoder_ctc(self, feats):
        ctc_probs, ctc_ids = self.session.run(
            ["ctc_probs", "ctc_ids"], {"feats": feats}
        )

        return ctc_probs, ctc_ids 

