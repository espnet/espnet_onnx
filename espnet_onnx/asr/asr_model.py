from pathlib import Path
from typing import List, Optional, Tuple, Union
from itertools import groupby
import logging

import numpy as np
from typeguard import check_argument_types

from espnet_onnx.asr.abs_asr_model import AbsASRModel
from espnet_onnx.asr.beam_search.hyps import Hypothesis


class Speech2Text(AbsASRModel):
    """Wrapper class for espnet2.asr.bin.asr_infer.Speech2Text"""

    def __init__(
        self,
        tag_name: str = None,
        model_dir: Union[Path, str] = None,
        providers: List[str] = ["CPUExecutionProvider"],
        use_quantized: bool = False,
    ):
        assert check_argument_types()
        self._check_argument(tag_name, model_dir)
        self._load_config()

        # check onnxruntime version and providers
        self._check_ort_version(providers)

        # check if model is exported for streaming.
        if self.config.encoder.enc_type == "ContextualXformerEncoder":
            raise RuntimeError(
                "Onnx model is built for streaming. Use StreamingSpeech2Text instead."
            )

        # Check quantization flag
        self._check_flags(use_quantized)

        # check if tag_name is exported with combined model.
        if "combined_model" in self.config.keys():
            self.greedy_ctc_decode = True
            self._build_model_combined(providers, use_quantized)
            logging.info("The encoder and ctc is combined during export. You can run greedy ctc decoding with greedy_ctc()")
        else:
            self.greedy_ctc_decode = False
            self._build_model(providers, use_quantized)

        if self.config.transducer.use_transducer_decoder:
            self.start_idx = 1
            self.last_idx = None
        else:
            self.start_idx = 1
            self.last_idx = -1

    def __call__(
        self, speech: np.ndarray
    ) -> List[Tuple[Optional[str], List[str], List[int], Union[Hypothesis],]]:
        """Inference
        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp
        """
        assert check_argument_types()

        # check dtype
        if speech.dtype != np.float32:
            speech = speech.astype(np.float32)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech[np.newaxis, :]
        # lengths: (1,)
        lengths = np.array([speech.shape[1]]).astype(np.int64)

        if self.greedy_ctc_decode:
            return self._greedy_ctc(speech, lengths)
        else:
            return self._ar_decode(speech, lengths)

    def _ar_decode(self, speech: np.ndarray, lengths: np.ndarray):
        # b. Forward Encoder
        enc, _ = self.encoder(speech=speech, speech_length=lengths)
        if isinstance(enc, tuple):
            enc = enc[0]
        assert len(enc) == 1, len(enc)

        nbest_hyps = self.beam_search(enc[0])[:1]

        results = []
        for hyp in nbest_hyps:
            # remove sos/eos and get results
            if self.last_idx is not None:
                token_int = list(hyp.yseq[self.start_idx : self.last_idx])
            else:
                token_int = list(hyp.yseq[self.start_idx :])

            # remove blank symbol id, which is assumed to be 0
            token_int = list([int(i) for i in filter(lambda x: x != 0, token_int)])

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        return results

    def _greedy_ctc(
            self, speech: np.ndarray, lengths: np.ndarray
    ) -> List[Tuple[Optional[str], List[str], List[int], Union[Hypothesis],]]:
        # b. Forward encoder and ctc
        ctc_probs, ctc_ids = self.combined_model(speech=speech, speech_length=lengths)
        y_hat = np.stack([x[0] for x in groupby(ctc_ids[0])])
        y_idx = np.nonzero(y_hat != 0)[0]
        token_ids = y_hat[y_idx].tolist()
        token = self.converter.ids2tokens(token_ids)
        text = self.tokenizer.tokens2text(token)
        return [
            (text, token, token_ids, ctc_probs[0][y_idx])
        ]

    def ids2text(self, ids: List[int]):
        text = "".join(self.converter.ids2tokens(ids))
        # return text.replace("<mask>", "_").replace("<space>", " ")
        return text.replace("<space>", " ")
