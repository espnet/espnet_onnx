import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from typeguard import check_argument_types

from espnet_onnx.asr.abs_asr_model import AbsASRModel
from espnet_onnx.asr.beam_search.batch_beam_search import BatchBeamSearch
from espnet_onnx.asr.beam_search.batch_beam_search_online_sim import \
    BatchBeamSearchOnlineSim
from espnet_onnx.asr.beam_search.hyps import Hypothesis


class StreamingSpeech2Text(AbsASRModel):
    """Speech2Text class to support streaming asr"""

    def __init__(
        self,
        tag_name: str = None,
        model_dir: Union[Path, str] = None,
        providers: List[str] = ["CPUExecutionProvider"],
        use_quantized: bool = False,
        block_size: int = 40,
        hop_size: int = 16,
        look_ahead: int = 16,
    ):
        assert check_argument_types()
        self._check_argument(tag_name, model_dir)
        self._load_config()

        # check onnxruntime version and providers
        self._check_ort_version(providers)

        # check if model is exported for streaming.
        if self.config.encoder.enc_type != "ContextualXformerEncoder":
            raise RuntimeError(
                "Onnx model is not build for streaming. Use Speech2Text instead."
            )

        if use_quantized and "quantized_model_path" not in self.config.encoder.keys():
            # check if quantized model config is defined.
            raise RuntimeError("Configuration for quantized model is not defined.")

        # build models
        self.config.encoder.block_size = block_size
        self.config.encoder.hop_size = hop_size
        self.config.encoder.look_ahead = look_ahead

        self._build_model(providers, use_quantized)

        # Fix beam_search components
        self.batch_beam_search = BatchBeamSearch(
            self.config.beam_search,
            self.config.token,
            scorers=self.scorers,
            weights=self.weights,
        )
        self.beam_search.__class__ = BatchBeamSearchOnlineSim
        self.beam_search.set_streaming_config(block_size, hop_size, look_ahead)
        logging.info("BatchBeamSearchOnlineSim implementation is selected.")

        # streaming related parameters
        self._init_streaming_config()
        self.hop_size = (
            self.config.encoder.frontend.stft.hop_length
            * self.config.encoder.subsample
            * hop_size
            + (
                self.config.encoder.frontend.stft.n_fft
                // self.config.encoder.frontend.stft.hop_length
            )
            * self.config.encoder.frontend.stft.hop_length
        )

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

        # b. Forward Encoder
        enc, self.streaming_states = self.encoder(
            speech=speech,
            speech_length=np.array([speech.shape[1]]),
            states=self.streaming_states,
        )
        assert len(enc) == 1, len(enc)
        self.enc_feats += enc[0].tolist()
        best_hyps = self.beam_search(np.array(self.enc_feats, dtype=np.float32))
        self.encoder.increment()

        if best_hyps == []:
            return []
        else:
            return self._get_result(best_hyps[0])

    def simulate(self, speech: np.ndarray, print_every_hypo: bool = False):
        # This function will simulate streaming asr with the given audio.
        self.start()
        process_num = len(speech) // self.hop_size + 1
        logging.info(f"Processing audio with {process_num} processes.")
        padded_speech = self.pad(speech, length=process_num * self.hop_size)
        for i in range(process_num):
            start = self.hop_size * i
            end = self.hop_size * (i + 1)
            nbest = self(padded_speech[start:end])
            if print_every_hypo and nbest != []:
                logging.info(f"Result at position {i} : {nbest[0][0]}")

        nbest = self.end()
        return nbest

    def _init_streaming_config(self):
        self.enc_feats = []
        self.streaming_states = {
            "buffer_before_downsampling": None,
            "buffer_after_downsampling": None,
            "prev_addin": None,
            "past_encoder_ctx": None,
        }

    def start(self):
        self._init_streaming_config()
        self.beam_search.__class__ = BatchBeamSearchOnlineSim
        self.encoder.reset()
        self.streaming_states = self.encoder.init_state()
        self.beam_search.start()

    def end(self):
        # compute final encoder process
        enc, *_ = self.encoder.forward_final(self.streaming_states)
        self.enc_feats += enc[0].tolist()
        best_hyps = self.batch_beam_search(np.array(self.enc_feats, dtype=np.float32))

        # initialize beam_search related parameters
        self._init_streaming_config()
        self.beam_search.end()

        if best_hyps == []:
            return []
        else:
            return self._get_result(best_hyps[0])

    def _get_result(self, hyp: Hypothesis):
        token_int = hyp.yseq[1:-1].astype(np.int64).tolist()
        # remove blank symbol id, which is assumed to be 0
        token_int = list(filter(lambda x: x != 0, token_int))
        # Change integer-ids to tokens
        token = self.converter.ids2tokens(token_int)
        if self.tokenizer is not None:
            text = self.tokenizer.tokens2text(token)
        else:
            text = None
        results = [(text, token, token_int, hyp)]
        return results

    def pad(self, x, length=None):
        if length:
            base = np.zeros((length,))
        else:
            base = np.zeros((self.hop_size,))
        base[: len(x)] = x
        return base
