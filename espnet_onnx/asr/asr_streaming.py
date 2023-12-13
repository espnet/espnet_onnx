import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from typeguard import check_argument_types

from espnet_onnx.asr.abs_asr_model import AbsASRModel
from espnet_onnx.asr.beam_search.batch_beam_search_online import BatchBeamSearchOnline
from espnet_onnx.asr.beam_search.hyps import Hypothesis


class StreamingSpeech2Text(AbsASRModel):
    """Speech2Text class to support streaming asr"""

    def __init__(
        self,
        tag_name: str = None,
        model_dir: Union[Path, str] = None,
        providers: List[str] = ["CPUExecutionProvider"],
        cache_dir: Optional[Union[Path, str]] = None,
        use_quantized: bool = False,
        block_size: int = 40,
        hop_size: int = 16,
        look_ahead: int = 16,
        disable_repetition_detection=False,
        encoded_feat_length_limit=0,
        decoder_text_length_limit=0,
    ):
        assert check_argument_types()
        self._check_argument(tag_name, model_dir, cache_dir)
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
        self.beam_search = BatchBeamSearchOnline(
            self.config.beam_search,
            self.config.token,
            scorers=self.scorers,
            weights=self.weights,
            block_size=block_size,
            hop_size=hop_size,
            look_ahead=look_ahead,
            disable_repetition_detection=disable_repetition_detection,
            encoded_feat_length_limit=encoded_feat_length_limit,
            decoder_text_length_limit=decoder_text_length_limit,
        )
        logging.info("BatchBeamSearchOnline is selected.")

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
        self.initial_wav_length = (
            self.config.encoder.frontend.stft.hop_length
            * self.config.encoder.subsample
            * (block_size + 2)
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
        process_num = (len(speech) - self.initial_wav_length) // self.hop_size + 1
        logging.info(f"Processing audio with {process_num + 1} processes.")
        padded_speech = self.pad(
            speech, length=process_num * self.hop_size + self.initial_wav_length
        )

        # initial iteration
        start = 0
        end = self.initial_wav_length
        nbest = self(padded_speech[start:end])
        if print_every_hypo and nbest != []:
            logging.info(f"Result at position {0} : {nbest[0][0]}")

        # second and later iterations
        for i in range(process_num):
            start = self.hop_size * i + self.initial_wav_length
            end = self.hop_size * (i + 1) + self.initial_wav_length
            nbest = self(padded_speech[start:end])
            if print_every_hypo and nbest != []:
                logging.info(f"Result at position {i+1} : {nbest[0][0]}")

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
        self.encoder.reset()
        self.streaming_states = self.encoder.init_state()
        self.beam_search.start()

    def _get_result(self, hyp):
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
