from typing import (
    Union,
    List,
    Tuple,
    Optional
)
from pathlib import Path
from typeguard import check_argument_types

import os
import logging
import numpy as np
import glob
import warnings
import onnxruntime

from espnet_onnx.asr.model.encoder import get_encoder
from espnet_onnx.asr.model.decoder import get_decoder
from espnet_onnx.asr.model.lm.seqrnn_lm import SequentialRNNLM
from espnet_onnx.asr.model.lm.transformer_lm import TransformerLM
from espnet_onnx.asr.scorer.ctc_prefix_scorer import CTCPrefixScorer
from espnet_onnx.asr.scorer.length_bonus import LengthBonus
from espnet_onnx.asr.scorer.interface import BatchScorerInterface
from espnet_onnx.asr.beam_search.hyps import Hypothesis
from espnet_onnx.asr.beam_search.beam_search import BeamSearch
from espnet_onnx.asr.beam_search.batch_beam_search_online_sim import BatchBeamSearchOnlineSim
from espnet_onnx.asr.beam_search.batch_beam_search import BatchBeamSearch
from espnet_onnx.asr.postprocess.build_tokenizer import build_tokenizer
from espnet_onnx.asr.postprocess.token_id_converter import TokenIDConverter

from espnet_onnx.utils.config import get_config
from espnet_onnx.utils.config import get_tag_config


class StreamingSpeech2Text:
    """Speech2Text class to support streaming asr
    """
    def __init__(self,
                 tag_name: str = None,
                 model_dir: Union[Path, str] = None,
                 providers: List[str] = ['CPUExecutionProvider'],
                 use_quantized: bool = False,
                 block_size: int = 40,
                 hop_size: int = 16,
                 look_ahead: int = 16
                 ):
        assert check_argument_types()
        if tag_name is None and model_dir is None:
            raise ValueError('tag_name or model_dir should be defined.')

        if tag_name is not None:
            tag_config = get_tag_config()
            if tag_name not in tag_config.keys():
                raise RuntimeError(f'Model path for tag_name "{tag_name}" is not set on tag_config.yaml.'
                                   + 'You have to export to onnx format with `espnet_onnx.export.asr.export_asr.ModelExport`,'
                                   + 'or have to set exported model path in tag_config.yaml.')
            model_dir = tag_config[tag_name]

        # check onnxruntime version and providers
        self.check_ort_version(providers)
        
        # 1. Build asr model
        config_file = glob.glob(os.path.join(model_dir, 'config.*'))[0]
        config = get_config(config_file)
        
        # check if model is exported for streaming.
        if config.encoder.enc_type != 'ContextualXformerEncoder':
            raise RuntimeError('Onnx model is not build for streaming. Use Speech2Text instead.')

        if use_quantized and 'quantized_model_path' not in config.encoder.keys():
            # check if quantized model config is defined.
            raise RuntimeError(
                'Configuration for quantized model is not defined.')

        # 2.
        config.encoder.block_size = block_size
        config.encoder.hop_size = hop_size
        config.encoder.look_ahead = look_ahead
        
        self.encoder = get_encoder(config.encoder, providers, use_quantized)
        decoder = get_decoder(config.decoder, config.transducer, providers, use_quantized)
        ctc = CTCPrefixScorer(config.ctc, config.token.eos, providers, use_quantized)

        scorers = {}
        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(config.token.list))
        )

        # 2. Build lm model
        if config.lm.use_lm:
            if config.lm.lm_type == 'SequentialRNNLM':
                scorers.update(
                    lm=SequentialRNNLM(config.lm, providers, use_quantized)
                )
            elif config.lm.lm_type == 'TransformerLM':
                scorers.update(
                    lm=TransformerLM(config.lm, providers, use_quantized)
                )

        # 3. Build ngram model
        # Currently not supported.
        # if config.ngram.use_ngram:
        #     if config.ngram.scorer_type == 'full':
        #         scorers.update(
        #             ngram=NgramFullScorer(
        #                 config.ngram.ngram_file,
        #                 config.token.list
        #             )
        #         )
        #     else:
        #         scorers.update(
        #             ngram=NgramPartialScorer(
        #                 config.ngram.ngram_file,
        #                 config.token.list
        #             )
        #         )

        # 4. Build beam search object
        weights = dict(
            decoder=config.weights.decoder,
            ctc=config.weights.ctc,
            lm=config.weights.lm,
            length_bonus=config.weights.length_bonus,
        )
        self.beam_search = BeamSearch(
            config.beam_search,
            config.token,
            scorers=scorers,
            weights=weights,
        )
        self.batch_beam_search = BatchBeamSearch(
            config.beam_search,
            config.token,
            scorers=scorers,
            weights=weights,
        )

        non_batch = [
            k
            for k, v in self.beam_search.full_scorers.items()
            if not isinstance(v, BatchScorerInterface)
        ]
        if len(non_batch) == 0:
            self.beam_search.__class__ = BatchBeamSearchOnlineSim
            self.beam_search.set_streaming_config(
                block_size, hop_size, look_ahead
            )
            logging.info(
                "BatchBeamSearchOnlineSim implementation is selected."
            )
        else:
            logging.warning(
                f"As non-batch scorers {non_batch} are found, "
                f"fall back to non-batch implementation."
            )

        # 5. Build text converter
        if config.tokenizer.token_type is None:
            self.tokenizer = None
        elif config.tokenizer.token_type == 'bpe':
            self.tokenizer = build_tokenizer(
                'bpe', config.tokenizer.bpemodel)
        else:
            self.tokenizer = build_tokenizer(
                token_type=config.tokenizer.token_type)

        self.converter = TokenIDConverter(token_list=config.token.list)
        self.config = config
        
        # streaming related parameters
        self.enc_feats = []
        self.streaming_states = {
            'buffer_before_downsampling' : None,
            'buffer_after_downsampling' : None,
            'prev_addin' : None,
            'past_encoder_ctx' : None,
        }
        self.hop_size = self.config.encoder.frontend.stft.hop_length * self.config.encoder.subsample * hop_size  \
            + (self.config.encoder.frontend.stft.n_fft // self.config.encoder.frontend.stft.hop_length) * self.config.encoder.frontend.stft.hop_length

    def __call__(self, speech: np.ndarray) -> List[
        Tuple[
            Optional[str],
            List[str],
            List[int],
            Union[Hypothesis],
        ]
    ]:
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
            return self.get_result(best_hyps[0])

    def simulate(self, speech: np.ndarray, print_every_hypo: bool = False):
        # This function will simulate streaming asr with the given audio.
        self.start()
        process_num = len(speech) // self.hop_size + 1
        logging.info(f'Processing audio with {process_num} processes.')
        padded_speech = self._pad(speech, length=process_num * self.hop_size)
        for i in range(process_num):
            start = self.hop_size * i
            end = self.hop_size * (i + 1)
            nbest = self(padded_speech[start:end])
            if print_every_hypo and nbest != []:
                logging.info(f'Result at position {i} : {nbest[0][0]}')
        
        nbest = self.end()
        return nbest

    def start(self):
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
        self.enc_feats = []
        self.streaming_states = {
            'buffer_before_downsampling' : None,
            'buffer_after_downsampling' : None,
            'prev_addin' : None,
            'past_encoder_ctx' : None,
        }
        self.beam_search.end()
        
        if best_hyps == []:
            return []
        else:
            return self.get_result(best_hyps[0])

    def get_result(self, hyp: Hypothesis):
        token_int = hyp.yseq[2:-1].astype(np.int64).tolist()
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
    
    def _pad(self, x, length=None):
        if length:
            base = np.zeros((length,))
        else:
            base = np.zeros((self.config.encoder.block_size * self.config.encoder.frontend.stft.hop_length,))
        base[:len(x)] = x
        return base
    
    def check_ort_version(self, providers: List[str]):
        # check cpu
        if onnxruntime.get_device() == 'CPU' and 'CPUExecutionProvider' not in providers:
            raise RuntimeError('If you want to use GPU, then follow `How to use GPU on espnet_onnx` chapter in readme to install onnxruntime-gpu.')
        
        # check GPU
        if onnxruntime.get_device() == 'GPU' and providers == ['CPUExecutionProvider']:
            warnings.warn('Inference will be executed on the CPU. Please provide gpu providers. Read `How to use GPU on espnet_onnx` in readme in detail.')
        
        logging.info(f'Providers [{" ,".join(providers)}] detected.')