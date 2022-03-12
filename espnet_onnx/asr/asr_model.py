from typing import Union
from typing import List
from typing import Tuple
from typing import Optional
from pathlib import Path
from typeguard import check_argument_types

import os
import logging
import numpy as np
import librosa
import glob

from espnet_onnx.asr.model.encoder import Encoder
from espnet_onnx.asr.model.decoder import get_decoder
from espnet_onnx.asr.model.lm.seqrnn_lm import SequentialRNNLM
from espnet_onnx.asr.model.lm.transformer_lm import TransformerLM
from espnet_onnx.asr.scorer.ctc_prefix_scorer import CTCPrefixScorer
from espnet_onnx.asr.scorer.length_bonus import LengthBonus
from espnet_onnx.asr.scorer.interface import BatchScorerInterface
from espnet_onnx.asr.beam_search.hyps import Hypothesis
from espnet_onnx.asr.beam_search.beam_search import BeamSearch
from espnet_onnx.asr.beam_search.batch_beam_search import BatchBeamSearch
from espnet_onnx.asr.postprocess.build_tokenizer import build_tokenizer
from espnet_onnx.asr.postprocess.token_id_converter import TokenIDConverter

from espnet_onnx.utils.config import get_config
from espnet_onnx.utils.config import get_tag_config


class Speech2Text:
    """Wrapper class for espnet2.asr.bin.asr_infer.Speech2Text

    """

    def __init__(self,
                 tag_name: str = None,
                 model_dir: Union[Path, str] = None,
                 use_quantized: bool = False,
                 ):
        assert check_argument_types()
        if tag_name is None and model_dir is None:
            raise ValueError('tag_name or model_dir should be defined.')

        if tag_name is not None:
            tag_config = get_tag_config()
            if not tag_name in tag_config.keys():
                raise RuntimeError(f'Model path for tag_name "{tag_name}" is not set on tag_config.yaml.'
                                   + 'You have to export to onnx format with `espnet_onnx.export.asr.export_asr.ModelExport`,'
                                   + 'or have to set exported model path in tag_config.yaml.')
            model_dir = tag_config[tag_name]

        # 1. Build asr model
        config_file = glob.glob(os.path.join(model_dir, 'config.*'))[0]
        config = get_config(config_file)

        if use_quantized and 'quantized_model_path' not in config.encoder.keys():
            # check if quantized model config is defined.
            raise RuntimeError(
                'Configuration for quantized model is not defined.')

        # 2.
        self.encoder = Encoder(config.encoder, use_quantized)
        decoder = get_decoder(config.decoder, config.token,
                              config.transducer, use_quantized)
        ctc = CTCPrefixScorer(config.ctc, config.token.eos, use_quantized)

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
                    lm=SequentialRNNLM(config.lm, use_quantized)
                )
            elif config.lm.lm_type == 'TransformerLM':
                scorers.update(
                    lm=TransformerLM(config.lm, use_quantized)
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
        if config.transducer.use_transducer_decoder:
            # self.beam_search = BSTransducer(
            #     config.beam_search,
            #     config.transducer,
            #     config.token,
            #     decoder=decoder,
            #     # lm=scorers["lm"] if "lm" in scorers else None,
            #     weights=weights
            # )
            raise ValueError('Transducer is currently not supported.')
        else:
            self.beam_search = BeamSearch(
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
            self.beam_search.__class__ = BatchBeamSearch
            logging.info("BatchBeamSearch implementation is selected.")
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
        # lengths: (1,)
        lengths = np.array([speech.shape[1]]).astype(np.int64)

        # b. Forward Encoder
        enc, _ = self.encoder(speech=speech, speech_length=lengths)
        if isinstance(enc, tuple):
            enc = enc[0]
        assert len(enc) == 1, len(enc)

        nbest_hyps = self.beam_search(enc[0])[:1]

        results = []
        for hyp in nbest_hyps:
            # remove sos/eos and get results
            last_pos = -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[1:last_pos].astype(np.int64)
            else:
                token_int = hyp.yseq[1:last_pos].astype(np.int64).tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)
            # since I add 'blank' before sos for onnx computing
            token = token[1:]

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        return results
