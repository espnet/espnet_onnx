from abc import ABC

import os
import glob
import logging

from espnet_onnx.asr.model.encoder import get_encoder
from espnet_onnx.asr.model.decoder import get_decoder
from espnet_onnx.asr.model.lm import get_lm
from espnet_onnx.asr.scorer.ctc_prefix_scorer import CTCPrefixScorer
from espnet_onnx.asr.scorer.length_bonus import LengthBonus
from espnet_onnx.asr.scorer.interface import BatchScorerInterface
from espnet_onnx.asr.beam_search.beam_search import BeamSearch
from espnet_onnx.asr.beam_search.batch_beam_search import BatchBeamSearch
from espnet_onnx.asr.postprocess.build_tokenizer import build_tokenizer
from espnet_onnx.asr.postprocess.token_id_converter import TokenIDConverter

from espnet_onnx.utils.config import get_config
from espnet_onnx.utils.config import get_tag_config


class AbsASRModel(ABC):
    
    def _check_argument(self, tag_name, model_dir):
        self.model_dir = model_dir
        
        if tag_name is None and model_dir is None:
            raise ValueError('tag_name or model_dir should be defined.')

        if tag_name is not None:
            tag_config = get_tag_config()
            if tag_name not in tag_config.keys():
                raise RuntimeError(f'Model path for tag_name "{tag_name}" is not set on tag_config.yaml.'
                                   + 'You have to export to onnx format with `espnet_onnx.export.asr.export_asr.ModelExport`,'
                                   + 'or have to set exported model path in tag_config.yaml.')
            self.model_dir = tag_config[tag_name]
    
    def _load_config(self):
        config_file = glob.glob(os.path.join(self.model_dir, 'config.*'))[0]
        self.config = get_config(config_file)
    
    def _build_beam_search(self, scorers, weights):
        if self.config.transducer.use_transducer_decoder:
            self.beam_search = BSTransducer(
                self.config.beam_search,
                self.config.token,
                scorers=scorers,
                weights=weights
            )
        else:
            self.beam_search = BeamSearch(
                self.config.beam_search,
                self.config.token,
                scorers=scorers,
                weights=weights,
            )
            non_batch = [
                k for k, v in self.beam_search.full_scorers.items()
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
    
    def _build_tokenizer(self):
        if self.config.tokenizer.token_type is None:
            self.tokenizer = None
        elif self.config.tokenizer.token_type == 'bpe':
            self.tokenizer = build_tokenizer(
                'bpe', self.config.tokenizer.bpemodel)
        else:
            self.tokenizer = build_tokenizer(
                token_type=self.config.tokenizer.token_type)

    def _build_token_converter(self):
        self.converter = TokenIDConverter(token_list=self.config.token.list)
    
    def _build_model(self, providers, use_quantized):
        self.encoder = get_encoder(self.config.encoder, providers, use_quantized)
        decoder = get_decoder(self.config.decoder, providers, use_quantized)
        scorers = {'decoder': decoder}
        weights = {}
        if not self.config.transducer.use_transducer_decoder:
            ctc = CTCPrefixScorer(self.config.ctc, self.config.token.eos, providers, use_quantized)
            scorers.update(
                ctc=ctc,
                length_bonus=LengthBonus(len(self.config.token.list))
            )
            weights.update(
                decoder=self.config.weights.decoder,
                ctc=self.config.weights.ctc,
                length_bonus=self.config.weights.length_bonus,
            )
        else:
            joint_network = JointNetwork(self.onfig.joint_network, providers, use_quantized)
            scorers.update(joint_network=joint_network)
            
        lm = get_lm(self.config, providers, use_quantized)
        if lm is not None:
            scorers.update(lm=lm)
            weights.update(lm=self.config.weights.lm)

        self._build_beam_search(scorers, weights)
        self._build_tokenizer()
        self._build_token_converter()
        self.scorers = scorers
        self.weights = weights

    def _check_ort_version(self):
        raise NotImplementedError