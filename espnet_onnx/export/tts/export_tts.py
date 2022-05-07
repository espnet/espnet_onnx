from typing import Union
from pathlib import Path
from typeguard import check_argument_types

import os
import glob
from datetime import datetime
import shutil
import logging

import numpy as np
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType

from espnet2.bin.tts_inference import Text2Speech
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet_model_zoo.downloader import ModelDownloader
from espnet_onnx.export.tts.models import get_tts_model
get_vocoder = None
# from .get_config import (
#     get_ngram_config,
#     get_beam_config,
#     get_token_config,
#     get_tokenizer_config,
#     get_weights_transducer,
#     get_trans_beam_config,
# )
from espnet_onnx.utils.config import (
    save_config,
    update_model_path
)


class TTSModelExport:
    def __init__(self, cache_dir: Union[Path, str] = None):
        assert check_argument_types()
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "espnet_onnx"

        self.cache_dir = Path(cache_dir)

    def export(
        self,
        model: Text2Speech,
        tag_name: str = None,
        quantize: bool = False,
        verbose: bool = False,
    ):
        assert check_argument_types()
        if tag_name is None:
            tag_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        base_dir = self.cache_dir / tag_name.replace(' ', '-')
        export_dir = base_dir / 'full'
        export_dir.mkdir(parents=True, exist_ok=True)

        model_config = self._create_config(model, export_dir)

        # export encoder
        tts_model = get_tts_model(model)
        self._export_tts(tts_model, export_dir, verbose)
        model_config.update(tts_model=tts_model.get_model_config(export_dir))

        # export vocoder
        if model.vocoder is not None:
            voc_model = get_vocoder(model.vocoder)
            self._export_vocoder(voc_model, export_dir, verbose)
            model_config.update(vocoder=voc_model.get_model_config(export_dir))

        if quantize:
            quantize_dir = base_dir / 'quantize'
            quantize_dir.mkdir(exist_ok=True)
            qt_config = self._quantize_model(export_dir, quantize_dir, verbose)
            for m in qt_config.keys():
                if 'predecoder' in m:
                    model_idx = int(m.split('_')[1])
                    model_config['decoder']['predecoder'][model_idx].update(
                        quantized_model_path=qt_config[m])
                else:
                    model_config[m].update(quantized_model_path=qt_config[m])

        config_name = base_dir / 'config.yaml'
        save_config(model_config, config_name)
        update_model_path(tag_name, base_dir)

    def export_from_pretrained(self, tag_name: str = None, zip_file: str = None, quantize: bool = False):
        assert check_argument_types()
        if ((tag_name is not None) and (zip_file is not None)) \
            or ((tag_name is None) and (zip_file is None)):
            raise RuntimeError('You should specify value for one of ["tag_name", "zip_file"]')
        _t = tag_name if tag_name is not None else zip_file
        model = Text2Speech.from_pretrained(_t)
        self.export(model, tag_name, quantize)

    def _create_config(self, model, path):
        ret = {}
        # if not model.asr_model.use_transducer_decoder:
        #     if "ngram" in list(model.beam_search.full_scorers.keys()) \
        #             + list(model.beam_search.part_scorers.keys()):
        #         ret.update(ngram=get_ngram_config(model))
        #     else:
        #         ret.update(ngram=dict(use_ngram=False))
        #     ret.update(weights=model.beam_search.weights)
        #     ret.update(beam_search=get_beam_config(
        #         model.beam_search, model.minlenratio, model.maxlenratio))
        # else:
        #     ret.update(weights=get_weights_transducer(
        #         model.beam_search_transducer))
        #     ret.update(beam_search=get_trans_beam_config(
        #         model.beam_search_transducer
        #     ))
            
        # ret.update(transducer=dict(use_transducer_decoder=model.asr_model.use_transducer_decoder))
        # ret.update(token=get_token_config(model.asr_model))
        # ret.update(tokenizer=get_tokenizer_config(model.tokenizer, path))
        return ret
    
    def _export_model(self, model, file_name, verbose, enc_size=None):
        if enc_size:
            dummy_input = model.get_dummy_inputs(enc_size)
        else:
            dummy_input = model.get_dummy_inputs()
            
        torch.onnx.export(
            model,
            dummy_input,
            file_name,
            verbose=verbose,
            opset_version=11,
            input_names=model.get_input_names(),
            output_names=model.get_output_names(),
            dynamic_axes=model.get_dynamic_axes()
        )

    def _export_tts(self, model, path, verbose):
        file_name = os.path.join(path, 'tts_model.onnx')
        if verbose:
            logging.info(f'TTS model is saved in {file_name}')
        self._export_model(model, file_name, verbose)
        
        # export submodels
        for sm in model.get_submodel():
            self._export_submodel(sm, path, verbose)

    def _export_submodel(self, model, path, verbose):
        file_name = os.path.join(path, model.get_model_name() + '.onnx')
        if verbose:
            logging.info(f'{model.get_model_name()} is saved in {file_name}')
        self._export_model(model, file_name, verbose)

    def _export_vocoder(self, model, path, verbose):
        file_name = os.path.join(path, 'vocoder.onnx')
        if verbose:
            logging.info(f'Vocoder model is saved in {file_name}')
        self._export_model(model, file_name, verbose, enc_size)

    def _quantize_model(self, model_from, model_to, verbose):
        if verbose:
            logging.info(f'Quantized model is saved in {model_to}.')
        ret = {}
        models = glob.glob(os.path.join(model_from, "*.onnx"))
        for m in models:
            basename = os.path.basename(m).split('.')[0]
            export_file = os.path.join(model_to, basename + '_qt.onnx')
            quantize_dynamic(
                m,
                export_file,
                weight_type=QuantType.QUInt8
            )
            ret[basename] = export_file
            os.remove(os.path.join(model_from, basename + '-opt.onnx'))
        return ret
