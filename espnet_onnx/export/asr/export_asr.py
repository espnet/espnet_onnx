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

from espnet2.bin.asr_inference import Speech2Text
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet_model_zoo.downloader import ModelDownloader
from .models import (
    get_encoder,
    get_decoder,
    RNNDecoder,
    PreDecoder,
    CTC,
    LanguageModel
)
from .get_config import (
    get_ngram_config,
    get_beam_config,
    get_token_config,
    get_tokenizer_config
)
from espnet_onnx.utils.config import (
    save_config,
    update_model_path
)


class ModelExport:
    def __init__(self, cache_dir: Union[Path, str] = None):
        assert check_argument_types()
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "espnet_onnx"

        self.cache_dir = Path(cache_dir)

    def export(
        self,
        model: Speech2Text,
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

        # copy model files
        self._copy_files(model, base_dir, verbose)

        model_config = self._create_config(model, export_dir)

        # export encoder
        enc_model = get_encoder(model.asr_model.encoder)
        enc_out_size = enc_model.get_output_size()
        self._export_encoder(enc_model, export_dir, verbose)
        model_config.update(encoder=enc_model.get_model_config(
            model.asr_model, export_dir))

        # export decoder
        dec_model = get_decoder(model.asr_model.decoder)
        self._export_decoder(dec_model, enc_out_size, export_dir, verbose)
        model_config.update(decoder=dec_model.get_model_config(export_dir))

        # export ctc
        ctc_model = CTC(model.asr_model.ctc.ctc_lo)
        self._export_ctc(ctc_model, enc_out_size, export_dir, verbose)
        model_config.update(ctc=ctc_model.get_model_config(export_dir))

        # export lm
        if 'lm' in model.beam_search.full_scorers.keys():
            lm_model = LanguageModel(model.beam_search.full_scorers['lm'])
            self._export_lm(lm_model, export_dir, verbose)
            model_config.update(lm=lm_model.get_model_config(export_dir))
        else:
            model_config.update(lm=dict(use_lm=False))

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

    def export_from_pretrained(self, tag_name: str, quantize: bool = False):
        assert check_argument_types()
        model = Speech2Text.from_pretrained(tag_name)
        self.export(model, tag_name, quantize)
    
    def export_from_zip(self, path: Union[Path, str], tag_name: str, quantize: bool = False):
        assert check_argument_types()
        cache_dir = Path(path).parent
        d = ModelDownloader(cache_dir)
        model_config = d.unpack_local_file(path)
        model = Speech2Text(**model_config)
        self.export(model, tag_name, quantize)

    def _create_config(self, model, path):
        ret = {}
        if "ngram" in list(model.beam_search.full_scorers.keys()) \
                + list(model.beam_search.part_scorers.keys()):
            ret.update(ngram=get_ngram_config(model))
        else:
            ret.update(ngram=dict(use_ngram=False))

        # Transducer is currently not supported
        ret.update(transducer=dict(use_transducer_decoder=False))
        ret.update(weights=model.beam_search.weights)
        ret.update(beam_search=get_beam_config(
            model.beam_search, model.minlenratio, model.maxlenratio))
        ret.update(token=get_token_config(model.asr_model))
        ret.update(tokenizer=get_tokenizer_config(model.tokenizer, path))
        return ret

    def _export_encoder(self, model, path, verbose):
        file_name = os.path.join(path, 'encoder.onnx')
        if verbose:
            logging.info(f'Encoder model is saved in {file_name}')
        torch.onnx.export(
            model,
            model.get_dummy_inputs(),
            file_name,
            verbose=verbose,
            opset_version=11,
            input_names=model.get_input_names(),
            output_names=model.get_output_names(),
            dynamic_axes=model.get_dynamic_axes()
        )

    def _export_decoder(self, dec_model, enc_size, path, verbose):
        file_name = os.path.join(path, 'decoder.onnx')
        if verbose:
            logging.info(f'Decoder model is saved in {file_name}')
        torch.onnx.export(
            dec_model,
            dec_model.get_dummy_inputs(enc_size),
            file_name,
            verbose=verbose,
            opset_version=11,
            input_names=dec_model.get_input_names(),
            output_names=dec_model.get_output_names(),
            dynamic_axes=dec_model.get_dynamic_axes()
        )
        # if decoder is RNNDecoder, then export predecoders
        if isinstance(dec_model, RNNDecoder):
            self._export_predecoder(dec_model, path, verbose)

    def _export_predecoder(self, dec_model, path, verbose):
        if verbose:
            logging.info(f'Pre-Decoder model is saved in {path}.' \
                + f'There should be {len(dec_model.model.att_list)} files.')
        for i, att in enumerate(dec_model.model.att_list):
            att_model = PreDecoder(att)
            if att_model.require_onnx():
                file_name = os.path.join(path, 'predecoder_%d.onnx' % i)
                torch.onnx.export(
                    att_model,
                    att_model.get_dummy_inputs(),
                    file_name,
                    verbose=verbose,
                    opset_version=11,
                    input_names=att_model.get_input_names(),
                    output_names=att_model.get_output_names(),
                    dynamic_axes=att_model.get_dynamic_axes()
                )

    def _export_ctc(self, ctc_model, enc_size, path, verbose):
        file_name = os.path.join(path, 'ctc.onnx')
        if verbose:
            logging.info(f'CTC model is saved in {file_name}')
        torch.onnx.export(
            ctc_model,
            ctc_model.get_dummy_inputs(enc_size),
            file_name,
            verbose=verbose,
            opset_version=11,
            input_names=ctc_model.get_input_names(),
            output_names=ctc_model.get_output_names(),
            dynamic_axes=ctc_model.get_dynamic_axes()
        )

    def _export_lm(self, lm_model, path, verbose):
        file_name = os.path.join(path, 'lm.onnx')
        if verbose:
            logging.info(f'LM model is saved in {file_name}')
        # export encoder
        torch.onnx.export(
            lm_model,
            lm_model.get_dummy_inputs(),
            file_name,
            verbose=verbose,
            opset_version=11,
            input_names=lm_model.get_input_names(),
            output_names=lm_model.get_output_names(),
            dynamic_axes=lm_model.get_dynamic_axes()
        )

    def _copy_files(self, model, path, verbose):
        # copy stats file
        if model.asr_model.normalize is not None \
                and hasattr(model.asr_model.normalize, 'stats_file'):
            stats_file = model.asr_model.normalize.stats_file
            shutil.copy(stats_file, path)
            if verbose:
                logging.info(f'`stats_file` was copied into {path}.')

        # copy bpemodel
        if isinstance(model.tokenizer, SentencepiecesTokenizer):
            bpemodel_file = model.tokenizer.model
            shutil.copy(bpemodel_file, path)
            if verbose:
                logging.info(f'bpemodel was copied into {path}.')
            
        # save position encoder parameters.
        if hasattr(model.asr_model.encoder, 'pos_enc'):
            np.save(
                path / 'pe',
                model.asr_model.encoder.pos_enc.pe.numpy()
            )
            if verbose:
                logging.info(f'Matrix for position encoding was copied into {path}.')

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
