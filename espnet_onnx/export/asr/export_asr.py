from typing import Union
from pathlib import Path
from typeguard import check_argument_types

import os
import glob
from datetime import datetime
import shutil

import numpy as np
import torch
from onnxruntime.quantization import quantize_dynamic

from espnet2.bin.asr_inference import Speech2Text
from .models import (
    Encoder,
    Decoder,
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
        
        self.cache_dir = cache_dir

    def export(self, model: Speech2Text, model_name: str = None, quantize: bool = False):
        assert check_argument_types()
        if model_name is None:
            model_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        base_dir = self.cache_dir / model_name
        export_dir = base_dir / 'full'
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # copy stats file
        if model.asr_model.normalize is not None:
            self._copy_stats(model.asr_model.normalize, base_dir)
        
        model_config = self._create_config(model, export_dir)
        
        # export encoder
        enc_model = Encoder(model.asr_model.encoder)
        enc_out_size = model.asr_model.encoder.encoders[0].size
        self._export_encoder(enc_model, export_dir)
        model_config.update(encoder=enc_model.get_model_config(model.asr_model, export_dir))
        
        # export decoder
        dec_model = Decoder(model.asr_model.decoder)
        self._export_decoder(dec_model, enc_out_size, export_dir)
        model_config.update(decoder=dec_model.get_model_config(export_dir))
        
        # export ctc
        ctc_model = CTC(model.asr_model.ctc.ctc_lo)
        self._export_ctc(ctc_model, enc_out_size, export_dir)
        model_config.update(ctc=ctc_model.get_model_config(export_dir))
        
        # export lm
        if 'lm' in model.beam_search.full_scorers.keys():
            lm_model = LanguageModel(model.beam_search.full_scorers['lm'])
            self._export_lm(lm_model, enc_out_size, export_dir)
            model_config.update(lm=lm_model.get_model_config(export_dir))
        else:
            model_config.update(lm=dict(use_lm=False))

        if quantize:
            quantize_dir = base_dir / 'quantize'
            quantize_dir.mkdir(exist_ok=True)
            qt_config = self._quantize_model(export_dir, quantize_dir)
            for m in qt_config.keys():
                model_config[m].update(quantized_model_path=qt_config[m])
        
        config_name = base_dir / 'config.yaml'
        save_config(model_config, config_name)
        update_model_path(model_name, base_dir)

    def export_from_pretrained(self, tag_name: str, quantize: bool = False):
        assert check_argument_types()
        model = Speech2Text.from_pretrained(tag_name)
        self.export(model, tag_name.replace(' ', '-'), quantize)
    
    def _create_config(self, model, path):
        ret = {}
        if "ngram" in list(model.beam_search.full_scorers.keys()) \
                + list(model.beam_search.part_scorers.keys()):
            ret.update(ngram=get_ngram_config(model))
        else:
            ret.update(ngram=dict(use_ngram=False))

        ret.update(weights=model.beam_search.weights)
        ret.update(beam_search=get_beam_config(model.beam_search, model.minlenratio, model.maxlenratio))
        ret.update(token=get_token_config(model.asr_model))
        ret.update(tokenizer=get_tokenizer_config(model.tokenizer))
        return ret

    def _export_encoder(self, model, path):
        file_name = os.path.join(path, 'encoder.onnx')
        torch.onnx.export(
            model,
            model.get_dummy_inputs(),
            file_name,
            verbose=False,
            opset_version=11,
            input_names=model.get_input_names(),
            output_names=model.get_output_names(),
            dynamic_axes=model.get_dynamic_axes()
        )
    
    def _export_decoder(self, dec_model, enc_size, path):
        file_name = os.path.join(path, 'decoder.onnx')
        torch.onnx.export(
            dec_model,
            dec_model.get_dummy_inputs(enc_size),
            file_name,
            verbose=True,
            opset_version=11,
            input_names=dec_model.get_input_names(),
            output_names=dec_model.get_output_names(),
            dynamic_axes=dec_model.get_dynamic_axes()
        )
    
    def _export_ctc(self, ctc_model, enc_size, path):
        file_name = os.path.join(path, 'ctc.onnx')
        torch.onnx.export(
            ctc_model,
            ctc_model.get_dummy_inputs(enc_size),
            file_name,
            verbose=True,
            opset_version=11,
            input_names=ctc_model.get_input_names(),
            output_names=ctc_model.get_output_names(),
            dynamic_axes=ctc_model.get_dynamic_axes()
        )
    
    def _export_lm(self, lm_model, enc_size, path):
        file_name = os.path.join(path, 'lm.onnx')
        # export encoder
        torch.onnx.export(
            lm_model,
            lm_model.get_dummy_inputs(enc_size),
            file_name,
            verbose=True,
            opset_version=11,
            input_names=lm_model.get_input_names(),
            output_names=lm_model.get_output_names(),
            dynamic_axes=lm_model.get_dynamic_axes()
        )
    
    def _copy_stats(self, model, path):
        stats_file = model.stats_file
        shutil.copyfile(stats_file, path / 'feats_stats.npz')
    
    def _quantize_model(self, model_from, model_to):
        ret = {}
        models = glob.glob(os.path.join(model_from, "*.onnx"))
        for m in models:
            basename = os.path.basename(m).split('.')[0]
            export_file = os.path.join(model_to, basename + '_qt.onnx')
            quantize_dynamic(
                m,
                export_file
            )
            ret[basename] = export_file
            os.remove(os.path.join(model_from, basename + '-opt.onnx'))
        return ret

        