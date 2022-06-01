from typing import Union
from pathlib import Path
from typeguard import check_argument_types

import os
import glob
from datetime import datetime
import logging

import numpy as np
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType

from espnet2.bin.tts_inference import Text2Speech
from espnet_onnx.export.tts.models import (
    get_tts_model,
    get_vocoder
)
from .get_config import (
    get_token_config,
    get_preprocess_config,
    get_vocoder_config
)
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
        self.export_config = {}

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
        tts_model = get_tts_model(model, self.export_config)
        self._export_tts(tts_model, export_dir, verbose)
        model_config.update(tts_model=tts_model.get_model_config(export_dir))

        # export vocoder
        if model.vocoder is not None:
            voc_model, require_export = get_vocoder(model.vocoder, self.export_config)
            if require_export:
                self._export_vocoder(voc_model, export_dir, verbose)
                model_config.update(vocoder=voc_model.get_model_config(export_dir))
            else:
                model_config.update(vocoder=get_vocoder_config(voc_model))

        if quantize:
            quantize_dir = base_dir / 'quantize'
            quantize_dir.mkdir(exist_ok=True)
            qt_config = self._quantize_model(export_dir, quantize_dir, verbose)
            for m in qt_config.keys():
                model_config[m].update(quantized_model_path=qt_config[m])

        config_name = base_dir / 'config.yaml'
        save_config(model_config, config_name)
        update_model_path(tag_name, base_dir)

    def export_from_pretrained(self, tag_name: str, quantize: bool = False):
        assert check_argument_types()
        model = Text2Speech.from_pretrained(tag_name)
        self.export(model, tag_name, quantize)
    
    def export_from_zip(self, path: Union[Path, str], tag_name: str, quantize: bool = False):
        assert check_argument_types()
        model = Text2Speech.from_pretrained(path)
        self.export(model, tag_name, quantize)
    
    def set_export_config(self, **kwargs):
        for k, v in kwargs.items():
            self.export_config[k] = v
    
    def _create_config(self, model, path):
        ret = {}
        ret.update(get_preprocess_config(model.preprocess_fn, path))
        ret.update(token=get_token_config(
            model.preprocess_fn.token_id_converter))
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
            opset_version=15,
            input_names=model.get_input_names(),
            output_names=model.get_output_names(),
            dynamic_axes=model.get_dynamic_axes(),
        )

    def _export_tts(self, model, path, verbose):
        file_name = os.path.join(path, 'tts_model.onnx')
        if verbose:
            logging.info(f'TTS model is saved in {file_name}')
        self._export_model(model, file_name, verbose)

    # def _export_vocoder(self, model, path, verbose):
    #     file_name = os.path.join(path, 'vocoder.onnx')
    #     if verbose:
    #         logging.info(f'Vocoder model is saved in {file_name}')
    #     self._export_model(model, file_name, verbose, enc_size)

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
