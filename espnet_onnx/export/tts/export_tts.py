import glob
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

import torch
from espnet2.bin.tts_inference import Text2Speech
from onnxruntime.quantization import quantize_dynamic
from typeguard import check_argument_types

from espnet_onnx.export.tts.get_config import (get_normalize_config,
                                               get_preprocess_config,
                                               get_token_config,
                                               get_vocoder_config)
from espnet_onnx.export.tts.models import get_tts_model, get_vocoder
from espnet_onnx.utils.abs_model import AbsExportModel
from espnet_onnx.utils.config import save_config, update_model_path


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

        base_dir = self.cache_dir / tag_name.replace(" ", "-")
        export_dir = base_dir / "full"
        export_dir.mkdir(parents=True, exist_ok=True)

        # copy model files
        self._copy_files(model, base_dir, verbose)

        model_config = self._create_config(model, export_dir)

        # export encoder
        tts_model = get_tts_model(model.model.tts, self.export_config)
        self._export_tts(tts_model, export_dir, verbose)
        if isinstance(tts_model, list):
            model_config.update(
                tts_model=dict(
                    encoder=tts_model[0].get_model_config(export_dir),
                    decoder=tts_model[1].get_model_config(export_dir),
                    model_type=tts_model[2],
                )
            )
        else:
            model_config.update(tts_model=tts_model.get_model_config(export_dir))

        # export vocoder
        voc_model, require_export = get_vocoder(model, self.export_config)
        if voc_model is not None and require_export:
            self._export_vocoder(voc_model, export_dir, verbose)
            model_config.update(vocoder=voc_model.get_model_config(export_dir))
        else:
            model_config.update(vocoder=get_vocoder_config(voc_model))

        if quantize:
            quantize_dir = base_dir / "quantize"
            quantize_dir.mkdir(exist_ok=True)
            qt_config = self._quantize_model(export_dir, quantize_dir, verbose)
            for m in qt_config.keys():
                if "predecoder" in m:
                    model_config["tts_model"]["decoder"]["predecoder"].update(
                        quantized_model_path=qt_config[m]
                    )
                elif "postdecoder" in m:
                    model_config["tts_model"]["decoder"]["postdecoder"].update(
                        quantized_model_path=qt_config[m]
                    )
                elif "tts_model_encoder" in m:
                    model_config["tts_model"]["encoder"].update(
                        quantized_model_path=qt_config[m]
                    )
                elif "tts_model_decoder" in m:
                    model_config["tts_model"]["decoder"].update(
                        quantized_model_path=qt_config[m]
                    )
                else:
                    model_config["tts_model"].update(quantized_model_path=qt_config[m])

        config_name = base_dir / "config.yaml"
        save_config(model_config, config_name)
        update_model_path(tag_name, base_dir)

    def export_from_pretrained(
        self, tag_name: str, quantize: bool = False, pretrained_config: Dict = {}
    ):
        assert check_argument_types()
        model = Text2Speech.from_pretrained(tag_name, **pretrained_config)
        self.export(model, tag_name, quantize)

    def export_from_zip(
        self, path: Union[Path, str], tag_name: str, quantize: bool = False
    ):
        assert check_argument_types()
        model = Text2Speech.from_pretrained(path)
        self.export(model, tag_name, quantize)

    def set_export_config(self, **kwargs):
        for k, v in kwargs.items():
            self.export_config[k] = v

    def _create_config(self, model, path):
        ret = {}
        ret.update(get_preprocess_config(model.preprocess_fn, path))
        ret.update(normalize=get_normalize_config(model.model.normalize, path))
        ret.update(token=get_token_config(model.preprocess_fn.token_id_converter))
        return ret

    def _copy_files(self, model, path, verbose):
        # copy stats file
        if model.model.normalize is not None and hasattr(
            model.model.normalize, "stats_file"
        ):
            stats_file = model.model.normalize.stats_file
            shutil.copy(stats_file, path)
            if verbose:
                logging.info(f"`stats_file` was copied into {path}.")

    def _export_model(self, model, verbose, path, enc_size=None):
        if isinstance(model, list):
            for m in model:
                if isinstance(m, AbsExportModel):
                    self._export_model(m, verbose, path, enc_size)
            return

        if hasattr(model, "onnx_export") and not model.onnx_export:
            return

        if enc_size:
            dummy_input = model.get_dummy_inputs(enc_size)
        else:
            dummy_input = model.get_dummy_inputs()

        torch.onnx.export(
            model,
            dummy_input,
            os.path.join(path, f"{model.model_name}.onnx"),
            verbose=verbose,
            opset_version=15,
            input_names=model.get_input_names(),
            output_names=model.get_output_names(),
            dynamic_axes=model.get_dynamic_axes(),
        )

        # export submodel if required
        if hasattr(model, "submodel"):
            for i, sm in enumerate(model.submodel):
                self._export_model(sm, verbose, path, enc_size)

    def _export_tts(self, model, path, verbose):
        if verbose:
            logging.info(f"TTS model is saved in {file_name}")
        self._export_model(model, verbose, path)

    def _export_vocoder(self, model, path, verbose):
        if verbose:
            logging.info(f"Vocoder model is saved in {file_name}")
        self._export_model(model, verbose, path)

    def _quantize_model(self, model_from, model_to, verbose):
        if verbose:
            logging.info(f"Quantized model is saved in {model_to}.")
        ret = {}
        models = glob.glob(os.path.join(model_from, "*.onnx"))
        for m in models:
            basename = os.path.basename(m).split(".")[0]
            export_file = os.path.join(model_to, basename + "_qt.onnx")
            quantize_dynamic(
                m, export_file, op_types_to_quantize=["Attention", "MatMul"]
            )
            ret[basename] = export_file
            temp_file_path = os.path.join(model_from, basename + "-opt.onnx")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        return ret
