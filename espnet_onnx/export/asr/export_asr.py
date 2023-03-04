import glob
import logging
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from espnet2.bin.asr_inference import Speech2Text
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet_model_zoo.downloader import ModelDownloader
from onnxruntime.quantization import quantize_dynamic
from typeguard import check_argument_types

from espnet_onnx.export.asr.get_config import (get_beam_config,
                                               get_ngram_config,
                                               get_token_config,
                                               get_tokenizer_config,
                                               get_trans_beam_config,
                                               get_weights_transducer)
from espnet_onnx.export.asr.models import (CTC, JointNetwork, get_decoder,
                                           get_encoder, get_lm)
from espnet_onnx.export.optimize.optimizer import optimize_model
from espnet_onnx.utils.config import save_config, update_model_path


class ASRModelExport:
    def __init__(self, cache_dir: Union[Path, str] = None):
        assert check_argument_types()
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "espnet_onnx"

        self.cache_dir = Path(cache_dir)
        self.export_config = dict(
            use_gpu=False,
            only_onnxruntime=False,
            float16=False,
            use_ort_for_espnet=False,
        )

    def export(
        self,
        model: Speech2Text,
        tag_name: str = None,
        quantize: bool = False,
        optimize: bool = False,
        verbose: bool = False,
    ):
        assert check_argument_types()
        if tag_name is None:
            tag_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.export_config["optimize"] = optimize

        base_dir = self.cache_dir / tag_name.replace(" ", "-")
        export_dir = base_dir / "full"
        export_dir.mkdir(parents=True, exist_ok=True)

        # copy model files
        self._copy_files(model, base_dir, verbose)

        model_config = self._create_config(model, export_dir)

        # export encoder
        enc_model = get_encoder(
            model.asr_model.encoder,
            model.asr_model.frontend,
            model.asr_model.preencoder,
            self.export_config,
        )
        enc_out_size = enc_model.get_output_size()
        self._export_encoder(enc_model, export_dir, verbose)
        model_config.update(
            encoder=enc_model.get_model_config(model.asr_model, export_dir)
        )

        # export decoder
        dec_model = get_decoder(model.asr_model.decoder, self.export_config)
        self._export_decoder(dec_model, enc_out_size, export_dir, verbose)
        model_config.update(decoder=dec_model.get_model_config(export_dir))

        # export joint_network if transducer decoder is used.
        if model.asr_model.use_transducer_decoder:
            joint_network = JointNetwork(
                model.asr_model.joint_network,
                model_config["beam_search"]["search_type"],
            )
            self._export_joint_network(joint_network, export_dir, verbose)
            model_config.update(
                joint_network=joint_network.get_model_config(export_dir)
            )

        # export ctc
        if model.asr_model.ctc is not None:
            ctc_model = CTC(model.asr_model.ctc.ctc_lo)
            self._export_ctc(ctc_model, enc_out_size, export_dir, verbose)
            model_config.update(ctc=ctc_model.get_model_config(export_dir))

        # export lm
        lm_model = None
        if not model.asr_model.use_transducer_decoder:
            if "lm" in model.beam_search.scorers.keys():
                lm_model = get_lm(model.beam_search.scorers["lm"], self.export_config)
        else:
            if model.beam_search_transducer.use_lm:
                lm_model = get_lm(model.beam_search_transducer.lm, self.export_config)

        if lm_model is not None:
            self._export_lm(lm_model, export_dir, verbose)
            model_config.update(lm=lm_model.get_model_config(export_dir))
        else:
            model_config.update(lm=dict(use_lm=False))

        if optimize:
            if enc_model.is_optimizable():
                self._optimize_model(enc_model, export_dir, "encoder")

            if dec_model.is_optimizable():
                self._optimize_model(dec_model, export_dir, "decoder")

            if lm_model is not None and lm_model.is_optimizable():
                self._optimize_model(lm_model, export_dir, "lm")

            if (
                enc_model.frontend_model is not None
                and enc_model.frontend_model.is_optimizable()
            ):
                self._optimize_model(enc_model.frontend_model, export_dir, "frontend")

        if quantize:
            quantize_dir = base_dir / "quantize"
            quantize_dir.mkdir(exist_ok=True)
            qt_config = self._quantize_model(
                export_dir, quantize_dir, optimize, verbose
            )

            for m in qt_config.keys():
                if "predecoder" in m:
                    model_idx = int(m.split("_")[1])
                    model_config["decoder"]["predecoder"][model_idx].update(
                        quantized_model_path=qt_config[m]
                    )
                elif "encoder" in m:
                    model_config["encoder"].update(quantized_model_path=qt_config[m])
                elif "decoder" in m:
                    model_config["decoder"].update(quantized_model_path=qt_config[m])
                elif "lm" in m:
                    model_config["lm"].update(quantized_model_path=qt_config[m])
                elif "frontend" in m:
                    model_config["encoder"]["frontend"].update(
                        quantized_model_path=qt_config[m]
                    )
                else:
                    model_config[m].update(quantized_model_path=qt_config[m])

        config_name = base_dir / "config.yaml"
        save_config(model_config, config_name)
        update_model_path(tag_name, base_dir)

    def export_from_pretrained(
        self,
        tag_name: str,
        quantize: bool = False,
        optimize: bool = False,
        pretrained_config: Dict = {},
    ):
        assert check_argument_types()
        model = Speech2Text.from_pretrained(tag_name, **pretrained_config)
        self.export(model, tag_name, quantize, optimize)

    def export_from_zip(
        self,
        path: Union[Path, str],
        tag_name: str,
        quantize: bool = False,
        optimize: bool = False,
    ):
        assert check_argument_types()
        cache_dir = Path(path).parent
        d = ModelDownloader(cache_dir)
        model_config = d.unpack_local_file(path)
        model = Speech2Text(**model_config)
        self.export(model, tag_name, quantize, optimize)

    def set_export_config(self, **kwargs):
        for k, v in kwargs.items():
            self.export_config[k] = v

    def _create_config(self, model, path):
        ret = {}
        if not model.asr_model.use_transducer_decoder:
            if "ngram" in list(model.beam_search.full_scorers.keys()) + list(
                model.beam_search.part_scorers.keys()
            ):
                ret.update(ngram=get_ngram_config(model))
            else:
                ret.update(ngram=dict(use_ngram=False))
            ret.update(weights=model.beam_search.weights)
            ret.update(
                beam_search=get_beam_config(
                    model.beam_search, model.minlenratio, model.maxlenratio
                )
            )
        else:
            ret.update(weights=get_weights_transducer(model.beam_search_transducer))
            ret.update(beam_search=get_trans_beam_config(model.beam_search_transducer))

        ret.update(
            transducer=dict(
                use_transducer_decoder=model.asr_model.use_transducer_decoder
            )
        )
        ret.update(token=get_token_config(model.asr_model))
        ret.update(tokenizer=get_tokenizer_config(model.tokenizer, path))
        return ret

    def _export_model(self, model, verbose, path, enc_size=None):
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
                if sm.require_onnx():
                    self._export_model(sm, verbose, path, enc_size)

    def _export_encoder(self, model, path, verbose):
        if verbose:
            logging.info(f"Encoder model is saved in {file_name}")
        self._export_model(model, verbose, path)

    def _export_frontend(self, model, path, verbose):
        if verbose:
            logging.info(f"Frontend model is saved in {file_name}")
        self._export_model(model, verbose, path)

    def _export_decoder(self, model, enc_size, path, verbose):
        if verbose:
            logging.info(f"Decoder model is saved in {file_name}")
        self._export_model(model, verbose, path, enc_size)

    def _export_ctc(self, model, enc_size, path, verbose):
        if verbose:
            logging.info(f"CTC model is saved in {file_name}")
        self._export_model(model, verbose, path, enc_size)

    def _export_lm(self, model, path, verbose):
        if verbose:
            logging.info(f"LM model is saved in {file_name}")
        self._export_model(model, verbose, path)

    def _export_joint_network(self, model, path, verbose):
        if verbose:
            logging.info(f"JointNetwork model is saved in {file_name}")
        self._export_model(model, verbose, path)

    def _copy_files(self, model, path, verbose):
        # copy stats file
        if model.asr_model.normalize is not None and hasattr(
            model.asr_model.normalize, "stats_file"
        ):
            stats_file = model.asr_model.normalize.stats_file
            shutil.copy(stats_file, path)
            if verbose:
                logging.info(f"`stats_file` was copied into {path}.")

        # copy bpemodel
        if isinstance(model.tokenizer, SentencepiecesTokenizer):
            bpemodel_file = model.tokenizer.model
            shutil.copy(bpemodel_file, path)
            if verbose:
                logging.info(f"bpemodel was copied into {path}.")

        # save position encoder parameters.
        if hasattr(model.asr_model.encoder, "pos_enc"):
            np.save(path / "pe", model.asr_model.encoder.pos_enc.pe.numpy())
            if verbose:
                logging.info(f"Matrix for position encoding was copied into {path}.")

    def _quantize_model(self, model_from, model_to, optimize, verbose):
        if verbose:
            logging.info(f"Quantized model is saved in {model_to}.")
        ret = {}
        models = glob.glob(os.path.join(model_from, "*.onnx"))

        if self.export_config["use_ort_for_espnet"]:
            op_types_to_quantize = [
                "Attention",
                "CrossAttention",
                "RelPosAttention",
                "MatMul",
            ]
        else:
            op_types_to_quantize = ["Attention", "MatMul"]

        for m in models:
            basename = os.path.basename(m).split(".")[0]
            export_file = os.path.join(model_to, basename + "_qt.onnx")
            quantize_dynamic(m, export_file, op_types_to_quantize=op_types_to_quantize)
            ret[basename] = export_file
            if os.path.exists(os.path.join(model_from, basename + "-opt.onnx")):
                os.remove(os.path.join(model_from, basename + "-opt.onnx"))
        return ret

    def _optimize_model(self, model, model_dir, model_type):
        if model_type == "encoder":
            if self.export_config["use_ort_for_espnet"]:
                model_type = "espnet"
            else:
                model_type = "bert"
        elif model_type in ("decoder", "lm", "frontend"):
            if self.export_config["use_ort_for_espnet"]:
                model_type = "espnet"
            else:
                warnings.warn(
                    "You cannot optimize TransformerDecoder or TransformerLM without custom version of onnxruntime."
                    + "Please follow the instruction on README.md to install onnxruntime for espnet_onnx"
                )
                model_type = None

        if model_type is not None:
            model_name = str(model_dir / model.model_name) + ".onnx"
            opt_name = str(model_dir / model.model_name) + ".opt.onnx"
            optimize_model(
                model_name,
                opt_name,
                model.num_heads,
                model.hidden_size,
                use_gpu=self.export_config["use_gpu"],
                only_onnxruntime=self.export_config["only_onnxruntime"],
                model_type=model_type,
            )
            os.remove(model_dir / model_name)
            os.rename(model_dir / opt_name, model_dir / model_name)
            return True
        else:
            return False
