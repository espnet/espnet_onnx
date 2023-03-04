import glob
import logging
import os
import warnings
from typing import List

import onnxruntime

from espnet_onnx.asr.frontend.normalize.global_mvn import GlobalMVN
from espnet_onnx.asr.postprocess.build_tokenizer import build_tokenizer
from espnet_onnx.asr.postprocess.token_id_converter import TokenIDConverter
from espnet_onnx.tts.model.duration_calculator import DurationCalculator
from espnet_onnx.tts.model.preprocess.common_processor import \
    CommonPreprocessor
from espnet_onnx.tts.model.tts_model import get_tts_model
from espnet_onnx.tts.model.vocoders.griffin_lim import Spectrogram2Waveform
from espnet_onnx.tts.model.vocoders.vocoder import Vocoder
from espnet_onnx.utils.abs_model import AbsModel
from espnet_onnx.utils.config import get_config, get_tag_config


class AbsTTSModel(AbsModel):
    def _check_argument(self, tag_name, model_dir):
        self.model_dir = model_dir

        if tag_name is None and model_dir is None:
            raise ValueError("tag_name or model_dir should be defined.")

        if tag_name is not None:
            tag_config = get_tag_config()
            if tag_name not in tag_config.keys():
                raise RuntimeError(
                    f'Model path for tag_name "{tag_name}" is not set on tag_config.yaml.'
                    + "You have to export to onnx format with `espnet_onnx.export.asr.export_asr.ModelExport`,"
                    + "or have to set exported model path in tag_config.yaml."
                )
            self.model_dir = tag_config[tag_name]

    def _load_config(self):
        config_file = glob.glob(os.path.join(self.model_dir, "config.*"))[0]
        self.config = get_config(config_file)

    def _build_tokenizer(self):
        if self.config.tokenizer.token_type is None:
            self.tokenizer = None
        elif self.config.tokenizer.token_type == "bpe":
            self.tokenizer = build_tokenizer("bpe", self.config.tokenizer.bpemodel)
        else:
            self.tokenizer = build_tokenizer(**self.config.tokenizer.dic)

    def _build_token_converter(self):
        self.converter = TokenIDConverter(token_list=self.config.token.list)

    def _build_vocoder(self, providers, use_quantized):
        self.vocoder = None
        if self.config.vocoder.vocoder_type == "not_used":
            logging.info("Vocoder is not used.")
        elif self.config.vocoder.vocoder_type == "Spectrogram2Waveform":
            self.vocoder = Spectrogram2Waveform(self.config.vocoder)
        elif self.config.vocoder.vocoder_type == "PretrainedPWGVocoder":
            raise RuntimeError("Currently, PWGVocoder is not supported.")
        elif self.config.vocoder.vocoder_type == "OnnxVocoder":
            self.vocoder = Vocoder(self.config.vocoder, providers, use_quantized)
        else:
            raise RuntimeError(
                f"vocoder type {self.config.vocoder_type} is not supported."
            )

    def _build_normalizer(self):
        self.normalize = None
        if self.config.normalize.use_normalize:
            self.normalize = GlobalMVN(self.config.normalize)

    def _build_model(self, providers, use_quantized):
        # build tts model such as vits
        self.tts_model = get_tts_model(self.config.tts_model, providers, use_quantized)

        self._build_tokenizer()
        self._build_token_converter()
        self.preprocess = CommonPreprocessor(
            tokenizer=self.tokenizer,
            token_id_converter=self.converter,
            cleaner_config=self.config.text_cleaner,
        )
        self.duration_calculator = DurationCalculator()
        self._build_normalizer()
        self._build_vocoder(providers, use_quantized)

    def _check_ort_version(self, providers: List[str]):
        # check cpu
        if (
            onnxruntime.get_device() == "CPU"
            and "CPUExecutionProvider" not in providers
        ):
            raise RuntimeError(
                "If you want to use GPU, then follow `How to use GPU on espnet_onnx` chapter in readme to install onnxruntime-gpu."
            )

        # check GPU
        if onnxruntime.get_device() == "GPU" and providers == ["CPUExecutionProvider"]:
            warnings.warn(
                "Inference will be executed on the CPU. Please provide gpu providers. Read `How to use GPU on espnet_onnx` in readme in detail."
            )

        logging.info(f'Providers [{" ,".join(providers)}] detected.')
