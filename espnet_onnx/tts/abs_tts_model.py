from abc import ABC

import os
import glob
import logging

from espnet_onnx.tts.model.preprocess import CommonProcessor
from espnet_onnx.tts.model.duration_calculator import DurationCalculator
from espnet_onnx.utils.config import get_config
from espnet_onnx.utils.config import get_tag_config


class AbsTTSModel(ABC):
    
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

    def _build_tokenizer(self):
        if self.config.preprocess.tokenizer.token_type is None:
            self.tokenizer = None
        elif self.config.preprocess.tokenizer.token_type == 'bpe':
            self.tokenizer = build_tokenizer(
                'bpe', self.config.preprocess.tokenizer.bpemodel)
        else:
            self.tokenizer = build_tokenizer(
                token_type=self.config.preprocess.tokenizer.token_type)

    def _build_token_converter(self):
        self.converter = TokenIDConverter(token_list=self.config.token.list)
    
    def _build_model(self, providers, use_quantized):
        # build tts model such as vits
        self.tts_model = get_tts_model(self.config.tts_model, providers, use_quantized)
        
        self._build_tokenizer()
        self._build_token_converter()
        self.preprocess = CommonProcessor(
            tokenizer=self.tokenizer,
            token_id_converter=self.converter,
            text_cleaner=args.cleaner,
        )
        self.duration_calculator = DurationCalculator()

        # vocoder is currently not supported
        # self.vocoder is get_vocoder()

    def _check_ort_version(self, providers: List[str]):
        # check cpu
        if onnxruntime.get_device() == 'CPU' and 'CPUExecutionProvider' not in providers:
            raise RuntimeError('If you want to use GPU, then follow `How to use GPU on espnet_onnx` chapter in readme to install onnxruntime-gpu.')
        
        # check GPU
        if onnxruntime.get_device() == 'GPU' and providers == ['CPUExecutionProvider']:
            warnings.warn('Inference will be executed on the CPU. Please provide gpu providers. Read `How to use GPU on espnet_onnx` in readme in detail.')
        
        logging.info(f'Providers [{" ,".join(providers)}] detected.')