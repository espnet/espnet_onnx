from typing import List
from espnet_onnx.utils.config import Config
from espnet_onnx.asr.model.lms.seqrnn_lm import SequentialRNNLM
from espnet_onnx.asr.model.lms.transformer_lm import TransformerLM


def get_lm(config: Config, providers: List[str], use_quantized: bool = False):
    if config.lm.use_lm:
        if config.lm.lm_type == 'SequentialRNNLM':
            return SequentialRNNLM(config.lm, use_quantized, providers)
        elif config.lm.lm_type == 'TransformerLM':
            return TransformerLM(config.lm, use_quantized, providers)
    return None
