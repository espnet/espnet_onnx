from typing import List

from espnet_onnx.asr.model.decoders.rnn import RNNDecoder
from espnet_onnx.asr.model.decoders.transducer import TransducerDecoder
from espnet_onnx.asr.model.decoders.xformer import XformerDecoder
from espnet_onnx.utils.config import Config


def get_decoder(config: Config, providers: List[str], use_quantized: bool = False):
    if config.dec_type == "RNNDecoder":
        return RNNDecoder(config, providers, use_quantized)
    elif config.dec_type == "TransducerDecoder":
        return TransducerDecoder(config, providers, use_quantized)
    else:
        return XformerDecoder(config, providers, use_quantized)
