from typing import List
from espnet_onnx.utils.config import Config
from espnet_onnx.asr.model.decoders.rnn import RNNDecoder
from espnet_onnx.asr.model.decoders.xformer import XformerDecoder


def get_decoder(config: Config, td_config: Config, providers: List[str], use_quantized: bool = False):
    """Get wrapper class of the onnx decoder model.

    Args:
        config (Config): Decoder config.
        td_config (Config): Transducer config.
        providers (List[str]): Providers for the decoder.
        use_quantized (bool, optional): Flag to use quantized model. Defaults to False.

    """
    if td_config.use_transducer_decoder:
        raise ValueError('Transducer is currently not supported.')
    else:
        if config.dec_type == 'RNNDecoder':
            return RNNDecoder(config, providers, use_quantized)
        else:
            return XformerDecoder(config, providers, use_quantized)
