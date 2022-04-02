from espnet_onnx.utils.config import Config
from espnet_onnx.asr.model.decoders.rnn import RNNDecoder
from espnet_onnx.asr.model.decoders.xformer import XformerDecoder


def get_decoder(config, token_config: Config, td_config: Config, use_quantized: bool = False):
    if td_config.use_transducer_decoder:
        raise ValueError('Transducer is currently not supported.')
    else:
        if config.dec_type == 'RNNDecoder':
            return RNNDecoder(config, use_quantized)
        else:
            return XformerDecoder(config, use_quantized)
