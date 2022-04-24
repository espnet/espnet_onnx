from espnet_onnx.utils.config import Config
from espnet_onnx.asr.model.encoders.encoder import Encoder
from espnet_onnx.asr.model.encoders.streaming import StreamingEncoder


def get_encoder(config: Config, use_quantized: bool = False):
    if config.enc_type == 'ContextualXformerEncoder':
        return StreamingEncoder(config, use_quantized)
    else:
        return Encoder(config, use_quantized)
