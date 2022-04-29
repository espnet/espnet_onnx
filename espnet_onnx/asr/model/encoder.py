from typing import List
from espnet_onnx.utils.config import Config
from espnet_onnx.asr.model.encoders.encoder import Encoder
from espnet_onnx.asr.model.encoders.streaming import StreamingEncoder


def get_encoder(config: Config, providers: List[str], use_quantized: bool = False):
    """Get wrapper class of the onnx encoder model.

    Args:
        config (Config): Decoder config.
        providers (List[str]): Providers for the decoder.
        use_quantized (bool, optional): Flag to use quantized model. Defaults to False.

    """
    if config.enc_type == 'ContextualXformerEncoder':
        return StreamingEncoder(config, providers, use_quantized)
    else:
        return Encoder(config, providers, use_quantized)
