
from .tts_models.vits import VITS
from .tts_models.fast_speech2 import FastSpeech2


def get_tts_model(config, providers, use_quantized):
    if config.model_type == 'VITS':
        return VITS(config, providers, use_quantized)
    elif config.model_type == 'FastSpeech2':
        return FastSpeech2(config, providers, use_quantized)
