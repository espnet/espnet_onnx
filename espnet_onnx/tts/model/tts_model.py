
from .tts_models.vits import VITS


def get_tts_model(config, providers, use_quantized):
    if config.model_type == 'VITS':
        return VITS(config, providers, use_quantized)
