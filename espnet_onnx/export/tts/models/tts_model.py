from espnet2.gan_tts.vits import VITS as espnetVITSModel

from .tts_models.vits import OnnxVITSModel


def get_tts_model(model, export_config):
    if isinstance(model.model.tts, espnetVITSModel):
        return OnnxVITSModel(model.model.tts, **export_config)
    else:
        raise RuntimeError('Currently only VITS model is supported.')
