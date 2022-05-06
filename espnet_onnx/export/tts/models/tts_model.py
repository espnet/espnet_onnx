
from .tts_models.vits import OnnxVITSModel


def get_tts_model(model):
    if isinstance(model, espnetVITSModel):
        return OnnxVITSModel(model.model.tts, use_teacher_forcing=model.use_teacher_forcing)
    else:
        raise RuntimeError('Currently only VITS model is supported.')