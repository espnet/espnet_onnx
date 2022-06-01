from espnet2.gan_tts.vits import VITS as espnetVITSModel
from espnet2.tts.fastspeech2.fastspeech2 import FastSpeech2 as espnetFastSpeech2
from .tts_models.vits import OnnxVITSModel
from .tts_models.fastspeech2 import OnnxFastSpeech2

from espnet2.utils.griffin_lim import Spectrogram2Waveform
from espnet2.tts.utils.parallel_wavegan_pretrained_vocoder import ParallelWaveGANPretrainedVocoder


def get_tts_model(model, export_config):
    if isinstance(model.model.tts, espnetVITSModel):
        return OnnxVITSModel(model.model.tts, **export_config)
    elif isinstance(model.model.tts, espnetFastSpeech2):
        return OnnxFastSpeech2(model.model.tts, **export_config)
    else:
        raise RuntimeError('Currently, VITS and FastSpeech2 is supported.')

def get_vocoder(model, export_config):
    if isinstance(model, Spectrogram2Waveform):
        return model, False
    elif isinstance(model, ParallelWaveGANPretrainedVocoder):
        raise RuntimeError('PWGVocoder is currently not supported.')
    else:
        raise RuntimeError('vocoder is not supported.')
