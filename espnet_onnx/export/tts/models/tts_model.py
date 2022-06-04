from espnet2.gan_tts.vits import VITS as espnetVITSModel
from espnet2.tts.fastspeech2.fastspeech2 import FastSpeech2 as espnetFastSpeech2
from espnet2.tts.tacotron2.tacotron2 import Tacotron2 as espnetTacotron2
from espnet2.gan_tts.joint.joint_text2wav import JointText2Wav as espnetJointText2Wav
from .tts_models.vits import OnnxVITSModel
from .tts_models.fastspeech2 import OnnxFastSpeech2
from .tts_models.tacotron2 import (
    OnnxTacotron2Encoder,
    OnnxTacotron2Decoder
)

# vocoder
from espnet2.utils.griffin_lim import Spectrogram2Waveform
from espnet2.tts.utils.parallel_wavegan_pretrained_vocoder import ParallelWaveGANPretrainedVocoder
from espnet2.gan_tts.hifigan.hifigan import HiFiGANGenerator
from espnet2.gan_tts.melgan.melgan import MelGANGenerator
from espnet2.gan_tts.parallel_wavegan.parallel_wavegan import ParallelWaveGANGenerator
from espnet2.gan_tts.style_melgan.style_melgan import StyleMelGANGenerator
from .vocoders.vocoder import OnnxVocoder


def get_tts_model(model, export_config):
    if isinstance(model, espnetVITSModel):
        return OnnxVITSModel(model, **export_config)
    
    elif isinstance(model, espnetFastSpeech2):
        return OnnxFastSpeech2(model, **export_config)
    
    elif isinstance(model, espnetTacotron2):
        return [
            OnnxTacotron2Encoder(model, **export_config),
            OnnxTacotron2Decoder(model, **export_config)
        ]   
    
    elif isinstance(model, espnetJointText2Wav):
        return get_tts_model(model.generator['text2mel'], export_config)
    else:
        raise RuntimeError('Currently, VITS and FastSpeech2 is supported.')

def get_vocoder(model, export_config):
    if (
        isinstance(model, HiFiGANGenerator)
        or isinstance(model, MelGANGenerator)
        or isinstance(model, ParallelWaveGANGenerator)
        or isinstance(model, StyleMelGANGenerator)
    ):
        return OnnxVocoder(model), True
    
    if hasattr(model, 'vocoder'):
        if isinstance(model.vocoder, Spectrogram2Waveform):
            return model, False
        elif isinstance(model.vocoder, ParallelWaveGANPretrainedVocoder):
            raise RuntimeError('Pretrained PWGVocoder is currently not supported.')
    
    if hasattr(model, 'model'):
        if isinstance(model.model.tts, espnetJointText2Wav):
            return get_vocoder(model.model.tts.generator['vocoder'], export_config)
