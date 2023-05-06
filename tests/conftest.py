import glob
import os
from pathlib import Path

import pytest
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,
    DynamicConvolutionTransformerDecoder,
    LightweightConvolution2DTransformerDecoder,
    LightweightConvolutionTransformerDecoder, TransformerDecoder)
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.contextual_block_conformer_encoder import \
    ContextualBlockConformerEncoder
from espnet2.asr.encoder.contextual_block_transformer_encoder import \
    ContextualBlockTransformerEncoder
from espnet2.asr.encoder.hubert_encoder import (FairseqHubertEncoder,
                                                FairseqHubertPretrainEncoder)
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.lm.abs_model import AbsLM
from espnet2.lm.seq_rnn_lm import SequentialRNNLM
from espnet2.lm.transformer_lm import TransformerLM
from espnet2.train.class_choices import ClassChoices

from espnet_onnx.export import ASRModelExport, TTSModelExport
from espnet_onnx.utils.config import get_config

try:
    from espnet2.asr.transducer.transducer_decoder import TransducerDecoder
except:
    from espnet2.asr.decoder.transducer_decoder import TransducerDecoder

from espnet2.gan_tts.hifigan import HiFiGANGenerator
from espnet2.gan_tts.jets import JETS
from espnet2.gan_tts.joint import JointText2Wav
from espnet2.gan_tts.melgan import MelGANGenerator
from espnet2.gan_tts.parallel_wavegan import ParallelWaveGANGenerator
from espnet2.gan_tts.style_melgan import StyleMelGANGenerator
from espnet2.gan_tts.vits import VITS
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.fastspeech import FastSpeech
from espnet2.tts.fastspeech2 import FastSpeech2
from espnet2.tts.tacotron2 import Tacotron2
from espnet2.tts.transformer import Transformer


def pytest_addoption(parser):
    parser.addoption(
        "--config_dir",
        action="store",
        default=None,
        type=str,
        help="Path to the config directory.",
    )
    parser.addoption(
        "--wav_dir",
        action="store",
        default=None,
        type=str,
        help="Path to the wav files for integration test.",
    )


@pytest.fixture
def load_config(request):
    config_dir = request.config.getoption("--config_dir")

    def _method(config_name, model_type="encoder"):
        return get_config(os.path.join(config_dir, model_type, config_name + ".yml"))

    return _method


@pytest.fixture
def model_export():
    return ASRModelExport(Path.home() / ".cache" / "espnet_onnx")


@pytest.fixture
def model_export_tts():
    return TTSModelExport(Path.home() / ".cache" / "espnet_onnx")


class_choices = {
    "frontend": ClassChoices(
        name="frontend",
        classes=dict(
            default=DefaultFrontend,
            sliding_window=SlidingWindow,
            s3prl=S3prlFrontend,
            fused=FusedFrontends,
        ),
        type_check=AbsFrontend,
        default="default",
    ),
    "encoder": ClassChoices(
        "encoder",
        classes=dict(
            conformer=ConformerEncoder,
            transformer=TransformerEncoder,
            contextual_block_transformer=ContextualBlockTransformerEncoder,
            contextual_block_conformer=ContextualBlockConformerEncoder,
            vgg_rnn=VGGRNNEncoder,
            rnn=RNNEncoder,
            wav2vec2=FairSeqWav2Vec2Encoder,
            hubert=FairseqHubertEncoder,
            hubert_pretrain=FairseqHubertPretrainEncoder,
        ),
        type_check=AbsEncoder,
        default="rnn",
    ),
    "decoder": ClassChoices(
        "decoder",
        classes=dict(
            transformer=TransformerDecoder,
            lightweight_conv=LightweightConvolutionTransformerDecoder,
            lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
            dynamic_conv=DynamicConvolutionTransformerDecoder,
            dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
            rnn=RNNDecoder,
            transducer=TransducerDecoder,
        ),
        type_check=AbsDecoder,
        default="rnn",
    ),
    "lm": ClassChoices(
        "lm",
        classes=dict(
            seq_rnn=SequentialRNNLM,
            transformer=TransformerLM,
        ),
        type_check=AbsLM,
        default="seq_rnn",
    ),
    "tts": ClassChoices(
        "tts",
        classes=dict(
            tacotron2=Tacotron2,
            transformer=Transformer,
            fastspeech=FastSpeech,
            fastspeech2=FastSpeech2,
            # NOTE(kan-bayashi): available only for inference
            vits=VITS,
            joint_text2wav=JointText2Wav,
            jets=JETS,
        ),
        type_check=AbsTTS,
        default="tacotron2",
    ),
    "vocoder": ClassChoices(
        "vocoder",
        classes=dict(
            hifigan_generator=HiFiGANGenerator,
            melgan_generator=MelGANGenerator,
            parallel_wavegan_generator=ParallelWaveGANGenerator,
            style_melgan_generator=StyleMelGANGenerator,
        ),
    ),
}


@pytest.fixture
def get_class():
    def _method(model_type, class_name, class_config, **kwargs):
        cc = class_choices[model_type]
        selected_class = cc.get_class(class_name)
        return selected_class(**kwargs, **class_config)

    return _method


@pytest.fixture
def wav_files(request):
    wav_dir = request.config.getoption("--wav_dir")
    return glob.glob(os.path.join(wav_dir, "*"))
