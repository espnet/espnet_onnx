import glob
import os
from pathlib import Path

import pytest
from espnet2.lm.abs_model import AbsLM
from espnet2.lm.seq_rnn_lm import SequentialRNNLM
from espnet2.lm.transformer_lm import TransformerLM
from espnet2.train.class_choices import ClassChoices

from espnet_onnx.export import ASRModelExport, TTSModelExport
from espnet_onnx.utils.config import get_config

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

from espnet2.tasks.asr import (
    encoder_choices,
    frontend_choices,
    decoder_choices,
)


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
def custom_dir_model_export():
    return ASRModelExport("./test_export_dir")

@pytest.fixture
def model_export_tts():
    return TTSModelExport(Path.home() / ".cache" / "espnet_onnx")


class_choices = {
    "frontend": frontend_choices,
    "encoder": encoder_choices,
    "decoder": decoder_choices,
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


@pytest.fixture
def get_convert_map():
    return Path(os.path.dirname(__file__)).parent / "espnet_onnx" / "export" / "convert_map.yml"
