import os
import pytest
from pathlib import Path

from espnet_onnx.utils.config import get_config
from espnet_onnx.export import ASRModelExport

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import DynamicConvolutionTransformerDecoder
from espnet2.asr.decoder.transformer_decoder import (
    LightweightConvolution2DTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import (
    LightweightConvolutionTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder

from espnet2.asr.encoder.hubert_encoder import FairseqHubertEncoder
from espnet2.asr.encoder.hubert_encoder import FairseqHubertPretrainEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,  # noqa: H301
)
from espnet2.asr.encoder.contextual_block_conformer_encoder import (
    ContextualBlockConformerEncoder,  # noqa: H301
)
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.transducer.transducer_decoder import TransducerDecoder
from espnet2.lm.abs_model import AbsLM
from espnet2.lm.seq_rnn_lm import SequentialRNNLM
from espnet2.lm.transformer_lm import TransformerLM
from espnet2.train.class_choices import ClassChoices


def pytest_addoption(parser):
    parser.addoption('--config_dir', action='store',
                     default=None, type=str,
                     help='Path to the config directory.')


@pytest.fixture
def load_config(request):
    config_dir = request.config.getoption('--config_dir')

    def _method(config_name, model_type='encoder'):
        return get_config(os.path.join(config_dir, model_type, config_name + '.yml'))
    return _method


@pytest.fixture
def get_config_path(request):
    config_dir = request.config.getoption('--config_dir')

    def _method(config_name, model_type='encoder'):
        return os.path.join(config_dir, model_type, config_name + '.yml')
    return _method


@pytest.fixture
def model_export():
    return ASRModelExport(Path.home() / ".cache" / "espnet_onnx")


@pytest.fixture
def frontend_choices():
    return ClassChoices(
        name="frontend",
        classes=dict(
            default=DefaultFrontend,
            sliding_window=SlidingWindow,
            s3prl=S3prlFrontend,
            fused=FusedFrontends,
        ),
        type_check=AbsFrontend,
        default="default",
    )


@pytest.fixture
def encoder_choices():
    return ClassChoices(
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
    )


@pytest.fixture
def decoder_choices():
    return ClassChoices(
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
    )


@pytest.fixture
def lm_choices():
    return ClassChoices(
        "lm",
        classes=dict(
            seq_rnn=SequentialRNNLM,
            transformer=TransformerLM,
        ),
        type_check=AbsLM,
        default="seq_rnn",
    )
