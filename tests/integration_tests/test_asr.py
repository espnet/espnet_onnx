import copy

import librosa
import pytest
import numpy as np

from espnet_onnx import Speech2Text as onnxSpeech2Text

from .test_utils import *

asr_config_names = [
    # 'original/rnn',
    "original/transformer",
    "custom/transformer",
    "original/conformer",
    "original/conformer_transducer",
    "custom/conformer_cpu",
    "custom/conformer_gpu",
]

asr_config_custom_dir = [
    "original/conformer"
]

asr_without_frontend = [
    "original/conformer"
]


@pytest.mark.parametrize("asr_config_names", asr_config_names)
def test_asr(asr_config_names, load_config, wav_files, model_export):
    config = load_config(asr_config_names, model_type="integration")
    config.tag_name = "test/integration/" + config.tag_name

    # build ASR model
    espnet_model = build_model(config.model_config)

    # test export
    export_model(model_export, copy.deepcopy(espnet_model), config)
    file_paths = check_models(
        model_export.cache_dir,
        config.tag_name,
        config.check_export,
        ("Quantize" in config.device),
    )
    check_optimize(config, file_paths)
    eval_model(espnet_model)

    # parity check with espnet model
    onnx_model = onnxSpeech2Text(config.tag_name)
    for wav_file in wav_files:
        y, _ = librosa.load(wav_file, sr=16000)
        espnet_output = espnet_model(y)[0]
        onnx_output = onnx_model(y)[0]

        assert espnet_output[2] == onnx_output[2]


@pytest.mark.parametrize("asr_config_names", asr_config_custom_dir)
def test_asr_custom_dir(asr_config_names, load_config, wav_files, custom_dir_model_export):
    config = load_config(asr_config_names, model_type="integration")
    config.tag_name = "test/integration/" + config.tag_name

    # build ASR model
    espnet_model = build_model(config.model_config)

    # test export
    export_model(custom_dir_model_export, copy.deepcopy(espnet_model), config)
    file_paths = check_models(
        custom_dir_model_export.cache_dir,
        config.tag_name,
        config.check_export,
        ("Quantize" in config.device),
    )
    check_optimize(config, file_paths)
    eval_model(espnet_model)

    # parity check with espnet model
    onnx_model = onnxSpeech2Text(config.tag_name, cache_dir=custom_dir_model_export.cache_dir)
    for wav_file in wav_files:
        y, _ = librosa.load(wav_file, sr=16000)
        espnet_output = espnet_model(y)[0]
        onnx_output = onnx_model(y)[0]

        assert espnet_output[2] == onnx_output[2]


@pytest.mark.parametrize("asr_config_names", asr_without_frontend)
def test_asr_without_frontend(asr_config_names, load_config, custom_dir_model_export):
    config = load_config(asr_config_names, model_type="integration")
    config.tag_name = "test/integration/" + config.tag_name

    # build ASR model
    espnet_model = build_model(config.model_config)
    espnet_model.asr_model.frontend = None

    # test export
    export_model(custom_dir_model_export, copy.deepcopy(espnet_model), config)
    file_paths = check_models(
        custom_dir_model_export.cache_dir,
        config.tag_name,
        config.check_export,
        False,
    )
    check_optimize(config, file_paths)
    eval_model(espnet_model)

    # parity check with espnet model
    onnx_model = onnxSpeech2Text(config.tag_name, cache_dir=custom_dir_model_export.cache_dir)
    for feat_length in [100, 200]:
        y = np.random.rand(feat_length, 80)
        espnet_output = espnet_model(y)[0]
        onnx_output = onnx_model(y)[0]

        assert espnet_output[2] == onnx_output[2]
