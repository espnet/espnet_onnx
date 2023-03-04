import copy

import librosa
import pytest

from espnet_onnx import Speech2Text as onnxSpeech2Text

from .test_utils import *

asr_config_names = [
    # 'original/rnn',
    "original/transformer",
    "original/conformer",
    "original/conformer_transducer",
    "custom/conformer_cpu",
    "custom/conformer_gpu",
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
