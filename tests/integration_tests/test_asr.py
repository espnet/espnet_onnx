# This test suite verifies that espnet_onnx exports
# model correctly and match the result.

import os
import glob
import pytest
import librosa

from espnet2.bin.asr_inference import Speech2Text as espnetSpeech2Text
from espnet_onnx import Speech2Text as onnxSpeech2Text

from ..op_test_utils import check_op_type_count



asr_config_names = [
    # 'original/rnn',
    'original/transformer',
    'original/conformer',
    'custom/conformer_cpu',
    'custom/conformer_gpu',
]


def export_model(model_export, config):
    model_export.set_export_config(
        max_seq_len=5000,
        use_ort_for_espnet=config['use_ort_for_espnet'],
        use_gpu=('GPU' in config['device'])
    )
    model_export.export_from_zip(
        config['model_dir'],
        config['tag_name'],
        quantize=('Quantize' in config['device']),
        optimize=(config['check_optimize'] is not None)
    )

def check_models(cache_dir, tag_name, check_export, check_quantize):
    file_paths = {'full': {}, 'quantize': {}}
    # test full model
    for emt in check_export:
        # test if models are properly exported
        test_path = glob.glob(os.path.join(cache_dir, tag_name.replace(' ', '-'), 'full', f'*{emt}.onnx'))
        assert len(test_path) == 1
        file_paths['full'][emt] = test_path[0]

    # test quantized model
    if check_quantize:
        for emt in check_export:
            # test if models are properly exported
            test_path = glob.glob(os.path.join(cache_dir, tag_name.replace(' ', '-'), 'quantize', f'*{emt}_qt.onnx'))
            assert len(test_path) == 1
            file_paths['quantize'][emt] = test_path[0]

    return file_paths

def check_optimize(model_config, file_paths):
    for device in model_config['device']:
        if device is not None and model_config['optimization'][device] is not None:
            if device == 'Quantize':
                for k in  model_config['optimization']['Quantize'].keys():
                    check_op_type_count(
                        file_paths['quantize'][k], model_config['optimization']['Quantize'][k]
                    )
            else:
                for k in  model_config['optimization'][device].keys():
                    check_op_type_count(
                        file_paths['full'][k], model_config['optimization'][device][k]
                    )

@pytest.mark.parametrize('config', asr_config_names)
def test_asr(config, load_config, wav_files, model_export):
    model_config = load_config(config, model_type='integration')
    # test export
    export_model(model_export, model_config)
    file_paths = check_models(
        model_export.cache_dir,
        model_config['tag_name'],
        model_config['check_export'],
        ('Quantize' in model_config['device'])
    )
    check_optimize(model_config, file_paths)

    # parity check with espnet model
    espnet_model = espnetSpeech2Text.from_pretrained(model_config['model_dir'])
    onnx_model = onnxSpeech2Text(model_config['tag_name'])
    for wav_file in wav_files:
        y, _ = librosa.load(wav_file, sr=16000)
        espnet_output = espnet_model(y)[0]
        onnx_output = onnx_model(y)[0]

        assert espnet_output[0] == onnx_output[0] # check output sentence
        # assert espnet_
