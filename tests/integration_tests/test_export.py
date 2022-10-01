# This test suite verifies that espnet_onnx exports
# model correctly and match the result.

import os
import shutil
import glob
import pytest
from pathlib import Path
import numpy as np

from ..op_test_utils import check_op_type_count


asr_config_names = [
    'original/rnn',
    'original/transformer',
    'original/conformer',
    'custom/conformer_cpu',
    'custom/conformer_gpu',
]


def check_models(cache_dir, tag_name, check_export):
    file_paths = {'full': {}, 'quantize': {}}
    # test full model
    for emt in check_export:
        # test if models are properly exported
        test_path = glob.glob(os.path.join(cache_dir, tag_name.replace(' ', '-'), 'full', f'*{emt}.onnx'))
        assert len(test_path) == 1
        file_paths['full'][emt] = test_path[0]

    # test quantized model
    for emt in check_export:
        # test if models are properly exported
        test_path = glob.glob(os.path.join(cache_dir, tag_name.replace(' ', '-'), 'quantize', f'*{emt}_qt.onnx'))
        assert len(test_path) == 1
        file_paths['quantize'][emt] = test_path[0]

    return file_paths


@pytest.mark.parametrize('config', asr_config_names)
def test_asr_cpu_orig(config, load_config, model_export):
    model = load_config(config, model_type='integration')
    # test exportation for CPU with original onnxruntime
    model_export.set_export_config(
        max_seq_len=5000,
        use_ort_for_espnet=model['use_ort_for_espnet'],
        use_gpu=('GPU' in model['device'])
    )
    model_export.export_from_pretrained(
        model['tag_name'],
        quantize=('Quantize' in model['device']),
        optimize=True
    )
    file_paths = check_models(model_export.cache_dir, model['tag_name'], model['check_export'])

    for device in model['device']:
        if device is not None:
            if device == 'Quantize':
                for k in  model['optimization']['Quantize'].keys():
                    check_op_type_count(
                        file_paths['quantize'][k], model['optimization']['Quantize'][k]
                    )
            else:
                for k in  model['optimization'][device].keys():
                    check_op_type_count(
                        file_paths['full'][k], model['optimization'][device][k]
                    )
