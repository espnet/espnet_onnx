# This test suite verifies that espnet_onnx exports
# model correctly and match the result.

import os
import glob
import pytest
from pathlib import Path
import numpy as np

from ..op_test_utils import check_op_type_count


asr_tag_names = [
    # RNN encoder and RNN decoder
    {
        'tag_name': 'kamo-naoyuki/timit_asr_train_asr_raw_word_valid.acc.ave',
        'optimization': {'CPU': {}, 'GPU': {}},
        'export_model_types': ['encoder', 'decoder', 'ctc']
    },

    # Transformer encoder and decoder
    {
        'tag_name': 'Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best',
        'optimization': {
            'CPU': {
                'encoder': {
                    'Attention': 12,
                },
                'decoder': {
                    'CrossAttention': 6
                },
                'lm': {
                    'CrossAttention': 12
                }
            },
            'Quantize': {
                'encoder': {
                    'QAttention': 12,
                },
                'decoder': {
                    'QCrossAttention': 6
                },
                'lm': {
                    'QCrossAttention': 12
                }
            },
            'GPU': {
                'encoder': {
                    'Attention': 12
                }
            }
        },
        'export_model_types': ['encoder', 'decoder', 'ctc', 'lm']
    },

    # Conformer encoder and decoder
    {
        'tag_name': 'pyf98/librispeech_conformer_hop_length160',
        'optimization': {
            'CPU': {
                'encoder': {
                    'RelPosAttention': 12
                },
                'decoder': {
                    'CrossAttention': 6,
                },
                'lm': {
                    'CrossAttention': 12
                }
            },
            'Quantize': {
                'encoder': {
                    'QRelPosAttention': 12,
                },
                'decoder': {
                    'QCrossAttention': 6
                },
                'lm': {
                    'QCrossAttention': 12
                }
            },
            'GPU': {
                'encoder': {
                    'RelativeShift': 12
                }
            }
        },
        'export_model_types': ['encoder', 'decoder', 'ctc', 'lm']
    }
]



def check_models(cache_dir, tag_name, export_model_types):
    file_paths = {'full': {}, 'quantize': {}}
    # test full model
    for emt in export_model_types:
        # test if models are properly exported
        test_path = glob.glob(os.path.join(cache_dir, tag_name, 'full', f'*{emt}.onnx'))
        assert len(test_path) == 1
        file_paths['full'][emt] = test_path[0]

    # test quantized model
    for emt in export_model_types:
        # test if models are properly exported
        test_path = glob.glob(os.path.join(cache_dir, tag_name, 'quantize', f'*{emt}_qt.onnx'))
        assert len(test_path) == 1
        file_paths['quantize'][emt] = test_path[0]

    return file_paths


@pytest.mark.parametrize('model', asr_tag_names)
def test_asr_cpu_orig(model, model_export):
    # test exportation for CPU with original onnxruntime
    model_export.set_export_config(
        max_seq_len=5000,
        use_ort_for_espnet=False,
    )
    model_export.export_from_pretrained(
        model['tag_name'],
        quantize=True,
        optimize=True
    )
    file_paths = check_models(model_export.cache_dir, model['tag_name'],model['export_model_types'])
    if 'encoder' in model['export_model_types'] and 'encoder' in model['optimization']['CPU'].keys():
        check_op_type_count(
            file_paths['encoder'], **model['optimization']['CPU']['encoder']
        )
    if 'encoder' in model['export_model_types'] and 'encoder' in model['optimization']['Quantize'].keys():
        check_op_type_count(
            file_paths['encoder'], **model['optimization']['Quantize']['encoder']
        )

@pytest.mark.parametrize('models', asr_tag_names)
def test_asr_cpu_custom(models, model_export):  
    # test exportation for CPU with original onnxruntime
    model_export.set_export_config(
        max_seq_len=5000,
        use_ort_for_espnet=True,
    )
    model_export.export_from_pretrained(
        model['tag_name'],
        quantize=True,
        optimize=True
    )
    file_paths = check_models(model_export.cache_dir, model['tag_name'],model['export_model_types'])
    for k in  model['optimization']['CPU'].keys():
        check_op_type_count(
            file_paths[k], **model['optimization']['CPU'][k]
        )
    for k in  model['optimization']['Quantize'].keys():
        check_op_type_count(
            file_paths[k], **model['optimization']['Quantize'][k]
        )

@pytest.mark.parametrize('models', asr_tag_names)
def test_asr_gpu_custom(models, model_export):  
    # test exportation for CPU with original onnxruntime
    model_export.set_export_config(
        max_seq_len=5000,
        use_ort_for_espnet=True,
        use_gpu=True
    )
    model_export.export_from_pretrained(
        model['tag_name'],
        quantize=True,
        optimize=True
    )
    file_paths = check_models(model_export.cache_dir, model['tag_name'],model['export_model_types'])
    for k in  model['optimization']['GPU'].keys():
        check_op_type_count(
            file_paths[k], **model['optimization']['GPU'][k]
        )

