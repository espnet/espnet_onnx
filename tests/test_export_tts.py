
# This test suite verifies that espnet_onnx exports
# model correctly and match the result.

import os
import pytest
from pathlib import Path
import torch

from espnet_onnx.export.tts.models.tts_models.vits import OnnxVITSModel

tts_cases = [
    ['vits', OnnxVITSModel],
]


def save_model(torch_model, onnx_model, model_export, model_type, model_name):
    export_dir = Path(model_export.cache_dir) / 'test' / \
        model_type / f'./cache_{model_name}'
    export_dir.mkdir(parents=True, exist_ok=True)
    
    if model_type == 'tts':
        model_export._export_tts(onnx_model, export_dir, verbose=False)
    
    torch.save(torch_model.state_dict(), str(export_dir / f'{model_type}.pth'))
    return export_dir


@pytest.mark.parametrize('tts_type, cls', tts_cases)
def test_export_tts(tts_type, cls, load_config, model_export_tts, get_class):
    model_config = load_config(tts_type, model_type='tts')
    tts = get_class(
        'tts',
        model_config.tts,
        model_config.tts_conf,
        idim=78, odim=513
    )
    tts_wrapper = cls(tts)
    export_dir = save_model(tts, tts_wrapper, model_export_tts, 'tts', tts_type)
    assert len(os.path.join(export_dir, '*.onnx')) > 0
