
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


@pytest.mark.parametrize('tts_type, cls', tts_cases)
def test_export_tts(tts_type, cls, load_config, model_export_tts,
                    tts_choices):
    model_config = load_config(tts_type, model_type='tts')
    tts_class = tts_choices.get_class(model_config.tts)
    tts = tts_class(idim=78, odim=513, **model_config.tts_conf.dic)
    tts_wrapper = cls(tts)

    export_dir = Path(model_export_tts.cache_dir) / 'test' / \
        'tts' / f'./cache_{tts_type}'
    export_dir.mkdir(parents=True, exist_ok=True)
    model_export_tts._export_tts(tts_wrapper, export_dir, verbose=False)
    torch.save(tts.state_dict(), str(export_dir / 'tts.pth'))
    assert os.path.isfile(os.path.join(export_dir, 'tts_model.onnx'))
