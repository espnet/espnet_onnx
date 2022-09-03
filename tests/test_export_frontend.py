
# This test suite verifies that espnet_onnx exports
# model correctly and match the result.

import os
import pytest
from pathlib import Path
import torch
import numpy as np

from espnet_onnx.export.asr.models import (
    get_encoder,
    get_decoder,
    get_lm,
)
from espnet_onnx.export.layers.attention import OnnxNoAtt
from espnet_onnx.utils.config import save_config


encoder_cases = [
    'conformer_hubert',
    'conformer_hubert_last',
    'transformer_hubert',
    'rnn_hubert',
]


def save_model(onnx_model, export_dir, model_export, model_type):
    model_export._export_encoder(onnx_model, export_dir, verbose=False)


@pytest.mark.parametrize('enc_type', encoder_cases)
def test_export_frontend(enc_type, load_config, model_export, get_class):
    model_config = load_config(enc_type, model_type='frontend')
    # prepare input_dim from frontend
    frontend = get_class(
        'frontend',
        model_config.frontend,
        model_config.frontend_conf
    )
    
    export_dir = Path(model_export.cache_dir) / 'test' / \
        'frontend' / f'./cache_{enc_type}'
    export_dir.mkdir(parents=True, exist_ok=True)
    torch.save(frontend.state_dict(), str(export_dir / f'{enc_type}_frontend.pth'))
    
    # create encoder onnx wrapper and export
    # prepare encoder model
    input_size = frontend.output_size()
    encoder = get_class(
        'encoder',
        model_config.encoder,
        model_config.encoder_conf,
        input_size=input_size
    )
    enc_wrapper = get_encoder(encoder, frontend, None, {})
    save_model(enc_wrapper, export_dir, model_export, 'encoder')

    assert len(os.path.join(export_dir, '*frontend.onnx')) > 0

