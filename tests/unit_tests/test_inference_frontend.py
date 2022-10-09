
# This test suite verifies that espnet_onnx exports
# model correctly and match the result.

import os
import glob
import pytest
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

from espnet_onnx.asr.model.decoder import get_decoder
from espnet_onnx.asr.model.lm import get_lm
from espnet_onnx.utils.config import get_config
from .forward_utils import (
    run_onnx_front
)

encoder_cases = [
    ('conformer_hubert', [16000, 32000]),
    ('conformer_hubert_last', [16000, 32000]),
    ('transformer_hubert', [16000, 32000]),
    ('rnn_hubert', [16000, 32000]),
]


CACHE_DIR = Path.home() / ".cache" / "espnet_onnx" / 'test'
PROVIDERS = ['CPUExecutionProvider']


def check_output(out_t, out_o):
    out_t = out_t.detach().cpu().numpy()
    assert out_t.shape == out_o.shape, \
        f"The shape of output of onnx {out_o.shape} should be the same with the output of torch model {out_t.shape}"

    mean_dif = np.mean((out_t - out_o)**2)
    assert mean_dif < 0.0005, \
        f"Result of torch model and onnx model differs."



@pytest.mark.parametrize('enc_type, wav_lens', encoder_cases)
def test_infer_frontend(enc_type, wav_lens, load_config, get_class):
    model_dir = CACHE_DIR / 'frontend' / f'./cache_{enc_type}'
    model_config = load_config(enc_type, model_type='frontend')
    
    # prepare input_dim from frontend
    frontend_espnet = get_class(
        'frontend',
        model_config.frontend,
        model_config.frontend_conf.dic
    )
    frontend_espnet.load_state_dict(torch.load(glob.glob(str(model_dir / '*frontend.pth'))[0]))
    frontend_espnet.eval()
    model_file = glob.glob(os.path.join(model_dir , '*frontend.onnx'))[0]
    frontend_onnx = ort.InferenceSession(model_file, providers=PROVIDERS)
    
    # test output
    for wl in wav_lens:
        dummy_input = torch.randn(1, wl)  # (B, L, D)
        # compute torch model
        torch_out, _ = frontend_espnet(dummy_input, torch.LongTensor([wl]))
        if type(torch_out) == tuple:
            torch_out = torch_out[0]
        # compute onnx model
        onnx_out = run_onnx_front(frontend_onnx, dummy_input.numpy())
        check_output(torch_out, onnx_out)

