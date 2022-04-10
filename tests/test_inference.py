
# This test suite verifies that espnet_onnx exports
# model correctly and match the result.

import os
import glob
import pytest
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
from espnet2.lm.espnet_model import ESPnetLanguageModel

from espnet_onnx.export.asr.models import (
    get_encoder,
    get_decoder,
    RNNDecoder,
    PreDecoder,
    CTC,
    LanguageModel
)
from espnet_onnx.asr.model.decoders.rnn import RNNDecoder
from espnet_onnx.asr.model.decoders.xformer import XformerDecoder
from espnet_onnx.asr.model.lm.seqrnn_lm import SequentialRNNLM
from espnet_onnx.asr.model.lm.transformer_lm import TransformerLM
from espnet_onnx.utils.function import (
    subsequent_mask,
    make_pad_mask,
    mask_fill
)
from espnet_onnx.utils.config import get_config
from .forward_utils import (
    rnn_onnx_enc,
    rnn_torch_dec,
    rnn_onnx_dec,
    xformer_onnx_enc,
    xformer_onnx_dec,
    xformer_torch_dec,
)

encoder_cases = [
    ('conformer_abs_pos', [50, 100]),
    ('conformer_rel_pos', [50, 100]),
    ('conformer_rpe_latest', [50, 100]),
    ('conformer_scaled', [50, 100]),
    ('transformer', [50, 100]),
    ('rnn_rnn', [50, 100]),
    ('rnn_rnnp', [50, 100]),
    ('rnn_vggrnn', [50, 100]),
]

decoder_cases = [
    ('transformer', [50, 100]),
    ('rnn_noatt', [50, 100]),
    ('rnn_dot', [50, 100]),
    ('rnn_add', [50, 100]),
    ('rnn_loc', [50, 100]),
    # ('rnn_loc2d', [50, 100]),
    ('rnn_coverage', [50, 100]),
    ('rnn_covloc', [50, 100]),
]

lm_cases = [
    ('transformer', [50, 100]),
    ('seqrnn', [50, 100]),
    ('transformer_pe', [50, 100]),
]

CACHE_DIR = Path.home() / ".cache" / "espnet_onnx" / 'test'


def check_output(out_t, out_o):
    out_t = out_t.detach().cpu().numpy()
    assert out_t.shape == out_o.shape, \
        f"The shape of output of onnx {out_o.shape} should be the same with the output of torch model {out_t.shape}"

    mean_dif = np.mean((out_t - out_o)**2)
    assert mean_dif < 0.05, \
        f"Result of torch model and onnx model differs."


def get_predec_models(dec_type):
    if dec_type[:3] != 'rnn':
        return []

    model_dir = CACHE_DIR / 'pre_decoder' / f'./cache_{dec_type}'
    predecoders = []
    for p in glob.glob(os.path.join(model_dir, '*.onnx')):
        predecoders.append(
            ort.InferenceSession(p)
        )
    return predecoders


@pytest.mark.parametrize('enc_type, feat_lens', encoder_cases)
def test_infer_encoder(enc_type, feat_lens, load_config,
                       model_export, frontend_choices, encoder_choices):
    model_dir = CACHE_DIR / 'encoder' / f'./cache_{enc_type}'
    model_config = load_config(enc_type, model_type='encoder')
    # prepare input_dim from frontend
    frontend_class = frontend_choices.get_class(model_config.frontend)
    frontend = frontend_class(**model_config.frontend_conf.dic)
    input_size = frontend.output_size()
    # prepare encoder model
    encoder_class = encoder_choices.get_class(model_config.encoder)
    encoder_espnet = encoder_class(input_size=input_size, **
                                   model_config.encoder_conf.dic)
    encoder_espnet.load_state_dict(torch.load(str(model_dir / 'encoder.pth')))
    encoder_espnet.eval()
    encoder_onnx = ort.InferenceSession(str(model_dir / 'encoder.onnx'))
    # test output
    for fl in feat_lens:
        dummy_input = torch.randn(1, fl, input_size)  # (B, L, D)
        # compute torch model
        with torch.no_grad():
            torch_out = encoder_espnet(dummy_input, torch.Tensor([fl]))
        if type(torch_out) == tuple:
            torch_out = torch_out[0]
        # compute onnx model
        if enc_type[:3] == 'rnn':
            onnx_out = rnn_onnx_enc(encoder_onnx, dummy_input.numpy())
        else:
            onnx_out = xformer_onnx_enc(encoder_onnx, dummy_input.numpy())
        check_output(torch_out, onnx_out)


@pytest.mark.parametrize('dec_type, feat_lens', decoder_cases)
def test_infer_decoder(dec_type, feat_lens, load_config, model_export, decoder_choices):
    model_dir = CACHE_DIR / 'decoder' / f'./cache_{dec_type}'
    model_config = load_config(dec_type, model_type='decoder')
    # prepare encoder model
    decoder_class = decoder_choices.get_class(model_config.decoder)
    decoder_espnet = decoder_class(
        vocab_size=32000,
        encoder_output_size=512,
        **model_config.decoder_conf.dic,
    )
    decoder_espnet.load_state_dict(torch.load(str(model_dir / 'decoder.pth')))
    decoder_espnet.eval()
    if dec_type[:3] == 'rnn':
        decoder_onnx = RNNDecoder(get_config(model_dir / 'config.yaml'))
    else:
        decoder_onnx = XformerDecoder(get_config(model_dir / 'config.yaml'))
    # test output
    for fl in feat_lens:
        dummy_input = torch.randn(1, fl, 512)
        if dec_type[:3] == 'rnn':
            with torch.no_grad():
                torch_out = rnn_torch_dec(decoder_espnet, dummy_input)
                onnx_out = rnn_onnx_dec(decoder_onnx, dummy_input.numpy())
        else:
            with torch.no_grad():
                torch_out = xformer_torch_dec(decoder_espnet, dummy_input)
                onnx_out = xformer_onnx_dec(decoder_onnx, dummy_input.numpy())

        check_output(torch_out, onnx_out)


@pytest.mark.parametrize('lm_type, feat_lens', lm_cases)
def test_infer_lm(lm_type, feat_lens, load_config, model_export, lm_choices):
    model_dir = CACHE_DIR / 'lm' / f'./cache_{lm_type}'
    model_config = load_config(lm_type, model_type='lm')
    # prepare language model
    lm_class = lm_choices.get_class(model_config.lm)
    lm = lm_class(vocab_size=32000, **model_config.lm_conf.dic)
    torch_model = ESPnetLanguageModel(
        lm=lm, vocab_size=32000, **model_config.model_conf.dic)
    torch_model.lm.load_state_dict(torch.load(str(model_dir / 'lm.pth')))
    torch_model.lm.eval()
    # create onnx wrapper and export
    lm_config = get_config(model_dir / 'config.yaml')
    if lm_config.lm_type == 'SequentialRNNLM':
        onnx_model = SequentialRNNLM(lm_config)
    else:
        onnx_model = TransformerLM(lm_config)
    # test output
    for fl in feat_lens:
        dummy_input = torch.randn(1, fl, 512)
        tgt = torch.LongTensor([0, 1]).unsqueeze(0)
        with torch.no_grad():
            torch_out, _ = torch_model.lm.batch_score(tgt, [None], dummy_input)
        onnx_out, _ = onnx_model.batch_score(
            tgt.numpy(), [None], dummy_input.numpy())
        check_output(torch_out, onnx_out)
