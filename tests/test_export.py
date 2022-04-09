
# This test suite verifies that espnet_onnx exports
# model correctly and match the result.

import os
import glob
import pytest
from pathlib import Path

from espnet_onnx.export.asr.models import (
    get_encoder,
    get_decoder,
    RNNDecoder,
    PreDecoder,
    CTC,
    LanguageModel
)
from espnet2.lm.espnet_model import ESPnetLanguageModel


@pytest.mark.parametrize('enc_type', [
    'conformer_abs_pos', 'conformer_rel_pos',
    'conformer_rpe_latest', 'conformer_scaled',
    'transformer', 'rnn_rnn', 'rnn_rnnp', 'rnn_vggrnn'
])
def test_export_encoder(enc_type, load_config, model_export,
                        frontend_choices, encoder_choices):
    model_config = load_config('export', enc_type, model_type='encoder')
    # prepare input_dim from frontend
    frontend_class = frontend_choices.get_class(model_config.frontend)
    frontend = frontend_class(**model_config.frontend_conf.dic)
    input_size = frontend.output_size()
    # prepare encoder model
    encoder_class = encoder_choices.get_class(model_config.encoder)
    encoder = encoder_class(input_size=input_size, **
                            model_config.encoder_conf.dic)
    # create encoder onnx wrapper and export
    enc_wrapper = get_encoder(encoder)
    export_dir = Path(model_export.cache_dir) / 'test' / \
        'encoder' / f'./cache_{enc_type}'
    export_dir.mkdir(parents=True, exist_ok=True)
    model_export._export_encoder(enc_wrapper, export_dir)
    assert os.path.isfile(os.path.join(export_dir, 'encoder.onnx'))


@pytest.mark.parametrize('dec_type', [
    'transformer', 'rnn'
])
def test_export_decoder(dec_type, load_config, model_export, decoder_choices):
    model_config = load_config('export', dec_type, model_type='decoder')
    # prepare encoder model
    decoder_class = decoder_choices.get_class(model_config.decoder)
    decoder = decoder_class(
        vocab_size=32000,
        encoder_output_size=512,
        **model_config.decoder_conf.dic,
    )
    # create onnx wrapper and export
    dec_wrapper = get_decoder(decoder)
    export_dir = Path(model_export.cache_dir) / 'test' / \
        'decoder' / f'./cache_{dec_type}'
    export_dir.mkdir(parents=True, exist_ok=True)
    model_export._export_decoder(dec_wrapper, 512, export_dir)
    assert os.path.isfile(os.path.join(export_dir, 'decoder.onnx'))


@pytest.mark.parametrize('dec_type', ['rnn'])
def test_export_predec(dec_type, load_config, model_export, decoder_choices):
    model_config = load_config('export', dec_type, model_type='decoder')
    # prepare encoder model
    decoder_class = decoder_choices.get_class(model_config.decoder)
    decoder = decoder_class(
        vocab_size=32000,
        encoder_output_size=512,
        **model_config.decoder_conf.dic,
    )
    # create onnx wrapper and export
    dec_wrapper = get_decoder(decoder)
    export_dir = Path(model_export.cache_dir) / 'test' / \
        'pre_decoder' / f'./cache_{dec_type}'
    export_dir.mkdir(parents=True, exist_ok=True)
    model_export._export_predecoder(dec_wrapper, export_dir)
    assert os.path.isfile(os.path.join(export_dir, 'predecoder_0.onnx'))


@pytest.mark.parametrize('lm_type', [
    'transformer', 'rnn', 'transformer_pe'
])
def test_export_lm(lm_type, load_config, model_export, lm_choices):
    model_config = load_config('export', lm_type, model_type='lm')
    # prepare language model
    lm_class = lm_choices.get_class(model_config.lm)
    lm = lm_class(vocab_size=32000, **model_config.lm_conf.dic)
    model = ESPnetLanguageModel(
        lm=lm, vocab_size=32000, **model_config.model_conf.dic)
    # create onnx wrapper and export
    lm_model = LanguageModel(model.lm)
    export_dir = model_export.cache_dir / 'test' / 'lm' / f'./cache_{lm_type}'
    export_dir.mkdir(parents=True, exist_ok=True)
    model_export._export_lm(lm_model, export_dir)
    assert os.path.isfile(os.path.join(export_dir, 'lm.onnx'))
