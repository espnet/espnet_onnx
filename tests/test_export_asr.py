
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
from espnet_onnx.export.asr.models.decoders.attention import OnnxNoAtt
from espnet_onnx.utils.config import save_config


encoder_cases = [
    'conformer_abs_pos',
    'conformer_rel_pos',
    'conformer_rpe_latest',
    'conformer_scaled',
    'transformer',
    'rnn_rnn',
    'rnn_rnnp',
    'rnn_vggrnn',
    'contextual_block_conformer',
    'contextual_block_transformer'
]

decoder_cases = [
    'transformer',
    'lightweight_conv',
    'lightweight_conv2d',
    # 'dynamic_conv',
    # 'dynamic_conv2d',
    'rnn_noatt',
    'rnn_dot',
    'rnn_add',
    'rnn_loc',
    # 'rnn_loc2d'
    'rnn_coverage',
    'rnn_covloc',
    'transducer',
]

lm_cases = [
    'transformer',
    'seqrnn',
    'transformer_pe'
]


def export_predec(dec_wrapper, model_export, export_dir):
    model_export._export_predecoder(dec_wrapper, export_dir, verbose=False)
    for i,a in enumerate(dec_wrapper.att_list):
        if not isinstance(a, OnnxNoAtt):
            assert os.path.isfile(os.path.join(export_dir, f'predecoder_{i}.onnx'))


def save_model(torch_model, onnx_model, model_export, model_type, model_name):
    export_dir = Path(model_export.cache_dir) / 'test' / \
        model_type / f'./cache_{model_name}'
    export_dir.mkdir(parents=True, exist_ok=True)
    
    if model_type == 'encoder':
        model_export._export_encoder(onnx_model, export_dir, verbose=False)
    elif model_type == 'decoder':
        model_export._export_decoder(onnx_model, 512, export_dir, verbose=False)
    elif model_type == 'lm':
        model_export._export_lm(onnx_model, export_dir, verbose=False)
    
    torch.save(torch_model.state_dict(), str(export_dir / f'{model_type}.pth'))
    return export_dir


@pytest.mark.parametrize('enc_type', encoder_cases)
def test_export_encoder(enc_type, load_config, model_export, get_class):
    model_config = load_config(enc_type, model_type='encoder')
    # prepare input_dim from frontend
    frontend = get_class(
        'frontend',
        model_config.frontend,
        model_config.frontend_conf
    )
    input_size = frontend.output_size()
    
    # prepare encoder model
    encoder = get_class(
        'encoder',
        model_config.encoder,
        model_config.encoder_conf,
        input_size=input_size
    )
    
    # create encoder onnx wrapper and export
    enc_wrapper = get_encoder(encoder)
    export_dir = save_model(encoder, enc_wrapper, model_export, 'encoder', enc_type)
    
    if enc_type in ('contextual_block_conformer', 'contextual_block_transformer'):
        # save position encoder parameters.
        np.save(
            export_dir / 'pe',
            encoder.pos_enc.pe.numpy()
        )
    assert os.path.isfile(os.path.join(export_dir, 'encoder.onnx'))


@pytest.mark.parametrize('dec_type', decoder_cases)
def test_export_decoder(dec_type, load_config, model_export, get_class):
    model_config = load_config(dec_type, model_type='decoder')
    
    # prepare encoder model
    if model_config.decoder == 'transducer':
        kwargs = { 'vocab_size': 32000, 'embed_pad': 0 }
    else:
        kwargs = { 'vocab_size': 32000, 'encoder_output_size': 512 }
    
    decoder = get_class(
        'decoder',
        model_config.decoder,
        model_config.decoder_conf,
        **kwargs
    )
    dec_wrapper = get_decoder(decoder)
    export_dir = save_model(decoder, dec_wrapper, model_export, 'decoder', dec_type)
    
    if dec_type[:3] == 'rnn':
        export_predec(dec_wrapper, model_export, export_dir)
        
    decoder_config = dec_wrapper.get_model_config(export_dir)
    save_config(decoder_config, export_dir / 'config.yaml')
    assert os.path.isfile(os.path.join(export_dir, 'decoder.onnx'))


@pytest.mark.parametrize('lm_type', lm_cases)
def test_export_lm(lm_type, load_config, model_export, get_class):
    model_config = load_config(lm_type, model_type='lm')
    
    # prepare language model
    lm = get_class(
        'lm',
        model_config.lm,
        model_config.lm_conf,
        vocab_size=32000,
    )
    lm_wrapper = get_lm(lm)
    export_dir = save_model(lm, lm_wrapper, model_export, 'lm', lm_type)
    
    lm_config = {'lm': lm_wrapper.get_model_config(export_dir)}
    lm_config['lm'].update({'use_lm': True})
    save_config(lm_config, export_dir / 'config.yaml')
    assert os.path.isfile(os.path.join(export_dir, 'lm.onnx'))
