import os
import pytest


@pytest.mark.parametrize('enc_type', [
    'conformer_abs_pos', 'conformer_rel_pos',
    'conformer_rpe_latest', 'conformer_scaled',
    'transformer', 'rnn'
])
def test_encoder():
    pass


@pytest.mark.parametrize('dec_type', [
    'transformer', 'rnn'
])
def test_decoder():
    pass


@pytest.mark.parametrize('lm_type', [
    'transformer', 'rnn', 'transformer_pe'
])
def test_lm():
    pass
