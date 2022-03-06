from typing import Union
from typing import Tuple
from pathlib import Path
from typeguard import check_argument_types
import os
import glob
import json

import numpy as np
import torch
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.bin.asr_inference import Speech2Text

import onnx
from onnxruntime.quantization import quantize_dynamic

from .asr_models import Encoder
from .asr_models import Decoder
from .asr_models import CTC
from .get_config import get_encoder_config
from .get_config import get_decoder_config
from .get_config import get_transducer_config
from .get_config import get_lm_config
from .get_config import get_ngram_config
from .get_config import get_beam_config
from .get_config import get_token_config
from espnet_onnx.utils.function import make_pad_mask
from espnet_onnx.utils.function import subsequent_mask


def export_encoder(model, feats, path):
    enc_input_names = ['feats', 'mask']
    enc_output_names = ['encoder_out', 'encoder_out_lens']
    mask = torch.from_numpy(make_pad_mask(
        np.array([feats.shape[1]]))[:, None, :])
    dynamic_axes = {
        'feats': {
            1: 'feats_length'
        },
        'mask': {
            2: 'mask_length'
        }
    }
    file_name = os.path.join(path, 'encoder.onnx')
    torch.onnx.export(
        Encoder(model),
        (feats, mask),
        file_name,
        verbose=False,
        opset_version=11,
        input_names=enc_input_names,
        output_names=enc_output_names,
        dynamic_axes=dynamic_axes
    )
    return model(feats, torch.LongTensor([feats.shape[1]]))


def export_decoder(model, x, path):
    tgt = torch.LongTensor([0, 4999]).unsqueeze(0)
    tgt_mask = torch.from_numpy(subsequent_mask(2)[np.newaxis, :])
    # get cache size
    emb_out = model.embed(tgt)
    dec_out, *_ = model.decoders[0](emb_out, tgt_mask, x, None)
    cache = [
        torch.zeros((1, 1, dec_out.shape[-1]))
        for _ in range(len(model.decoders))
    ]
    dec_input_names = ['tgt', 'tgt_mask', 'memory'] \
        + ['cache_%d' % i for i in range(len(cache))]
    dec_output_names = ['y'] \
        + ['out_cache_%d' % i for i in range(len(cache))]
    dynamic_axes = {
        'tgt': {
            0: 'tgt_batch',
            1: 'tgt_length'
        },
        'tgt_mask': {
            1: 'tgt_mask_length',
            2: 'tgt_mask_height'
        },
        'memory': {
            0: 'memory_batch',
            1: 'memory_length'
        }
    }
    dynamic_axes.update({
        'cache_%d' % d: {
            0: 'cache_%d_batch' % d,
            1: 'cache_%d_length' % d
        }
        for d in range(len(model.decoders))
    })
    dynamic_axes.update({
        'out_cache_%d' % d: {
            0: 'out_cache_%d_batch' % d,
            1: 'out_cache_%d_length' % d
        }
        for d in range(len(model.decoders))
    })
    file_name = os.path.join(path, 'decoder.onnx')
    torch.onnx.export(
        Decoder(model),
        (tgt, tgt_mask, x, cache),
        file_name,
        verbose=True,
        opset_version=11,
        input_names=dec_input_names,
        output_names=dec_output_names,
        dynamic_axes=dynamic_axes
    )
    return dec_out.shape[-1]


def export_ctc(model, x, path):
    ctc_input_names = ['x']
    ctc_output_names = ['ctc_out']
    file_name = os.path.join(path, 'ctc.onnx')
    dynamic_axes = {
        "x": {1: "ctc_in_length" },
        "ctc_out": {1: "ctc_out_length"}
    }
    torch.onnx.export(
        CTC(model),
        (x),
        file_name,
        verbose=True,
        opset_version=11,
        input_names=ctc_input_names,
        output_names=ctc_output_names,
        dynamic_axes=dynamic_axes
    )


def create_config(model, path, decoder_odim):
    ret = {}
    # encoder
    ret.update(encoder=get_encoder_config(model.asr_model, path))
    ret.update(decoder=get_decoder_config(model.asr_model, path, decoder_odim))
    ret.update(transducer=get_transducer_config(model.asr_model, path))
    ret.update(ctc=dict(model_path=os.path.join(path, "ctc.onnx")))

    if "lm" in model.beam_search.nn_dict.keys():
        ret.update(lm=get_lm_config(model.asr_model))
    else:
        ret.update(lm=dict(use_lm=False))

    if "ngram" in list(model.beam_search.full_scorers.keys()) \
            + list(model.beam_search.part_scorers.keys()):
        ret.update(ngram=get_ngram_config(model))
    else:
        ret.update(ngram=dict(use_ngram=False))

    ret.update(weights=model.beam_search.weights)
    ret.update(beam_search=get_beam_config(model.beam_search))
    ret.update(token=get_token_config(model.asr_model))
    ret.update(bpemodel=model.tokenizer.model)
    return ret


def quantize(model_from, model_to):
    models = glob.glob(os.path.join(model_from, "*.onnx"))
    for m in models:
        basename = os.path.basename(m).split('.')[0]
        quantize_dynamic(
            m,
            os.path.join(model_to, basename + '_qt.onnx')
        )


def export_model(
    model: Speech2Text,
    onnx_path: Union[Path, str],
    create_qt: bool = False,
    qt_path: Union[Path, str] = None
):
    assert check_argument_types()

    if not os.path.exists(onnx_path):
        os.mkdir(onnx_path)

    sample_feat = torch.randn((1, 100, 80))
    enc_out = export_encoder(model.asr_model.encoder, sample_feat, onnx_path)
    if isinstance(enc_out, Tuple):
        enc_out = enc_out[0]

    decoder_odim = export_decoder(model.asr_model.decoder, enc_out, onnx_path)
    export_ctc(model.asr_model.ctc.ctc_lo, enc_out, onnx_path)

    model_config = create_config(model, onnx_path, decoder_odim)
    config_name = os.path.join(onnx_path, 'config.json')
    with open(config_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(model_config))

    if create_qt:
        if qt_path is None:
            raise Error('You have to specify export path when creating quantized model.')
             
        if not os.path.exists(qt_path):
            os.mkdir(qt_path)
        quantize(onnx_path, qt_path)


def export_from_pretrained(
    model_name: str,
    onnx_path: Union[Path, str],
    create_qt: bool = False,
    qt_path: Union[Path, str] = None
):
    model = Speech2Text.from_pretrained(model_name)
    export_model(model, onnx_path, create_qt, qt_path)
