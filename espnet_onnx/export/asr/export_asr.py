from typing import Union
from typing import Tuple
from pathlib import Path
from typeguard import check_argument_types
import os
import glob
import json
from datetime import datetime
import hashlib

import numpy as np
import torch
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.bin.asr_inference import Speech2Text
from espnet2.lm.seq_rnn_lm import SequentialRNNLM
from espnet2.lm.transformer_lm import TransformerLM
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder

import onnx
from onnxruntime.quantization import quantize_dynamic

from .asr_models import Encoder
from .asr_models import Decoder
from .asr_models import CTC
from .lm.lm import SequentialRNNLM as onnxSeqRNNLM
from .lm.lm import TransformerLM as onnxTransformerLM
from .get_config import get_encoder_config
from .get_config import get_decoder_config
from .get_config import get_transducer_config
from .get_config import get_lm_config
from .get_config import get_ngram_config
from .get_config import get_beam_config
from .get_config import get_token_config
from .get_config import get_tokenizer_config
from espnet_onnx.utils.function import make_pad_mask
from espnet_onnx.utils.function import subsequent_mask


def str_to_hash(string: Union[str, Path]) -> str:
    return hashlib.md5(str(string).encode("utf-8")).hexdigest()


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


def export_transformer_dec(model, x, path, sos_token):
    tgt = torch.LongTensor([0, sos_token]).unsqueeze(0)
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


def export_decoder(model, x, path, sos_token):
    if isinstance(model, TransformerDecoder):
        return export_transformer_dec(model, x, path, sos_token)
    elif isinstance(model, RNNDecoder):
        raise ValueError('Currently RNNDecoder is not supported.')


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


def export_seq_rnn(model, path, sos_token):
    tgt = torch.LongTensor([0, sos_token]).unsqueeze(0)
    hidden = torch.randn(model.nlayers, 1, model.nhid)
    file_name = os.path.join(path, 'lm.onnx')
    lm_input_names = ['x', 'in_hidden1']
    lm_output_names = ['y', 'out_hidden1']
    lm_inputs = (tgt, hidden)
    dynamic_axes = {
        'x': {
            0: 'x_batch',
            1: 'x_length'
        },
        'y': {
            0: 'y_batch'
        },
        'in_hidden1': {
            1: 'hidden1_batch'
        },
        'out_hidden1': {
            1: 'out_hidden1_batch'
        }
    }
    if model.rnn_type == 'LSTM':
        lm_input_names += ['in_hidden2']
        lm_output_names += ['out_hidden2']
        lm_inputs = (tgt, hidden, hidden)
        dynamic_axes.update({
            'in_hidden2': {
                1: 'hidden2_batch'
            },
            'out_hidden2': {
                1: 'out_hidden2_batch'
            }
        })
    # export encoder
    torch.onnx.export(
        onnxSeqRNNLM(model),
        lm_inputs,
        file_name,
        verbose=True,
        opset_version=11,
        input_names=lm_input_names,
        output_names=lm_output_names,
        dynamic_axes=dynamic_axes
    )


def export_transformer_lm(model, path, sos_token):
    tgt = torch.LongTensor([0, sos_token]).unsqueeze(0)
    ys_mask = tgt != 0
    m = torch.from_numpy(subsequent_mask(ys_mask.shape[-1])[None, :])
    mask = ys_mask[None, :] * m
    cache = [
        torch.zeros((1, 1, model.encoder.encoders[0].size))
        for _ in range(len(model.encoder.encoders))
    ]
    lm_input_names = ['tgt', 'tgt_mask'] \
        + ['cache_%d' % i for i in range(len(cache))]
    lm_output_names = ['y'] \
        + ['out_cache_%d' % i for i in range(len(cache))]
    dynamic_axes = {
        'tgt': {
            0: 'tgt_batch',
            1: 'tgt_length'
        },
        'tgt_mask': {
            0: 'tgt_mask_batch',
            1: 'tgt_mask_length',
            2: 'tgt_mask_height'
        }
    }
    dynamic_axes.update({
        'cache_%d' % d: {
            0: 'cache_%d_batch' % d,
            1: 'cache_%d_length' % d
        }
        for d in range(len(model.encoder.encoders))
    })
    dynamic_axes.update({
        'out_cache_%d' % d: {
            0: 'out_cache_%d_batch' % d,
            1: 'out_cache_%d_length' % d
        }
        for d in range(len(model.encoder.encoders))
    })
    lm_inputs = (tgt, mask, cache)
    file_name = os.path.join(path, 'lm.onnx')
    # export encoder
    torch.onnx.export(
        onnxTransformerLM(model),
        lm_inputs,
        file_name,
        verbose=True,
        opset_version=11,
        input_names=lm_input_names,
        output_names=lm_output_names,
        dynamic_axes=dynamic_axes
    )


def export_lm(model, path, sos_token):
    if isinstance(model, SequentialRNNLM):
        export_seq_rnn(model, path, sos_token)
        
    elif isinstance(model, TransformerLM):
        export_transformer_lm(model, path, sos_token)


def create_config(model, path, decoder_odim):
    ret = {}
    # encoder
    ret.update(encoder=get_encoder_config(model.asr_model, path))
    ret.update(decoder=get_decoder_config(model.asr_model, path, decoder_odim))
    ret.update(transducer=get_transducer_config(model.asr_model, path))
    ret.update(ctc=dict(model_path=os.path.join(path, "ctc.onnx")))

    if "lm" in model.beam_search.nn_dict.keys():
        ret.update(lm=get_lm_config(model.beam_search.full_scorers['lm'], path))
    else:
        ret.update(lm=dict(use_lm=False))

    if "ngram" in list(model.beam_search.full_scorers.keys()) \
            + list(model.beam_search.part_scorers.keys()):
        ret.update(ngram=get_ngram_config(model))
    else:
        ret.update(ngram=dict(use_ngram=False))

    ret.update(weights=model.beam_search.weights)
    ret.update(beam_search=get_beam_config(model.beam_search, model.minlenratio, model.maxlenratio))
    ret.update(token=get_token_config(model.asr_model))
    ret.update(tokenizer=get_tokenizer_config(model.tokenizer))
    return ret


def quantize_model(model_from, model_to):
    ret = {}
    models = glob.glob(os.path.join(model_from, "*.onnx"))
    for m in models:
        basename = os.path.basename(m).split('.')[0]
        export_file = os.path.join(model_to, basename + '_qt.onnx')
        quantize_dynamic(
            m,
            export_file
        )
        ret[basename] = export_file
        os.remove(os.path.join(model_from, basename + '-opt.onnx'))
    return ret


class ModelExport:
    def __init__(self, cache_dir: Union[Path, str] = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "espnet_onnx"
        
        self.cache_dir = cache_dir

    def export(self, model: Speech2Text, model_name: str = None, quantize: bool = False):
        assert check_argument_types()
        if model_name is None:
            model_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        export_dir = self.cache_dir / model_name / 'full'
        export_dir.mkdir(parents=True, exist_ok=True)
        
        sos_token = model.asr_model.sos
        sample_feat = torch.randn((1, 100, 80))
        
        enc_out = export_encoder(model.asr_model.encoder, sample_feat, export_dir)
        if isinstance(enc_out, Tuple):
            enc_out = enc_out[0]

        decoder_odim = export_decoder(model.asr_model.decoder, enc_out, export_dir, sos_token)
        export_ctc(model.asr_model.ctc.ctc_lo, enc_out, export_dir)
        
        if 'lm' in model.beam_search.full_scorers.keys():
            export_lm(model.beam_search.full_scorers['lm'], export_dir, sos_token)

        config_name = self.cache_dir / model_name / 'config.json'
        model_config = create_config(model, export_dir, decoder_odim)
        
        if quantize:
            quantize_dir = self.cache_dir / model_name / 'quantize'
            quantize_dir.mkdir(exist_ok=True)
            
            qt_config = quantize_model(export_dir, quantize_dir)
            for m in qt_config.keys():
                model_config[m].update(quantized_model_path=qt_config[m])
        
        with open(config_name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(model_config))

    def export_from_pretrained(self, tag_name: str, quantize: bool = False):
        assert check_argument_types()
        model = Speech2Text.from_pretrained(tag_name)
        self.export(model, str_to_hash(tag_name), quantize)
