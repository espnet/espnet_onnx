import os

from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.transducer.transducer_decoder import TransducerDecoder
from espnet2.asr.transducer.beam_search_transducer import BeamSearchTransducer
from espnet2.lm.seq_rnn_lm import SequentialRNNLM
from espnet2.lm.transformer_lm import TransformerLM


def get_frontend_config(model):
    # currently only default config is supported.
    assert isinstance(
        model, DefaultFrontend), 'Currently only DefaultFrontend is supported.'
    
    stft_config = dict(
        n_fft=model.stft.n_fft,
        win_length=model.stft.win_length,
        hop_length=model.stft.hop_length,
        window=model.stft.window,
        center=model.stft.center,
        onesided=model.stft.onesided,
        normalized=model.stft.normalized,
    )
    logmel_config = model.logmel.mel_options
    logmel_config.update(log_base=model.logmel.log_base)
    return {
        "stft": stft_config,
        "logmel": logmel_config
    }


def get_gmvn_config(model):
    return {
        "norm_means": model.norm_means,
        "norm_vars": model.norm_vars,
        "eps": model.eps,
        "stats_file": str(model.stats_file)
    }


def get_encoder_config(model, path):
    ret = {}
    ret.update(
        model_path=os.path.join(path, 'encoder.onnx'),
        frontend=get_frontend_config(model.frontend),
        do_normalize=model.normalize is not None,
        do_preencoder=model.preencoder is not None,
        do_postencoder=model.postencoder is not None
    )
    if ret['do_normalize']:
        ret.update(gmvn=get_gmvn_config(model.normalize))
    if ret['do_preencoder']:
        ret.update(preencoder=get_preenc_config(model.preencoder))
    if ret['do_postencoder']:
        ret.update(postencoder=get_postenc_config(model.postencoder))
    return ret


def get_decoder_config(model, path, decoder_odim):
    file_name = os.path.join(path, 'decoder.onnx')
    if isinstance(model.decoder, TransformerDecoder):
        return {
            "model_path": file_name,
            "n_layers": len(model.decoder.decoders),
            "odim": decoder_odim
        }
    elif isinstance(model.decoder, TransducerDecoder):
        return {
            "model_path": file_name,
            "rnn_type": model.decoder.dtype,
            "n_layers": model.decoder.dlayers,
            "hidden_size": model.decoder.dunits,
            "odim": decoder_odim
        }


# def get_search_type(func):
#     if isinstance(func, )


def get_transducer_config(model, path):
    if isinstance(model.decoder, TransformerDecoder):
        return {"use_transducer_decoder": False}
    else:
        return {
            "use_transducer_decoder": True,
            "joint_network": {
                "model_path": os.path.join(path, "joint_network.onnx")
            },
            # NOTE: Currently only default search function is supported.
            "search_type": "default",
            "score_norm": model.beam_search.score_norm,
            "nbest": model.beam_search.nbest
        }


def get_lm_config(model, path):
    if isinstance(model, SequentialRNNLM):
        return {
            "use_lm": True,
            "model_path": os.path.join(path, "rnn_lm.onnx"),
            "lm_type": "SequentialRNNLM",
            "rnn_type": model.rnn_type,
            "nhid": model.nhid,
            "nlayers": model.nlayers
        }
    else:
        raise Error('TransformerLm is not supported.')


def get_ngram_config(model):
    return {
        "use_ngram": True
    }


def get_beam_config(model):
    return {
        "beam_size": model.beam_size,
        "pre_beam_ratio": model.pre_beam_size / model.beam_size,
        "pre_beam_score_key": model.pre_beam_score_key
    }


def get_token_config(model):
    return {
        "sos": model.sos,
        "eos": model.eos,
        "blank": model.blank_id,
        "list": model.token_list
    }
