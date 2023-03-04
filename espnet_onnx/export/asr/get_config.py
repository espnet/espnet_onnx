import os

from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.text.char_tokenizer import CharTokenizer
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet2.text.word_tokenizer import WordTokenizer


def get_ngram_config(model):
    return {"use_ngram": True}


def get_weights_transducer(model):
    return {"lm": model.lm_weight}


def get_beam_config(model, minlenratio, maxlenratio):
    return {
        "beam_size": model.beam_size,
        "pre_beam_ratio": model.pre_beam_size / model.beam_size,
        "pre_beam_score_key": model.pre_beam_score_key,
        "maxlenratio": maxlenratio,
        "minlenratio": minlenratio,
    }


def get_trans_beam_config(model):
    # check search algorithm
    search_type = ""
    search_args = {}
    if model.beam_size <= 1:
        search_type = "greedy"
    elif hasattr(model, "max_sym_exp"):
        search_type = "tsd"
        search_args["max_sym_exp"] = model.max_sym_exp
    elif hasattr(model, "u_max"):
        search_type = "alsd"
        search_args["u_max"] = model.u_max
    elif hasattr(model, "nstep"):
        search_args["prefix_alpha"] = model.prefix_alpha
        if hasattr(model, "expansion_gamma"):
            search_type = "maes"
            search_args["nstep"] = max(2, model.nstep)
            search_args["expansion_gamma"] = model.expansion_gamma
            search_args["expansion_beta"] = model.expansion_beta
        else:
            search_type = "nsc"
            search_args["nstep"] = model.nstep
    else:
        search_type = "default"

    return {
        "beam_size": model.beam_size,
        "search_type": search_type,
        "search_args": search_args,
        "score_norm": model.score_norm,
    }


def get_token_config(model):
    return {
        "sos": model.sos,
        "eos": model.eos,
        "blank": model.blank_id,
        "list": model.token_list,
    }


def get_tokenizer_config(model, path):
    if model is None:
        return {}
    elif isinstance(model, SentencepiecesTokenizer):
        model_name = os.path.basename(model.model)
        return {"token_type": "bpe", "bpemodel": str(path.parent / model_name)}
    elif isinstance(model, WordTokenizer):
        return {"token_type": "word"}
    elif isinstance(model, CharTokenizer):
        return {"token_type": "char"}
    elif isinstance(model, PhonemeTokenizer):
        return {"token_type": "phn"}


def get_frontend_config(asr_frontend_model, frontend=None, **kwargs):
    # currently only default config is supported.
    if isinstance(asr_frontend_model, S3prlFrontend):
        frontend_config = frontend.get_model_config(**kwargs)
    elif isinstance(asr_frontend_model, DefaultFrontend):
        frontend_config = get_default_frontend(asr_frontend_model)
    else:
        raise ValueError("Currently only s3prl is supported.")

    return frontend_config


def get_default_frontend(frontend, **kwargs):
    return {
        "frontend_type": "default",
        "stft": get_stft_config(frontend.stft, **kwargs),
        "logmel": get_logmel_config(frontend.logmel, **kwargs),
    }


def get_enh_config(frontend):
    if frontend is None:
        return {}
    else:
        return {
            "use_wpe": frontend.use_wpe,
            "use_dnn_mask_for_wpe": frontend.use_dnn_mask_for_wpe,
            "use_beamformer": frontend.use_beamformer,
        }


def get_stft_config(stft, stft_center: bool = True):
    return {
        "n_fft": stft.n_fft,
        "win_length": stft.win_length,
        "hop_length": stft.hop_length,
        "window": stft.window,
        "center": stft_center,  # This could be False in streaming model.
        "onesided": stft.onesided,
        "normalized": stft.normalized,
    }


def get_logmel_config(logmel):
    logmel_config = logmel.mel_options
    logmel_config.update(log_base=logmel.log_base)
    return logmel_config


def get_norm_config(normalize, path):
    if isinstance(normalize, GlobalMVN):
        return {
            "type": "gmvn",
            "norm_means": normalize.norm_means,
            "norm_vars": normalize.norm_vars,
            "eps": normalize.eps,
            "stats_file": str(path.parent / "feats_stats.npz"),
        }
    elif isinstance(normalize, UtteranceMVN):
        return {
            "type": "utterance_mvn",
            "norm_means": normalize.norm_means,
            "norm_vars": normalize.norm_vars,
            "eps": normalize.eps,
        }
