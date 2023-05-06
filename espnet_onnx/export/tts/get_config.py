import os

from espnet2.text.char_tokenizer import CharTokenizer
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet2.text.word_tokenizer import WordTokenizer


def get_token_config(model):
    return {"list": model.token_list}


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
        return {"token_type": "phn", "g2p_type": model.g2p_type}


def get_preprocess_config(model, path):
    ret = {}
    if hasattr(model, "text_cleaner") and model.text_cleaner is not None:
        ret.update(
            {
                "text_cleaner": {
                    "cleaner_types": [ct for ct in model.text_cleaner.cleaner_types]
                }
            }
        )
    else:
        ret.update({"text_cleaner": None})

    if model is not None:
        ret.update({"tokenizer": get_tokenizer_config(model.tokenizer, path)})
    else:
        ret.update({"tokenizer": {}})
    return ret


def get_vocoder_config(model):
    if model is None:
        return {"vocoder_type": "not_used"}
    else:
        ret = {
            "vocoder_type": "Spectrogram2Waveform",
        }
        if hasattr(model, "params"):
            ret.update(model.params)
    return ret


def get_normalize_config(model, path):
    ret = {"use_normalize": model is not None}
    if ret["use_normalize"]:
        ret.update(
            {
                "type": "gmvn",
                "norm_means": model.norm_means,
                "norm_vars": model.norm_vars,
                "eps": model.eps,
                "stats_file": str(path.parent / "feats_stats.npz"),
            }
        )
    return ret
