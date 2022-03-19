import os

from espnet2.text.char_tokenizer import CharTokenizer
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet2.text.word_tokenizer import WordTokenizer


def get_ngram_config(model):
    return {
        "use_ngram": True
    }


def get_beam_config(model, minlenratio, maxlenratio):
    return {
        "beam_size": model.beam_size,
        "pre_beam_ratio": model.pre_beam_size / model.beam_size,
        "pre_beam_score_key": model.pre_beam_score_key,
        "maxlenratio": maxlenratio,
        "minlenratio": minlenratio
    }


def get_token_config(model):
    return {
        "sos": model.sos,
        "eos": model.eos,
        "blank": model.blank_id,
        "list": model.token_list
    }


def get_tokenizer_config(model, path):
    if model is None:
        return {}
    elif isinstance(model, SentencepiecesTokenizer):
        model_name = os.path.basename(model.model)
        return {
            "token_type": "bpe",
            "bpemodel": str(path.parent / model_name)
        }
    elif isinstance(model, WordTokenizer):
        return {
            "token_type": "word"
        }
    elif isinstance(model, CharTokenizer):
        return {
            "token_type": "char"
        }
    elif isinstance(model, PhonemeTokenizer):
        return {
            "token_type": "phn"
        }
