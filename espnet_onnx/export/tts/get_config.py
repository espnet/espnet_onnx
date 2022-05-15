import os

from espnet2.text.char_tokenizer import CharTokenizer
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet2.text.word_tokenizer import WordTokenizer


def get_token_config(model):
    return {
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
            "token_type": "phn",
            "g2p_type": model.g2p_type
        }


def get_preprocess_config(model, path):
    ret = {}
    if model.text_cleaner is not None:
        ret.update({'text_cleaner': {
                'cleaner_types': model.text_cleaner.cleaner_types[0]
        }})
    else:
        ret.update({'text_cleaner': None})
        
    ret.update({'tokenizer': get_tokenizer_config(model.tokenizer, path)})
    return ret
