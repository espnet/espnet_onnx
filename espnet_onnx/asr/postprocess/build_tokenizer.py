from pathlib import Path
from typing import Iterable, Union

from typeguard import check_argument_types

from espnet_onnx.asr.postprocess.tokenizers.char_tokenizer import CharTokenizer
from espnet_onnx.asr.postprocess.tokenizers.phoneme_tokenizer import \
    PhonemeTokenizer
from espnet_onnx.asr.postprocess.tokenizers.sentencepiece_tokenizer import \
    SentencepiecesTokenizer
from espnet_onnx.asr.postprocess.tokenizers.word_tokenizer import WordTokenizer


def build_tokenizer(
    token_type: str,
    bpemodel: Union[Path, str, Iterable[str]] = None,
    non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
    remove_non_linguistic_symbols: bool = False,
    space_symbol: str = "<space>",
    delimiter: str = None,
    g2p_type: str = None,
):
    """A helper function to instantiate Tokenizer"""
    assert check_argument_types()
    if token_type == "bpe":
        if bpemodel is None:
            raise ValueError('bpemodel is required if token_type = "bpe"')

        if remove_non_linguistic_symbols:
            raise ValueError(
                "remove_non_linguistic_symbols is not implemented for token_type=bpe"
            )
        return SentencepiecesTokenizer(bpemodel)

    elif token_type == "word":
        if remove_non_linguistic_symbols and non_linguistic_symbols is not None:
            return WordTokenizer(
                delimiter=delimiter,
                non_linguistic_symbols=non_linguistic_symbols,
                remove_non_linguistic_symbols=True,
            )
        else:
            return WordTokenizer(delimiter=delimiter)

    elif token_type == "char":
        return CharTokenizer(
            non_linguistic_symbols=non_linguistic_symbols,
            space_symbol=space_symbol,
            remove_non_linguistic_symbols=remove_non_linguistic_symbols,
        )

    elif token_type == "phn":
        return PhonemeTokenizer(
            g2p_type=g2p_type,
            non_linguistic_symbols=non_linguistic_symbols,
            space_symbol=space_symbol,
            remove_non_linguistic_symbols=remove_non_linguistic_symbols,
        )

    else:
        raise ValueError(
            f"Token_mode must be one of bpe, word, char or phn: {token_type}"
        )
