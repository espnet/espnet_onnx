from typing import Iterable
from typing import List, Union
from pathlib import Path
import sentencepiece as spm

class SentencepiecesTokenizer():
    def __init__(self, model: Union[Path, str]):
        self.model = str(model)
        # NOTE(kamo):
        # Don't build SentencePieceProcessor in __init__()
        # because it's not picklable and it may cause following error,
        # "TypeError: can't pickle SwigPyObject objects",
        # when giving it as argument of "multiprocessing.Process()".
        self.sp = None

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def _build_sentence_piece_processor(self):
        # Build SentencePieceProcessor lazily.
        if self.sp is None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.model)

    def text2tokens(self, line: str) -> List[str]:
        self._build_sentence_piece_processor()
        return self.sp.EncodeAsPieces(line)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        self._build_sentence_piece_processor()
        return self.sp.DecodePieces(list(tokens))


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
    return SentencepiecesTokenizer(bpemodel)