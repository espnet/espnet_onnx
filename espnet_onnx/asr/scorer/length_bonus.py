"""Length bonus module."""
from typing import Any, List, Tuple

import numpy as np

from espnet_onnx.asr.scorer.interface import BatchScorerInterface


class LengthBonus(BatchScorerInterface):
    """Length bonus in beam search."""

    def __init__(self, n_vocab: int):
        """Initialize class.
        Args:
            n_vocab (int): The number of tokens in vocabulary for beam search
        """
        self.n = n_vocab

    def score(self, y, state, x):
        """Score new token.
        Args:
            y (np.ndarray): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (np.ndarray): 2D encoder feature that generates ys.
        Returns:
            tuple[np.ndarray, Any]: Tuple of
                torch.float32 scores for next token (n_vocab)
                and None
        """
        return np.array([1.0], dtype=x.dtype).repeat(self.n), None

    def batch_score(
        self, ys: np.ndarray, states: List[Any], xs: np.ndarray
    ) -> Tuple[np.ndarray, List[Any]]:
        """Score new token batch.
        Args:
            ys (np.ndarray): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (np.ndarray):
                The encoder feature that generates ys (n_batch, xlen, n_feat).
        Returns:
            tuple[np.ndarray, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.
        """
        return (
            np.array([1.0], dtype=xs.dtype)
            .repeat(ys.shape[0] * self.n)
            .reshape(ys.shape[0] * self.n),
            None,
        )
