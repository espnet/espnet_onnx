import warnings
from typing import Any, List, Tuple

import numpy as np


class ScorerInterface:
    """Scorer interface for beam search.
    The scorer performs scoring of the all tokens in vocabulary.
    Examples:
        * Search heuristics
            * :class:`espnet.nets.scorers.length_bonus.LengthBonus`
        * Decoder networks of the sequence-to-sequence models
            * :class:`espnet.nets.pytorch_backend.nets.transformer.decoder.Decoder`
            * :class:`espnet.nets.pytorch_backend.nets.rnn.decoders.Decoder`
        * Neural language models
            * :class:`espnet.nets.pytorch_backend.lm.transformer.TransformerLM`
            * :class:`espnet.nets.pytorch_backend.lm.default.DefaultRNNLM`
            * :class:`espnet.nets.pytorch_backend.lm.seq_rnn.SequentialRNNLM`
    """

    def init_state(self, x: np.ndarray) -> Any:
        """Get an initial state for decoding (optional).
        Args:
            x (np.ndarray): The encoded feature tensor
        Returns: initial state
        """
        return None

    def select_state(self, state: Any, i: int, new_id: int = None) -> Any:
        """Select state with relative ids in the main beam search.
        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label index to select a state if necessary
        Returns:
            state: pruned state
        """
        return None if state is None else state[i]

    def score(self, y: np.ndarray, state: Any, x: np.ndarray) -> Tuple[np.ndarray, Any]:
        """Score new token (required).
        Args:
            y (np.ndarray): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (np.ndarray): The encoder feature that generates ys.
        Returns:
            tuple[np.ndarray, Any]: Tuple of
                scores for next token that has a shape of `(n_vocab)`
                and next state for ys
        """
        raise NotImplementedError

    def final_score(self, state: Any) -> float:
        """Score eos (optional).
        Args:
            state: Scorer state for prefix tokens
        Returns:
            float: final score
        """
        return 0.0


class BatchScorerInterface(ScorerInterface):
    """Batch scorer interface."""

    def batch_init_state(self, x: np.ndarray) -> Any:
        """Get an initial state for decoding (optional).
        Args:
            x (np.ndarray): The encoded feature tensor
        Returns: initial state
        """
        return self.init_state(x)

    def batch_score(
        self, ys: np.ndarray, states: List[Any], xs: np.ndarray
    ) -> Tuple[np.ndarray, List[Any]]:
        """Score new token batch (required).
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
        warnings.warn(
            "{} batch score is implemented through for loop not parallelized".format(
                self.__class__.__name__
            )
        )
        scores = list()
        outstates = list()
        for i, (y, state, x) in enumerate(zip(ys, states, xs)):
            score, outstate = self.score(y, state, x)
            outstates.append(outstate)
            scores.append(score)
        scores = np.concatenate(scores, axis=0).reshape(ys.shape[0], -1)
        return scores, outstates


class PartialScorerInterface(ScorerInterface):
    """Partial scorer interface for beam search.
    The partial scorer performs scoring when non-partial scorer finished scoring,
    and receives pre-pruned next tokens to score because it is too heavy to score
    all the tokens.
    Examples:
         * Prefix search for connectionist-temporal-classification models
             * :class:`espnet.nets.scorers.ctc.CTCPrefixScorer`
    """

    def score_partial(
        self, y: np.ndarray, next_tokens: np.ndarray, state: Any, x: np.ndarray
    ) -> Tuple[np.ndarray, Any]:
        """Score new token (required).
        Args:
            y (np.ndarray): 1D prefix token
            next_tokens (np.ndarray): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (np.ndarray): The encoder feature that generates ys
        Returns:
            tuple[np.ndarray, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys
        """
        raise NotImplementedError


class BatchPartialScorerInterface(BatchScorerInterface, PartialScorerInterface):
    """Batch partial scorer interface for beam search."""

    def batch_score_partial(
        self,
        ys: np.ndarray,
        next_tokens: np.ndarray,
        states: List[Any],
        xs: np.ndarray,
    ) -> Tuple[np.ndarray, Any]:
        """Score new token (required).
        Args:
            ys (np.ndarray): torch.int64 prefix tokens (n_batch, ylen).
            next_tokens (np.ndarray): torch.int64 tokens to score (n_batch, n_token).
            states (List[Any]): Scorer states for prefix tokens.
            xs (np.ndarray):
                The encoder feature that generates ys (n_batch, xlen, n_feat).
        Returns:
            tuple[np.ndarray, Any]:
                Tuple of a score tensor for ys that has a shape `(n_batch, n_vocab)`
                and next states for ys
        """
        raise NotImplementedError
