from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np


class Hypothesis(NamedTuple):
    """Hypothesis data type."""

    yseq: np.ndarray
    score: Union[float, np.ndarray] = 0
    scores: Dict[str, Union[float, np.ndarray]] = dict()
    states: Dict[str, Any] = dict()

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
            scores={k: float(v) for k, v in self.scores.items()},
        )._asdict()


class BatchHypothesis(NamedTuple):
    """Batchfied/Vectorized hypothesis data type."""

    yseq: np.ndarray = np.ndarray([])  # (batch, maxlen)
    score: np.ndarray = np.ndarray([])  # (batch,)
    length: np.ndarray = np.ndarray([])  # (batch,)
    scores: Dict[str, np.ndarray] = dict()  # values: (batch,)
    states: Dict[str, Dict] = dict()

    def __len__(self) -> int:
        """Return a batch size."""
        return len(self.length)


@dataclass
class TransducerHypothesis:
    """Default hypothesis definition for Transducer search algorithms."""

    score: float
    yseq: List[int]
    dec_state: Union[
        Tuple[np.ndarray, Optional[np.ndarray]],
        List[Optional[np.ndarray]],
        np.ndarray,
    ]
    lm_state: Union[Dict[str, Any], List[Any]] = None


@dataclass
class ExtendedHypothesis(TransducerHypothesis):
    """Extended hypothesis definition for NSC beam search and mAES."""

    dec_out: List[np.ndarray] = None
    lm_scores: np.ndarray = None
