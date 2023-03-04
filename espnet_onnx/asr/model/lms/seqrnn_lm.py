from typing import List, Tuple, Union

import numpy as np
import onnxruntime
from scipy.special import log_softmax

from espnet_onnx.asr.scorer.interface import BatchScorerInterface
from espnet_onnx.utils.config import Config


class SequentialRNNLM(BatchScorerInterface):
    """Sequential RNNLM.
    See also:
        https://github.com/pytorch/examples/blob/4581968193699de14b56527296262dd76ab43557/word_language_model/model.py
    """

    def __init__(
        self, config: Config, providers: List[str], use_quantized: bool = False
    ):
        if use_quantized:
            self.lm_session = onnxruntime.InferenceSession(
                config.quantized_model_path, providers=providers
            )
        else:
            self.lm_session = onnxruntime.InferenceSession(
                config.model_path, providers=providers
            )
        self.enc_output_names = ["y"] + [
            d.name for d in self.lm_session.get_outputs() if "hidden" in d.name
        ]
        self.enc_in_cache_names = [
            d.name for d in self.lm_session.get_inputs() if "hidden" in d.name
        ]

        self.rnn_type = config.rnn_type
        self.nhid = config.nhid
        self.nlayers = config.nlayers

    def zero_state(self):
        """Initialize LM state filled with zero values."""
        if self.rnn_type == "LSTM":
            h = np.zeros((self.nlayers, self.nhid), dtype=np.float32)
            c = np.zeros((self.nlayers, self.nhid), dtype=np.float32)
            state = h, c
        else:
            state = np.zeros((self.nlayers, self.nhid), dtype=np.float32)
        return state

    def create_cache(self):
        c = np.zeros((self.nlayers, 1, self.nhid), dtype=np.float32)
        states = (c,)
        if self.rnn_type == "LSTM":
            states = (c, c)
        return states

    def score(
        self,
        y: np.ndarray,
        state: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        x: np.ndarray,
    ) -> Tuple[np.ndarray, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """Score new token.
        Args:
            y: 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x: 2D encoder feature that generates ys.
        Returns:
            Tuple of
                torch.float32 scores for next token (n_vocab)
                and next state for ys
        """
        input_dic = {"x": y[-1].reshape(1, 1)}

        if state is None:
            state = self.create_cache()

        input_dic.update({k: v for k, v in zip(self.enc_in_cache_names, state)})
        decoded, *new_state = self.lm_session.run(self.enc_output_names, input_dic)
        logp = log_softmax(decoded, axis=-1).reshape(-1)
        return logp, new_state

    def batch_score(
        self, ys: np.ndarray, states: np.ndarray, xs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        if states[0] is None:
            states = self.create_cache()

        elif self.rnn_type == "LSTM":
            # states: Batch x 2 x (Nlayers, Dim) -> 2 x (Nlayers, Batch, Dim)
            h = np.concatenate([h[:, None] for h, c in states], axis=1)
            c = np.concatenate([c[:, None] for h, c in states], axis=1)
            states = h, c

        else:
            # states: Batch x (Nlayers, Dim) -> (Nlayers, Batch, Dim)
            states = np.concatenate([states[:, None] for s in states], axis=1)

        input_dic = {"x": ys[:, -1:].astype(np.int64)}
        input_dic.update({k: v for k, v in zip(self.enc_in_cache_names, states)})
        decoded, *new_states = self.lm_session.run(self.enc_output_names, input_dic)
        decoded = decoded.squeeze(1)
        logp = log_softmax(decoded, axis=-1)
        # state: Change to batch first
        if self.rnn_type == "LSTM":
            # h, c: (Nlayers, Batch, Dim)
            h, c = new_states
            # states: Batch x 2 x (Nlayers, Dim)
            new_states = [(h[:, i], c[:, i]) for i in range(h.shape[1])]
        else:
            # states: (Nlayers, Batch, Dim) -> Batch x (Nlayers, Dim)
            new_states = [new_states[:, i] for i in range(new_states.shape[1])]

        return logp, new_states
