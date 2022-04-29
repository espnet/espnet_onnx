from typing import Union
from typing import Tuple
from typing import Any
from typing import List

import numpy as np
import onnxruntime
from scipy.special import log_softmax

from espnet_onnx.asr.scorer.interface import BatchScorerInterface
from espnet_onnx.utils.function import subsequent_mask


class TransformerLM(BatchScorerInterface):
    def __init__(
        self,
        config,
        providers: List[str],
        use_quantized=False
    ):
        if use_quantized:
            self.lm_session = onnxruntime.InferenceSession(
                config.quantized_model_path,
                providers=providers
            )
        else:
            self.lm_session = onnxruntime.InferenceSession(
                config.model_path,
                providers=providers
            )
        self.enc_output_names = ['y'] \
            + [d.name for d in self.lm_session.get_outputs() if 'cache' in d.name]
        self.enc_in_cache_names = [
            d.name for d in self.lm_session.get_inputs() if 'cache' in d.name]

        self.nlayers = config.nlayers
        self.odim = config.odim

    def _target_mask(self, ys_in_pad):
        # ys_in_pad : (B, D)
        ys_mask = ys_in_pad != 0
        m = subsequent_mask(ys_mask.shape[-1])[None, :]
        return ys_mask[:, None, :] * m

    def score(
        self, y: np.ndarray, state: Any, x: np.ndarray
    ) -> Tuple[np.ndarray, Any]:
        """Score new token.

        Args:
            y (np.ndarray): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (np.ndarray): encoder feature that generates ys.

        Returns:
            tuple[np.ndarray, Any]: Tuple of
                torch.float32 scores for next token (vocab_size)
                and next state for ys

        """
        y = y[None, :]

        input_dic = {'tgt': y, 'tgt_mask': self._target_mask(y)}
        input_dic.update({
            k: v for k, v in zip(self.enc_in_cache_names, state)
        })
        decoded, *new_state = self.lm_session.run(
            self.enc_output_names,
            input_dic
        )

        logp = log_softmax(decoded, axis=-1).squeeze(0)
        return logp, new_state

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
                batchfied scores for next token with shape of `(n_batch, vocab_size)`
                and next state list for ys.

        """
        # merge states
        ys = ys.astype(np.int64)
        n_batch = len(ys)
        if states[0] is None:
            batch_state = [
                np.zeros((1, 1, self.odim), dtype=np.float32)
                for _ in range(self.nlayers)
            ]
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                np.concatenate([states[b][i][None, :] for b in range(n_batch)])
                for i in range(self.nlayers)
            ]

        input_dic = {'tgt': ys, 'tgt_mask': self._target_mask(ys)}
        input_dic.update({
            k: v for k, v in zip(self.enc_in_cache_names, batch_state)
        })
        decoded, *new_state = self.lm_session.run(
            self.enc_output_names,
            input_dic
        )
        logp = log_softmax(decoded, axis=-1)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[new_state[i][b]
                       for i in range(self.nlayers)] for b in range(n_batch)]
        return logp, state_list
