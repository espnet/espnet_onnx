from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import onnxruntime

from espnet_onnx.utils.function import mask_fill
from espnet_onnx.utils.function import make_pad_mask
from espnet_onnx.asr.scorer.interface import BatchScorerInterface


def cross_entropy(pred, trg, reduction='mean', epsilon=1e-12):
    predictions = np.clip(pred, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ln = trg * np.log(pred + 1e-9)
    if reduction == 'none':
        return -ln
    elif reduction == 'mean':
        return -np.sum(ln) / N
    elif reduction == 'sum':
        return -np.sum(ln)


class LM(BatchScorerInterface):
    def __init__(self, config):
        """Initialize class.
        Args:
            n_vocab (int): The size of the vocabulary
            args (argparse.Namespace): configurations. see py:method:`add_arguments`
        """
        self.lm_session = onnxruntime.InferenceSession(config.model_path)
        self.output_names = ['y']\
            + [d.name for d in self.lm_session.get_outputs() if 'cache' in d.name]
        self.in_cache_names = [
            d.name for d in self.lm_session.get_inptus() if 'cache' in d.name]
        self.n_layers = config.n_enc_layers
        self.odims = config.cache_odim

    def _target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != 0
        m = np.expand_dims(subsequent_mask(ys_mask.shape[-1]), 0)
        return np.expand_dims(ys_mask, -2) & m

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
                torch.float32 scores for next token (n_vocab)
                and next state for ys
        """
        y = np.expand_dims(y, 0)

        input_dic = {
            'y': y,
            'mask': self._target_mask(y)
        }
        input_dic.update(
            {
                k: v for k, v in zip(self.in_cache_names, state)
            }
        )
        logp, *cache = self.lm_session.run(
            self.output_names,
            input_dic
        )

        return logp.squeeze(0), cache

    # batch beam search API (see BatchScorerInterface)
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
        # merge states
        n_batch = len(ys)
        if states[0] is None:
            batch_state = [
                np.zeros((1, 1, self.odims))
                for _ in range(self.n_layers)
            ]
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                np.stack([states[b][i] for b in range(n_batch)])
                for i in range(self.n_layers)
            ]

        input_dic = {
            'y': y,
            'mask': self._target_mask(y)
        }
        input_dic.update(
            {
                k: v for k, v in zip(self.in_cache_names, batch_state)
            }
        )
        logp, *cache = self.lm_session.run(
            self.output_names,
            input_dic
        )

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[cache[i][b]
                       for i in range(self.n_layers)] for b in range(n_batch)]
        return logp, state_list
