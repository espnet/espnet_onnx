from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F

import numpy as np
import onnxruntime

from espnet_onnx.utils.function import mask_fill, make_pad_mask


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


class LM:
    def __init__(self, lm_config, token_config):
        self.lm_session = onnxruntime.InferenceSession(lm_config.model_path)
        self.sos = token_config.sos
        self.eos = token_config.eos

        # ignore_id may be assumed as 0, shared with CTC-blank symbol for ASR.
        self.ignore_id = token_config.blank

    def nll(
        self,
        text: np.ndarray,
        text_lengths: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute negative log likelihood(nll)
        Normally, this function is called in batchify_nll.
        Args:
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        batch_size = text.shape[0]
        # For data parallel
        text = text[:, : text_lengths.max()]

        # 1. Create a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # text: (Batch, Length) -> x, y: (Batch, Length + 1)
        x = np.pad(text, [[0, 0], [1, 0]], "constant", constant_values=self.eos)
        t = np.pad(text, [[0, 0], [1, 0]], "constant", constant_values=self.ignore_id)
        for i, l in enumerate(text_lengths):
            t[i, l] = self.sos
        x_lengths = text_lengths + 1
        # 2. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        y = self.lm_session.run(['lm_out'], {'x': x})

        # 3. Calc negative log likelihood
        # nll: (BxL,)
        t = t.reshape(-1)
        target = np.zeros((len(t), y.shape[-1]))
        for i in range(len(t)):
            target[i, t[i]] = 1
        nll = cross_entropy(y.reshape(-1, y.shape[-1]), target, reduction="none")
        
        # nll: (BxL,) -> (BxL,)
        nll = mask_fill(nll, make_pad_mask(x_lengths).reshape(-1), 0.0)
        # nll: (BxL,) -> (B, L)
        nll = nll.reshape(batch_size, -1)
        return nll, x_lengths

    def batchify_nll(
        self, text: np.ndarray, text_lengths: np.ndarray, batch_size: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute negative log likelihood(nll) from transformer language model
        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            text: (Batch, Length)
            text_lengths: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
        """
        total_num = text.shape[0]
        if total_num <= batch_size:
            nll, x_lengths = self.nll(text, text_lengths)
        else:
            nlls = []
            x_lengths = []

            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_text = text[start_idx:end_idx, :]
                batch_text_lengths = text_lengths[start_idx:end_idx]
                # batch_nll: [B * T]
                batch_nll, batch_x_lengths = self.nll(
                    batch_text, batch_text_lengths
                )
                nlls.append(batch_nll)
                x_lengths.append(batch_x_lengths)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = np.concatenate(nlls)
            x_lengths = np.concatenate(x_lengths)
        assert nll.size(0) == total_num
        assert x_lengths.size(0) == total_num
        return nll, x_lengths

    def forward(
        self, text: np.ndarray, text_lengths: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        nll, y_lengths = self.nll(text, text_lengths)
        ntokens = y_lengths.sum()
        loss = nll.sum() / ntokens
        stats = dict(loss=loss)
        return loss, stats, ntokens

    def collect_feats(
        self, text: np.ndarray, text_lengths: np.ndarray
    ) -> Dict[str, np.ndarray]:
        return {}