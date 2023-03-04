from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime

from espnet_onnx.asr.scorer.interface import BatchScorerInterface
from espnet_onnx.utils.config import Config


class XformerDecoder(BatchScorerInterface):
    def __init__(
        self,
        config: Config,
        providers: List[str],
        use_quantized: bool = False,
    ):
        """Onnx support for espnet2.asr.decoder.transformer_decoder

        Args:
            config (Config):
            use_quantized (bool): Flag to use quantized model
        """
        if use_quantized:
            self.decoder = onnxruntime.InferenceSession(
                config.quantized_model_path,
                providers=providers,
            )
        else:
            self.decoder = onnxruntime.InferenceSession(
                config.model_path,
                providers=providers,
            )
        self.config = config
        self.n_layers = config.n_layers
        self.odim = config.odim
        self.in_caches = [
            d.name for d in self.decoder.get_inputs() if "cache" in d.name
        ]
        self.out_caches = [
            d.name for d in self.decoder.get_outputs() if "cache" in d.name
        ]

    def batch_score(
        self, ys: np.ndarray, states: List[Any], xs: np.ndarray
    ) -> Tuple[np.ndarray, List[Any]]:
        """Score new token batch.
        Args:
            ys (np.ndarray): np.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (np.ndarray):
                The encoder feature that generates ys (n_batch, xlen, n_feat).
        Returns:
            tuple[np.ndarray, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.
        """
        # merge states
        if len(ys.shape) == 1:
            ys = ys[None, :]

        n_batch = len(ys)
        if states[0] is None:
            batch_state = [
                np.zeros((1, 1, self.odim), dtype=np.float32)
                for _ in range(self.n_layers)
            ]
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                np.concatenate([states[b][i][None, :] for b in range(n_batch)])
                for i in range(self.n_layers)
            ]

        # batch decoding
        input_dict = self.get_input_dict(ys, xs, batch_state)

        logp, *states = self.decoder.run(["y"] + self.out_caches, input_dict)

        if type(self.n_layers) == 1:
            states = [states]

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [
            [states[i][b] for i in range(self.n_layers)] for b in range(n_batch)
        ]

        return logp, state_list

    def get_input_dict(self, ys, xs, state):
        in_names = [d.name for d in self.decoder.get_inputs() if "cache" not in d.name]
        ret = {}
        if "tgt" in in_names:
            ret.update(tgt=ys.astype(np.int64))
        if "memory" in in_names:
            ret.update(memory=xs)
        ret.update({k: v for (k, v) in zip(self.in_caches, state)})
        return ret
