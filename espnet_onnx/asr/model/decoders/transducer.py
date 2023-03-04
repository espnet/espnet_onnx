from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime

from espnet_onnx.asr.beam_search.hyps import ExtendedHypothesis, Hypothesis
from espnet_onnx.asr.scorer.interface import BatchScorerInterface
from espnet_onnx.utils.config import Config


class TransducerDecoder(BatchScorerInterface):
    def __init__(
        self, config: Config, providers: List[str], use_quantized: bool = False
    ):
        """Onnx support for espnet2.asr.decoder.transformer_decoder

        Args:
            config (Config):
            use_quantized (bool): Flag to use quantized model
        """
        if use_quantized:
            self.decoder = onnxruntime.InferenceSession(
                config.quantized_model_path, providers=providers
            )
        else:
            self.decoder = onnxruntime.InferenceSession(
                config.model_path, providers=providers
            )
        self.n_layers = config.n_layers
        self.odim = config.odim
        self.dtype = config.dtype
        self.output_names = (
            ["sequence"]
            + sorted(
                [d.name for d in self.decoder.get_outputs() if "h_cache" in d.name]
            )
            + sorted(
                [d.name for d in self.decoder.get_outputs() if "c_cache" in d.name]
            )
        )

    def score(
        self, hyp: Hypothesis, cache: Dict[str, Any]
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]], np.ndarray]:
        """One-step forward hypothesis.

        Args:
            hyp: Hypothesis.
            cache: Pairs of (dec_out, state) for each label sequence. (key)

        Returns:
            dec_out: Decoder output sequence. (1, D_dec)
            new_state: Decoder hidden states. ((N, 1, D_dec), (N, 1, D_dec))
            label: Label ID for LM. (1,)

        """
        label = np.full((1, 1), hyp.yseq[-1], dtype=np.int64)
        str_labels = "_".join(list(map(str, hyp.yseq)))

        if str_labels in cache:
            dec_out, dec_state = cache[str_labels]
        else:
            input_dict = self.get_input_dict(label, hyp.dec_state)
            dec_out, *next_states = self.decoder.run(self.output_names, input_dict)
            dec_state = self.split(next_states)
            cache[str_labels] = (dec_out, dec_state)

        return dec_out[0][0], dec_state, label[0]

    def batch_score(
        self,
        hyps: Union[List[Hypothesis], List[ExtendedHypothesis]],
        dec_states: Tuple[np.ndarray, Optional[np.ndarray]],
        cache: Dict[str, Any],
        use_lm: bool,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """One-step forward hypotheses.

        Args:
            hyps: Hypotheses.
            states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))
            cache: Pairs of (dec_out, dec_states) for each label sequences. (keys)
            use_lm: Whether to compute label ID sequences for LM.

        Returns:
            dec_out: Decoder output sequences. (B, D_dec)
            dec_states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))
            lm_labels: Label ID sequences for LM. (B,)

        """
        final_batch = len(hyps)

        process = []
        done = [None] * final_batch

        for i, hyp in enumerate(hyps):
            str_labels = "_".join(list(map(str, hyp.yseq)))

            if str_labels in cache:
                done[i] = cache[str_labels]
            else:
                process.append((str_labels, hyp.yseq[-1], hyp.dec_state))

        if process:
            labels = np.array([[p[1]] for p in process], dtype=np.int64)
            p_dec_states = self.create_batch_states(
                self.init_state(labels.shape[0]), [p[2] for p in process]
            )
            input_dict = self.get_input_dict(labels, p_dec_states)
            dec_out, *next_states = self.decoder.run(self.output_names, input_dict)
            new_states = self.split(next_states)

        j = 0
        for i in range(final_batch):
            if done[i] is None:
                state = self.select_state(new_states, j)
                done[i] = (dec_out[j], state)
                cache[process[j][0]] = (dec_out[j], state)
                j += 1

        dec_out = np.concatenate([d[0] for d in done], axis=0)
        dec_states = self.create_batch_states(dec_states, [d[1] for d in done])
        if use_lm:
            lm_labels = np.array([h.yseq[-1] for h in hyps], dtype=np.int64).view(
                final_batch, 1
            )

            return dec_out, dec_states, lm_labels

        return dec_out, dec_states, None

    def get_input_dict(self, labels, states):
        ret = {"labels": labels, "h_cache": states[0], "c_cache": states[1]}
        return ret

    def select_state(
        self, states: Tuple[np.ndarray, Optional[np.ndarray]], idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get specified ID state from decoder hidden states.

        Args:
            states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))
            idx: State ID to extract.

        Returns:
            : Decoder hidden state for given ID.
              ((N, 1, D_dec), (N, 1, D_dec))

        """
        return (
            states[0][:, idx : idx + 1, :],
            states[1][:, idx : idx + 1, :] if self.dtype == "lstm" else None,
        )

    def init_state(
        self, batch_size: int = 1
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Initialize decoder states.

        Args:
            batch_size: Batch size.

        Returns:
            : Initial decoder hidden states. ((N, B, D_dec), (N, B, D_dec))

        """
        h_n = np.zeros((self.n_layers, batch_size, self.odim), dtype=np.float32)

        if self.dtype == "lstm":
            c_n = np.zeros((self.n_layers, batch_size, self.odim), dtype=np.float32)
            return (h_n, c_n)

        return (h_n, None)

    def create_batch_states(
        self,
        states: Tuple[np.ndarray, Optional[np.ndarray]],
        new_states: List[Tuple[np.ndarray, Optional[np.ndarray]]],
        check_list: Optional[List] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create decoder hidden states.

        Args:
            states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))
            new_states: Decoder hidden states. [N x ((1, D_dec), (1, D_dec))]

        Returns:
            states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))

        """
        return (
            np.concatenate([s[0] for s in new_states], axis=1),
            np.concatenate([s[1] for s in new_states], axis=1)
            if self.dtype == "lstm"
            else None,
        )

    def split(self, states):
        if self.dtype == "lstm":
            h_next = states[0]
            c_next = states[1]
            return (h_next, c_next)
        else:
            return (states, None)
