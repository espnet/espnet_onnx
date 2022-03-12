from typing import List
from typing import Any
from typing import Tuple
from typing import Optional
from typing import Dict
from typing import Union

import numpy as np
import onnxruntime

from espnet_onnx.asr.scorer.interface import BatchScorerInterface
from espnet_onnx.asr.beam_search.hyps import TransducerHypothesis
from espnet_onnx.utils.function import subsequent_mask
from espnet_onnx.utils.config import Config


def get_decoder(config, token_config: Config, td_config: Config, use_quantized: bool = False):
    if td_config.use_transducer_decoder:
        # return TransducerDecoder(config, token_config)
        raise ValueError('Transducer is currently not supported.')
    else:
        return OnnxDecoderModel(config, use_quantized)


class OnnxDecoderModel(BatchScorerInterface):
    def __init__(
        self,
        config: Config,
        use_quantized: bool = False
    ):
        """Onnx support for espnet2.asr.decoder.transformer_decoder

        Args:
            config (Config):
            use_quantized (bool): Flag to use quantized model
        """
        if use_quantized:
            self.decoder = onnxruntime.InferenceSession(
                config.quantized_model_path)
        else:
            self.decoder = onnxruntime.InferenceSession(config.model_path)
        self.n_layers = config.n_layers
        self.odim = config.odim
        self.in_caches = [d.name for d in self.decoder.get_inputs()
                          if 'cache' in d.name]
        self.out_caches = [d.name for d in self.decoder.get_outputs()
                           if 'cache' in d.name]

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
        ys_mask = subsequent_mask(ys.shape[-1])[None, :]

        input_dict = {
            'tgt': ys.astype(np.int64),
            'tgt_mask': ys_mask,
            'memory': xs
        }
        input_dict.update(
            {k: v for (k, v) in zip(self.in_caches, batch_state)})

        logp, *states = self.decoder.run(
            ['y'] + self.out_caches,
            input_dict
        )

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b]
                       for i in range(self.n_layers)] for b in range(n_batch)]
        return logp, state_list


# class TransducerDecoder:
#     """(RNN-)Transducer decoder module.
#     Args:
#         vocab_size: Output dimension.
#         layers_type: (RNN-)Decoder layers type.
#         num_layers: Number of decoder layers.
#         hidden_size: Number of decoder units per layer.
#         dropout: Dropout rate for decoder layers.
#         dropout_embed: Dropout rate for embedding layer.
#         embed_pad: Embed/Blank symbol ID.
#     """

#     def __init__(
#         self,
#         config: Config,
#         token_config: bool = False
#     ):
#         self.decoder = onnxruntime.InferenceSession(config.model_path)
#         self.cache_in_name = [
#             d.name for d in self.decoder.get_inputs() if 'cache' in d.name]
#         self.cache_out_name = [
#             d.name for d in self.decoder.get_outputs() if 'cache' in d.name]

#         self.dlayers = config.n_layers
#         self.dunits = config.hidden_size
#         self.dtype = config.rnn_type
#         self.odim = len(token_config.list)

#         self.ignore_id = -1
#         self.blank_id = token_config.blank

#     def init_state(
#         self, batch_size: int
#     ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
#         """Initialize decoder states.
#         Args:
#             batch_size: Batch size.
#         Returns:
#             : Initial decoder hidden states. ((N, B, D_dec), (N, B, D_dec))
#         """
#         h_n = np.zeros((self.dlayers, batch_size, self.dunits))

#         if self.dtype == "lstm":
#             c_n = np.zeros((self.dlayers, batch_size, self.dunits))
#             return (h_n, c_n)

#         return (h_n, None)

#     def score(
#         self, hyp: TransducerHypothesis, cache: Dict[str, Any]
#     ) -> Tuple[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]], np.ndarray]:
#         """One-step forward hypothesis.
#         Args:
#             hyp: TransducerHypothesis.
#             cache: Pairs of (dec_out, state) for each label sequence. (key)
#         Returns:
#             dec_out: Decoder output sequence. (1, D_dec)
#             new_state: Decoder hidden states. ((N, 1, D_dec), (N, 1, D_dec))
#             label: Label ID for LM. (1,)
#         """
#         label = np.full((1, 1), hyp.yseq[-1], dtype=np.int64)
#         str_labels = "_".join(list(map(str, hyp.yseq)))

#         if str_labels in cache:
#             dec_out, dec_state = cache[str_labels]

#         else:
#             input_dic = self.create_input_dic(label, *hyp.dec_state)
#             dec_out, *nexts = self.decoder.run(
#                 ['sequence'] + self.cache_out_name,
#                 input_dic
#             )
#             h_nexts, c_nexts = self.split(nexts)
#             dec_state = (h_next, c_next)
#             cache[str_labels] = (dec_out, dec_state)

#         return dec_out[0][0], dec_state, label[0]

#     def create_input_dic(self, label, h_prev, c_prev):
#         ret = {'label': label}
#         if self.dtype == "lstm":
#             ret.update({
#                 k: v for k, v in zip(self.cache_in_name, h_prev + c_prev)
#             })
#         else:
#             ret.update({
#                 k: v for k, v in zip(self.cache_in_name, h_prev)
#             })
#         return ret

#     def split(self, nexts):
#         if self.dtype == "lstm":
#             split_idx = len(nexts) // 2
#             return nexts[:split_idx], nexts[split_idx:]
#         else:
#             return nexts, None

#     def batch_score(
#         self,
#         hyps: Union[List[TransducerHypothesis]],
#         dec_states: Tuple[np.ndarray, Optional[np.ndarray]],
#         cache: Dict[str, Any],
#         use_lm: bool,
#     ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray]:
#         """One-step forward hypotheses.
#         Args:
#             hyps: TransducerHypothesis.
#             states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))
#             cache: Pairs of (dec_out, dec_states) for each label sequences. (keys)
#             use_lm: Whether to compute label ID sequences for LM.
#         Returns:
#             dec_out: Decoder output sequences. (B, D_dec)
#             dec_states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))
#             lm_labels: Label ID sequences for LM. (B,)
#         """
#         final_batch = len(hyps)

#         process = []
#         done = [None] * final_batch

#         for i, hyp in enumerate(hyps):
#             str_labels = "_".join(list(map(str, hyp.yseq)))

#             if str_labels in cache:
#                 done[i] = cache[str_labels]
#             else:
#                 process.append((str_labels, hyp.yseq[-1], hyp.dec_state))

#         if process:
#             labels = np.array([[p[1]] for p in process]).astype(np.int64)
#             h_prev, c_prev = self.create_batch_states(
#                 self.init_state(labels.shape[0]), [p[2] for p in process]
#             )

#             input_dic = self.create_input_dic(labels, h_prev, c_prev)
#             dec_out, *nexts = self.decoder.run(
#                 ['sequence'] + self.cache_out_name,
#                 input_dic
#             )
#             h_nexts, c_nexts = self.split(nexts)
#             new_states = (h_next, c_next)

#         j = 0
#         for i in range(final_batch):
#             if done[i] is None:
#                 state = self.select_state(new_states, j)

#                 done[i] = (dec_out[j], state)
#                 cache[process[j][0]] = (dec_out[j], state)

#                 j += 1

#         dec_out = np.concatenate([d[0] for d in done], axis=0)
#         dec_states = self.create_batch_states(dec_states, [d[1] for d in done])

#         if use_lm:
#             lm_labels = np.array([h.yseq[-1] for h in hyps]) \
#                 .reshape(final_batch, 1).astype(np.int64)

#             return dec_out, dec_states, lm_labels

#         return dec_out, dec_states, None

#     def select_state(
#         self, states: Tuple[np.ndarray, Optional[np.ndarray]], idx: int
#     ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
#         """Get specified ID state from decoder hidden states.
#         Args:
#             states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))
#             idx: State ID to extract.
#         Returns:
#             : Decoder hidden state for given ID.
#               ((N, 1, D_dec), (N, 1, D_dec))
#         """
#         return (
#             states[0][:, idx: idx + 1, :],
#             states[1][:, idx: idx + 1, :] if self.dtype == "lstm" else None,
#         )

#     def create_batch_states(
#         self,
#         states: Tuple[np.ndarray, Optional[np.ndarray]],
#         new_states: List[Tuple[np.ndarray, Optional[np.ndarray]]],
#         check_list: Optional[List] = None,
#     ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
#         """Create decoder hidden states.
#         Args:
#             states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))
#             new_states: Decoder hidden states. [N x ((1, D_dec), (1, D_dec))]
#         Returns:
#             states: Decoder hidden states. ((N, B, D_dec), (N, B, D_dec))
#         """
#         return (
#             np.concatenate([s[0] for s in new_states], axis=1),
#             np.concatenate([s[1] for s in new_states], axis=1)
#             if self.dtype == "lstm"
#             else None,
#         )
