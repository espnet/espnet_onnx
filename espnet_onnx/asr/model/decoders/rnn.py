from typing import List

import numpy as np
import onnxruntime

from espnet_onnx.asr.scorer.interface import BatchScorerInterface
from espnet_onnx.utils.config import Config
from espnet_onnx.utils.function import make_pad_mask


class RNNDecoder(BatchScorerInterface):
    def __init__(
        self, config: Config, providers: List[str], use_quantized: bool = False
    ):
        """Onnx support for espnet2.asr.decoder.rnn_decoder.RNNDecoder

        Args:
            config (Config):
            use_quantized (bool): Flag to use quantized model
        """
        self.config = config
        # predecoder
        self.predecoders = []
        self.is_noatt = []
        for p in config.predecoder:
            if use_quantized:
                model_path = p.quantized_model_path
            else:
                model_path = p.model_path

            self.is_noatt.append(p.att_type == "noatt")
            if model_path:
                self.predecoders.append(
                    onnxruntime.InferenceSession(model_path, providers=providers)
                )

        # decoder
        if use_quantized:
            self.decoder = onnxruntime.InferenceSession(
                config.quantized_model_path, providers=providers
            )
        else:
            self.decoder = onnxruntime.InferenceSession(
                config.model_path, providers=providers
            )

        # HP
        self.num_encs = len(self.predecoders)
        self.dunits = config.dunits
        self.dlayers = config.dlayers
        self.rnn_type = config.rnn_type
        self.decoder_length = config.decoder_length

        # predecoder
        self.decoder_output_names = self.get_decoder_output_names()

        # cache pre_computed features
        self.pre_compute_enc_h = []
        self.enc_h = []
        self.mask = []

        self.init_input_names()

    def init_input_names(self):
        # a_prev, enc_h, pceh, mask is not required
        # with some attention type.
        self.required_input_names = {}
        input_names = [d.name for d in self.decoder.get_inputs()]
        self.required_input_names["a_prev"] = (
            True if "a_prev" in str(input_names) else False
        )
        self.required_input_names["enc_h"] = (
            True if "enc_h" in str(input_names) else False
        )
        self.required_input_names["pceh"] = (
            True if "pceh" in str(input_names) else False
        )
        self.required_input_names["mask"] = (
            True if "mask" in str(input_names) else False
        )

    def get_decoder_output_names(self):
        return [d.name for d in self.decoder.get_outputs()]

    def zero_state(self, hs_pad):
        return np.zeros((hs_pad.shape[0], self.dunits), dtype=np.float32)

    def get_att_prev(self, x, att_type=None):
        att_prev = 1.0 - make_pad_mask([x[0].shape[0]])
        att_prev = (att_prev / np.array([x[0].shape[0]])[..., None]).astype(np.float32)
        if att_type == "location2d":
            att_prev = att_prev[..., None].reshape(-1, self.config.att_win, -1)
        if att_type in ("coverage", "coverage_location"):
            att_prev = att_prev[:, None, :]
        return att_prev

    def init_state(self, x):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs <= 1:
            x = [x]

        c_list = [self.zero_state(x[0][None, :])]
        z_list = [self.zero_state(x[0][None, :])]
        for _ in range(1, self.dlayers):
            c_list.append(self.zero_state(x[0][None, :]))
            z_list.append(self.zero_state(x[0][None, :]))

        strm_index = 0
        att_idx = min(strm_index, len(self.predecoders) - 1)

        a = []
        for att in self.config.predecoder:
            a += [self.get_att_prev(x, att.att_type)]

        if self.num_encs != 1:
            a += [self.get_att_prev(x)]  # atts + han

        # initialize cached parameters
        self.pre_compute_enc_h = []
        self.enc_h = []
        self.mask = []
        return dict(
            c_prev=c_list[:],
            z_prev=z_list[:],
            a_prev=a,
            workspace=(att_idx, z_list, c_list),
        )

    def score(self, yseq, state, x):
        att_idx, z_list, c_list = state["workspace"]
        vy = np.array([yseq[-1]])

        if self.num_encs <= 1:
            x = [x]

        # set initial state if attention type is NoAtt
        if self.num_encs == 0:
            # we d
            self.enc_h.append(x[0][None, :])
            self.mask.append(
                np.where(make_pad_mask([x[0].shape[0]]) == 1, -float("inf"), 0).astype(
                    np.float32
                )
            )

        # pre compute states of attention.
        if len(self.pre_compute_enc_h) == 0:
            for idx in range(self.num_encs):
                _pceh = self.predecoders[idx].run(
                    ["pre_compute_enc_h"], {"enc_h": x[idx][None, :]}
                )[0]
                self.pre_compute_enc_h.append(_pceh)
                self.enc_h.append(x[idx][None, :])
                self.mask.append(
                    np.where(
                        make_pad_mask([x[idx].shape[0]]) == 1, -float("inf"), 0
                    ).astype(np.float32)
                )

        input_dict = self.create_input_dic(vy, x, state)

        logp, *status_lists = self.decoder.run(self.decoder_output_names, input_dict)
        c_list, z_list, att_w = self._split(status_lists)
        return (
            logp,
            dict(
                c_prev=c_list,
                z_prev=z_list,
                a_prev=att_w,
                workspace=(att_idx, z_list, c_list),
            ),
        )

    def create_input_dic(self, vy, x, state):
        ret = {
            "vy": vy.astype(np.int64),
        }
        ret.update(
            {"z_prev_%d" % d: state["z_prev"][d] for d in range(self.decoder_length)}
        )
        ret.update(
            {"c_prev_%d" % d: state["c_prev"][d] for d in range(self.decoder_length)}
        )
        necs = max(1, self.num_encs)
        if self.required_input_names["a_prev"]:
            ret.update({"a_prev_%d" % d: state["a_prev"][d] for d in range(necs)})
        if self.required_input_names["enc_h"]:
            ret.update({"enc_h_%d" % d: self.enc_h[d] for d in range(necs)})
        if self.required_input_names["pceh"]:
            ret.update(
                {"pceh_%d" % d: self.pre_compute_enc_h[d] for d in range(self.num_encs)}
            )
        if self.required_input_names["mask"]:
            ret.update(
                {
                    "mask_%d" % d: self.mask[d]
                    for d in range(self.num_encs + sum(self.is_noatt))
                }
            )
        return ret

    def _split(self, status_lists):
        c_list = status_lists[: self.decoder_length]
        z_list = status_lists[self.decoder_length : 2 * self.num_encs]
        att_w = status_lists[2 * self.decoder_length :]
        return c_list, z_list, att_w
