from typing import List

import numpy as np
import onnxruntime

from espnet_onnx.utils.config import Config
from espnet_onnx.utils.function import make_pad_mask


class Tacotron2:
    def __init__(
        self,
        config: Config,
        providers: List[str],
        use_quantized: bool = False,
    ):
        self.config = config
        # load encoder and decoder.
        self.encoder = self._load_model(config.encoder, providers, use_quantized)
        self.predecoder = self._load_model(
            config.decoder.predecoder, providers, use_quantized
        )
        self.decoder = self._load_model(config.decoder, providers, use_quantized)

        if self.config.decoder.postdecoder.onnx_export:
            self.postdecoder = self._load_model(
                config.decoder.postdecoder, providers, use_quantized
            )
        else:
            self.postdecoder = None

        # HP
        self.input_names = [d.name for d in self.encoder.get_inputs()]
        self.use_sids = "sids" in self.input_names
        self.use_lids = "lids" in self.input_names
        self.use_feats = "feats" in self.input_names
        self.use_spk_embed_dim = "spembs" in self.input_names
        self.dlayers = self.config.decoder.dlayers
        self.dunits = self.config.decoder.dunits
        self.threshold = self.config.decoder.threshold
        self.decoder_input_names = [d.name for d in self.decoder.get_inputs()]
        self.decoder_output_names = [d.name for d in self.decoder.get_outputs()]
        self.use_att_constraint = self.config.decoder.use_att_constraint

    def _load_model(self, config, providers, use_quantized):
        if use_quantized:
            return onnxruntime.InferenceSession(
                config.quantized_model_path, providers=providers
            )
        else:
            return onnxruntime.InferenceSession(config.model_path, providers=providers)

    def __call__(
        self,
        text: np.ndarray,
        feats: np.ndarray = None,
        sids: np.ndarray = None,
        spembs: np.ndarray = None,
        lids: np.ndarray = None,
        backward_window: np.ndarray = 1,
        forward_window: np.ndarray = 3,
    ):
        # compute encoder and initialize states
        input_enc = self.get_input_enc(text, feats, sids, spembs, lids)
        h = self.encoder.run(["h"], input_enc)[0]
        hs = h[None, :]

        idx = 0
        outs = []
        probs = []
        att_ws = []
        maxlen = int(h.shape[0] * self.config.decoder.maxlenratio)
        minlen = int(h.shape[0] * self.config.decoder.minlenratio)
        c_list, z_list, prev_out = self.init_state(hs)

        # compute decoder
        prev_att_w = None
        last_attended_idx = 0
        while True:
            idx += self.config.decoder.reduction_factor
            input_dec = self.get_input_dec(
                c_list,
                z_list,
                prev_att_w,
                prev_out,
                last_attended_idx,
                backward_window,
                forward_window,
            )
            out, prob, a_prev, prev_out, *cz_states = self.decoder.run(
                self.decoder_output_names, input_dec
            )

            if self.config.decoder.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + a_prev
            else:
                prev_att_w = a_prev

            c_list, z_list = self._split(cz_states)

            outs += [out]
            probs += [prob]
            att_ws += [a_prev]

            if self.use_att_constraint:
                last_attended_idx = int(a_prev.argmax())

            # check whether to finish generation
            if int(sum(probs[-1] >= self.threshold)) > 0 or idx >= maxlen:
                # check mininum length
                if idx < minlen:
                    continue
                outs = np.concatenate(outs, axis=2)  # (1, odim, L)
                if self.postdecoder is not None:
                    outs = self.postdecoder.run(["out"], {"x": outs})[0]  # (1, odim, L)
                probs = np.concatenate(probs, axis=0)
                att_ws = np.concatenate(att_ws, axis=0)
                break
        return dict(feat_gen=outs, prob=probs, att_w=att_ws)

    def get_input_enc(self, text, feats, sids, spembs, lids):
        ret = {"text": text}
        ret = self._set_input_dict(ret, "feats", feats)
        ret = self._set_input_dict(ret, "sids", sids)
        ret = self._set_input_dict(ret, "spembs", spembs)
        ret = self._set_input_dict(ret, "lids", lids)
        return ret

    def get_input_dec(
        self,
        c_list,
        z_list,
        a_prev,
        prev_in,
        last_attended_idx,
        backward_window,
        forward_window,
    ):
        ret = {}
        ret.update({f"c_prev_{i}": cl for i, cl in enumerate(c_list)})
        ret.update({f"z_prev_{i}": zl for i, zl in enumerate(z_list)})
        ret.update(
            {
                "pceh": self.pre_compute_enc_h,
                "enc_h": self.enc_h,
                "mask": self.mask,
                "prev_in": prev_in,
            }
        )
        if a_prev is not None:
            ret.update(a_prev=a_prev)
        else:
            ret.update(a_prev=self.get_att_prev(self.enc_h))

        if self.use_att_constraint:
            backward_idx = last_attended_idx - backward_window
            forward_idx = last_attended_idx + forward_window
            last_att_mask = np.zeros(self.enc_h.shape[1])
            if backward_idx > 0:
                last_att_mask[:backward_idx] = -10000.0
            if forward_idx < self.enc_h.shape[1]:
                last_att_mask[forward_idx:] = -10000.0
            ret.update(last_att_mask=last_att_mask.astype(np.float32))
        return ret

    def _set_input_dict(self, dic, key, value):
        if key in self.input_names:
            assert value is not None
            dic[key] = value
        return dic

    def zero_state(self):
        return np.zeros((1, self.dunits), dtype=np.float32)

    def get_att_prev(self, x, att_type=None):
        # x : (1, T, D)
        att_prev = 1.0 - make_pad_mask([x.shape[1]])
        att_prev = (att_prev / np.array([x.shape[1]])[..., None]).astype(np.float32)
        if att_type == "location2d":
            att_prev = att_prev[..., None].reshape(-1, self.config.att_win, -1)
        if att_type in ("coverage", "coverage_location"):
            att_prev = att_prev[:, None, :]
        return att_prev

    def init_state(self, x):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        c_list = [self.zero_state()]
        z_list = [self.zero_state()]
        for _ in range(1, self.dlayers):
            c_list.append(self.zero_state())
            z_list.append(self.zero_state())

        prev_out = np.zeros((1, self.config.decoder.odim), dtype=np.float32)

        # compute predecoder
        self.pre_compute_enc_h = self.predecoder.run(
            ["pre_compute_enc_h"], {"enc_h": x}
        )[0]
        self.enc_h = x
        self.mask = np.where(
            make_pad_mask([x[0].shape[0]]) == 1, -float("inf"), 0
        ).astype(np.float32)
        return c_list[:], z_list[:], prev_out

    def _split(self, status_lists):
        len_list = len(status_lists)
        c_list = status_lists[: len_list // 2]
        z_list = status_lists[len_list // 2 :]
        return c_list, z_list
