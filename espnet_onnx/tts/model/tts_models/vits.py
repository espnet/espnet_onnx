from typing import List

import numpy as np
import onnxruntime

from espnet_onnx.utils.config import Config


class VITS:
    def __init__(
        self,
        config: Config,
        providers: List[str],
        use_quantized: bool = False,
    ):
        self.config = config
        if use_quantized:
            self.model = onnxruntime.InferenceSession(
                self.config.quantized_model_path, providers=providers
            )
        else:
            self.model = onnxruntime.InferenceSession(
                self.config.model_path, providers=providers
            )

        self.input_names = [d.name for d in self.model.get_inputs()]
        self.use_sids = "sids" in self.input_names
        self.use_spk_embed_dim = "spembs" in self.input_names
        self.use_lids = "lids" in self.input_names
        self.use_feats = "feats" in self.input_names

    def __call__(
        self,
        text: np.ndarray,
        feats: np.ndarray = None,
        sids: np.ndarray = None,
        spembs: np.ndarray = None,
        lids: np.ndarray = None,
    ):
        output_names = ["wav", "att_w", "dur"]
        input_dict = self.get_input_dict(text, feats, sids, spembs, lids)
        wav, att_w, dur = self.model.run(output_names, input_dict)
        return dict(wav=wav, att_w=att_w, dur=dur)

    def get_input_dict(self, text, feats, sids, spembs, lids):
        ret = {"text": text}
        ret = self._set_input_dict(ret, "feats", feats)
        ret = self._set_input_dict(ret, "sids", sids)
        ret = self._set_input_dict(ret, "spembs", spembs)
        ret = self._set_input_dict(ret, "lids", lids)
        return ret

    def _set_input_dict(self, dic, key, value):
        if key in self.input_names:
            assert value is not None
            dic[key] = value
        return dic
