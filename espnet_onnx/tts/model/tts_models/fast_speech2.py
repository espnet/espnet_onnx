from typing import List

import numpy as np
import onnxruntime

from espnet_onnx.utils.config import Config


class FastSpeech2:
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
        self.output_names = ["feat_gen", "out_duration", "out_pitch", "out_energy"]
        self.use_sids = "sids" in self.input_names
        self.use_lids = "lids" in self.input_names
        self.use_feats = "feats" in self.input_names
        self.use_spk_embed_dim = "spembs" in self.input_names

    def __call__(
        self,
        text: np.ndarray,
        feats: np.ndarray = None,
        sids: np.ndarray = None,
        spembs: np.ndarray = None,
        lids: np.ndarray = None,
    ):
        input_dict = self.get_input_dict(text, feats, sids, spembs, lids)
        feat_gen, dur, pitch, energy = self.model.run(self.output_names, input_dict)
        return dict(feat_gen=feat_gen, dur=dur, pitch=pitch, energy=energy)

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
