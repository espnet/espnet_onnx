from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from typeguard import check_argument_types

from espnet_onnx.tts.abs_tts_model import AbsTTSModel


class Text2Speech(AbsTTSModel):
    """Wrapper class for espnet2.asr.bin.tts_inference.Text2Speech"""

    def __init__(
        self,
        tag_name: str = None,
        model_dir: Union[Path, str] = None,
        providers: List[str] = ["CPUExecutionProvider"],
        use_quantized: bool = False,
    ):
        assert check_argument_types()
        self._check_argument(tag_name, model_dir)
        self._load_config()

        # check onnxruntime version and providers
        self._check_ort_version(providers)

        # check if there is quantized model if use_quantized=True
        if self.config.tts_model.model_type == "Tacotron2":
            if (
                use_quantized
                and "quantized_model_path" not in self.config.tts_model.encoder.keys()
            ):
                raise RuntimeError("Configuration for quantized model is not defined.")
        elif (
            use_quantized and "quantized_model_path" not in self.config.tts_model.keys()
        ):
            # check if quantized model config is defined.
            raise RuntimeError("Configuration for quantized model is not defined.")

        self._build_model(providers, use_quantized)

    def __call__(
        self,
        text: str,
        feats: np.ndarray = None,
        sids: np.ndarray = None,
        spembs: np.ndarray = None,
        lids: np.ndarray = None,
    ) -> Dict[str, np.ndarray]:
        """Inference

        Args:
            data: Input speech data

        Returns:
            Dict[str, np.ndarray]
        """
        assert check_argument_types()

        # check argument
        options = dict()
        if self.tts_model.use_sids:
            if sids is None or spembs is None:
                raise RuntimeError("'sids' is missing.")
            else:
                options.update(sids=sids)
        if self.tts_model.use_spk_embed_dim:
            if spembs is None:
                raise RuntimeError("'spembs' is missing.")
            else:
                options.update(spembs=spembs)
        if self.tts_model.use_lids:
            if lids is None:
                raise RuntimeError("Missing required argument: 'lids'")
            else:
                options.update(lids=lids)
        if self.tts_model.use_feats:
            if feats is None:
                raise RuntimeError("Missing required argument: 'feats'")
            else:
                options.update(feats=feats)

        # preprocess text
        text = self.preprocess(text)
        output_dict = self.tts_model(text, **options)

        # postprocess
        if output_dict.get("feat_gen") is not None:
            output_dict["feat_gen"] = self.postprocess(output_dict["feat_gen"])

        if output_dict.get("att_w") is not None:
            duration, focus_rate = self.duration_calculator(output_dict["att_w"])
            output_dict.update(duration=duration, focus_rate=focus_rate)

        # vocoder is currently not supported.
        if self.vocoder is not None:
            input_feat = output_dict["feat_gen"]
            wav = self.vocoder(input_feat)
            output_dict.update(wav=wav)

        return output_dict

    def postprocess(self, feat):
        if self.normalize is not None:
            feat_length = np.array([feat.shape[0]], dtype=np.int64)
            feat, feat_length = self.normalize.inverse(feat[None, :], feat_length)
            return feat[0]
        else:
            return feat
