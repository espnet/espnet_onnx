from typing import (
    Tuple,
    List
)

import onnxruntime
import numpy as np

from espnet_onnx.asr.frontend.frontend import Frontend
from espnet_onnx.asr.frontend.global_mvn import GlobalMVN
from espnet_onnx.asr.frontend.utterance_mvn import UtteranceMVN
from espnet_onnx.asr.scorer.interface import BatchScorerInterface
from espnet_onnx.utils.function import (
    subsequent_mask,
    make_pad_mask,
    mask_fill
)
from espnet_onnx.utils.config import Config


class VITS:
    def __init__(
        self,
        config: Config,
        providers: List[str],
        use_quantized: bool = False,
    ):
        self.config = config
        # vits requires duration_predictor as submodel
        self.durpred_config = config.submodel.duration_predictor
        
        if use_quantized:
            self.model = onnxruntime.InferenceSession(
                self.config.quantized_model_path,
                providers=providers
            )
            self.duration_predictor = onnxruntime.InferenceSession(
                self.durpred_config.quantized_model_path,
                providers=providers
            )
        else:
            self.model = onnxruntime.InferenceSession(
                self.config.model_path,
                providers=providers
            )
            self.duration_predictor = onnxruntime.InferenceSession(
                self.durpred_config.model_path,
                providers=providers
            )

        # vits requires duration_predictor as submodel
        self.durpred_config = config.submodel.duration_predictor
    
    def __call__(
        self,
        text: np.ndarray,
        feats: np.ndarray = None,
        spks: Optional[np.ndarray] = None,
        spemb: Optional[np.ndarray] = None,
        lids: Optional[np.ndarray] = None,
    ):
        # check input
        if len(text.shape) == 1:
            text = text[None, :]
        
        # duration predict if not teacher_forcing
        output_names = ['logw', 'm_p', 'logs_p', 'g']
        input_dict_dur = self.get_input_dur(text, spks, spemb, lids)
        out_dur = self.duration_predictor.run(output_names, input_dict_dur)
        
        # compute duration
        dur = np.ceil(np.exp(out_dir[0]) * input_dict_dur['x_mask'] * self.durpred_config.alpha)
        
        # compute vits flows
        output_names = ['wav', 'att_w', 'out_duration']
        input_dict_vits = self.get_input_vits(dur, out_dur, input_dict_dur, feats)
        wav, att_w, dur = self.model.run(output_names, input_dict_vits)
        return dict(wav=wav, att_w=att_w, dur=dur)
    
    def get_input_vits(self, dur, out_dur, input_dict_dur, feats):
        ret = {
            'm_p': out_dur[1],
            'logs_p': out_dur[2],
            'x_mask': input_dict_dur['x_mask'],
            'duration': dur
        }
        y_length = np.clip(np.sum(dur), 1, None)
        y_mask = ~make_pad_mask(y_length)[:, None]
        ret.update({'y_mask': y_mask})
        if self.config.use_teacher_forcing:
            assert feats is not None
            ret.update({'feats': feats})
            path = np.zeros((1, feats.shape[-1], input_dict_dur['text'].shape[1]))
        else:
            _attn_mask = input_dict_dur['x_mask'][:, :, None] * y_mask[..., None]
            path = torch.arange(_attn_mask.shape[2])
        ret.update({'path': path})
        return ret
    
    def get_input_dur(self, text, spks, spemb, lids):
        ret = {'text': text }
        input_names = [d.name for d in self.duration_predictor.get_inputs]
        x_mask = ~make_pad_mask(np.array([text.shape[1]]))
        z = np.random.random((1, 2, text.shape[1])) * self.durpred_config.noise_scale
        ret.update(
            x_mask=x_mask,
            z=z
        )
        if 'spks' in input_names:
            ret.update(
                spks=spks,
                spemb=spemb,
            )
        if 'lids' in input_names:
            ret.update(lids=lids)
        return ret