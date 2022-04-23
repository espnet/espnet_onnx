from typing import Tuple

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


class StreamingEncoder:
    def __init__(
        self,
        encoder_config: Config,
        use_quantized: bool = False
    ):
        self.config = encoder_config
        if use_quantized:
            self.encoder = onnxruntime.InferenceSession(
                self.config.quantized_model_path)
        else:
            self.encoder = onnxruntime.InferenceSession(
                self.config.model_path)

        self.frontend = Frontend(self.config.frontend)
        if self.config.do_normalize:
            if self.config.normalize.type == 'gmvn':
                self.normalize = GlobalMVN(self.config.normalize)
            elif self.config.normalize.type == 'utterance_mvn':
                self.normalize = UtteranceMVN(self.config.normalize)
        
        self.pos_enc = np.load(self.config.pe_path)
        self.n_processed_blocks = 0
        self.n_layers = self.config.n_layers
        self.overlap_size = self.config.block_size - self.config.hop_size

        # if self.config.do_preencoder:
        #     self.preencoder = Preencoder(self.config.preencoder)

        # if self.config.do_postencoder:
        #     self.postencoder = Postencoder(self.config.postencoder)

    def __call__(
        self, speech: np.ndarray, speech_length: np.ndarray, states
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        # 1. Extract feature
        feats, feat_length = self.frontend(speech, speech_length)
        
        # 2. normalize with global MVN
        if self.config.do_normalize:
            feats, feat_length = self.normalize(feats, feat_length)

        # if self.config.do_preencoder:
        #     feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 3. forward encoder
        encoder_out, next_states = \
            self.forward_encoder(feats, states)

        # if self.config.do_postencoder:
        #     encoder_out, encoder_out_lens = self.postencoder(
        #         encoder_out, encoder_out_lens
        #     )

        return encoder_out, next_states

    def forward_encoder(self, feats, states):
        # outputs = ['ys_pad', 'next_buffer_before_downsampling', 'next_buffer_after_downsampling',
        #         'next_addin', 'next_encoder_ctx']
        outputs = ['ys_pad', 'next_buffer_after_downsampling',
                'next_addin', 'next_encoder_ctx']
        input_dict = self.get_input_dict(feats, states)
        # for k,v in input_dict.items():
        #     print((k, v.shape))
        # ys_pad, nbbd, nbad, naddin, nec = \
        #     self.encoder.run(outputs, input_dict)
        ys_pad, nbad, naddin, nec = \
            self.encoder.run(outputs, input_dict)
        ret = {
            # 'buffer_before_downsampling' : nbbd,
            'buffer_after_downsampling' : nbad,
            'prev_addin' : naddin,
            'past_encoder_ctx' : nec,
        }

        return ys_pad, ret

    def get_input_dict(self, x, state):
        # x.length : hop_size * subsample + 1
        mask = np.zeros((1, 1, self.config.block_size+2, self.config.block_size+2), dtype=np.float32)
        mask[..., 1:, :-1] = 1
        if self.n_processed_blocks == 0:
            start = 0
            indicies = np.array([0, self.config.block_size-self.config.look_ahead, self.overlap_size], dtype=np.int64)
        else:
            start = self.config.hop_size * self.n_processed_blocks
            offset = self.config.block_size - self.config.look_ahead - self.config.hop_size + 1
            indicies = np.array([offset, offset+self.config.hop_size, self.overlap_size], dtype=np.int64)
        return {
            'xs_pad': x[:, :-1],
            'mask': mask,
            # 'buffer_before_downsampling': state['buffer_before_downsampling'],
            'buffer_after_downsampling': state['buffer_after_downsampling'],
            'prev_addin': state['prev_addin'],
            'pos_enc_xs': self.pos_enc[:, start : start + self.config.block_size],
            'pos_enc_addin': self.pos_enc[:, self.n_processed_blocks : self.n_processed_blocks + 1],
            'past_encoder_ctx': state['past_encoder_ctx'],
            'indicies': indicies
        }
    
    def reset(self):
        self.n_processed_blocks = 0
    
    def increment(self):
        self.n_processed_blocks += 1
    
    def init_state(self):
        return {
            # 'buffer_before_downsampling' : np.zeros((1, self.config.subsample*2, self.config.frontend.logmel.n_mels), dtype=np.float32),
            'buffer_after_downsampling' : np.zeros((1, self.config.block_size - self.config.hop_size, self.pos_enc.shape[-1]), dtype=np.float32),
            'prev_addin' : np.zeros((1, 1, self.pos_enc.shape[-1]), dtype=np.float32),
            'past_encoder_ctx' : np.zeros((1, self.n_layers, self.pos_enc.shape[-1]), dtype=np.float32),
        }
