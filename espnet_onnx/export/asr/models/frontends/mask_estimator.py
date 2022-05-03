

import torch
import torch.nn as nn

from espnet_onnx.export.asr.encoder.rnn import RNNEncoderLayer
from ..abs_model import AbsModel


class MaskEstimator(nn.Module, AbsModel):
    def __init__(self, model):
        self.rnn = RNNEncoderLayer(model.brnn)
        self.linears = model.linears
    
    def forward(self, xs, ilens):
        # xs.shape : (B*C, T, D)
        xs, _, _ = self.rnn(xs, ilens)
        masks = []
        for linear in self.linears:
            mask = linear(xs)
            mask = torch.sigmoid(mask)
            masks.append(mask)
        
        return masks
    
    def get_dummy_inputs(self, feat_dim):
        xs = torch.randn(1, 100, feat_dim)
        feats_length = torch.LongTensor([feats.size(1)])
        return (xs, feats_length)

    def get_input_names(self):
        return ['feats', 'feats_length']

    def get_output_names(self):
        return [f'mask_{i}' for i in range(len(self.linears))]
    
    def get_dynamic_axes(self):
        ret = {
            'feats': { 1: 'feats_length'}
        }
        for i in range(len(self.linears)):
            ret.update({
                f'mask_{i}': { 1: f'mask_{i}_length' }
            })
        return ret

    def get_model_config(self, path, model_prefix:str = ""):
        return {
            'model_path': os.path.join(path, model_prefix + 'mask_estimator.onnx')
        }
