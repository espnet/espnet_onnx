

import torch
import torch.nn as nn

from ..abs_model import AbsModel


class AttentionReference(nn.Module, AbsModel):
    def __init__(self, model, scaling = 2.0):
        self.mlp_psd = model.mlp_psd
        self.gvec = model.gvec
        self.scaling = scaling
    
    def forward(self, psd_feat):
        # xs.shape : (B, C, F)
        mlp_psd = self.mlp_psd(psd_feat)
        e = self.gvec(torch.tanh(mlp_psd)).squeeze(-1)
        u = F.softmax(self.scaling * e, dim=-1)
        return u
    
    def get_dummy_inputs(self, feat_dim):
        psd_feat = torch.randn(1, 100, feat_dim)
        return psd_feat

    def get_input_names(self):
        return ['psd_feat']

    def get_output_names(self):
        return ['u']
    
    def get_dynamic_axes(self):
        return {
            'psd_feat': { 1: 'feats_length'},
            'u': { 1: 'u_length'}
        }

    def get_model_config(self, path):
        return {
            'model_path': os.path.join(path, 'attention_reference.onnx'),
        }
