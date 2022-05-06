
import torch
import torch.nn as nn


class OnnxVITSModel(nn.Module):
    def __init__(self, model, use_teacher_forcing: bool = False):
        self.model = model
        self.use_teacher_forcing = use_teacher_forcing
    
    def forward(self, text, feats, sids, spembs, lids,
                durations):
        pass
    
    