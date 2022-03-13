import torch
import torch.nn as nn

from espnet2.lm.seq_rnn_lm import SequentialRNNLM as espnetSequentialRNNLM
from espnet2.lm.transformer_lm import TransformerLM as espnetTransformerLM

from .language_models.lm import (
    SequentialRNNLM,
    TransformerLM
)
from .abs_model import AbsModel


class LanguageModel(nn.Module, AbsModel):
    def __init__(self, model):
        super().__init__()
        if isinstance(model, espnetSequentialRNNLM):
            self.model = SequentialRNNLM(model)

        elif isinstance(model, espnetTransformerLM):
            self.model = TransformerLM(model)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def get_dummy_inputs(self, *args, **kwargs):
        return self.model.get_dummy_inputs(*args, **kwargs)

    def get_input_names(self, *args, **kwargs):
        return self.model.get_input_names(*args, **kwargs)

    def get_output_names(self, *args, **kwargs):
        return self.model.get_output_names(*args, **kwargs)

    def get_dynamic_axes(self, *args, **kwargs):
        return self.model.get_dynamix_axes(*args, **kwargs)

    def get_model_config(self, *args, **kwargs):
        return self.model.get_model_config(*args, **kwargs)
