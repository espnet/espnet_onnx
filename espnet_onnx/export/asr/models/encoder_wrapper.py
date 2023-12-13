import os
import torch
import torch.nn as nn
from espnet_onnx.utils.abs_model import AbsExportModel
from espnet_onnx.export.asr.get_config import (
    get_frontend_config,
    get_norm_config,
)


class DefaultEncoder(nn.Module, AbsExportModel):
    def __init__(self, model, frontend, feats_dim=80, **kwargs):
        super().__init__()
        self.model = model
        self.model_name = "default_encoder"
        self.frontend = frontend
        self.feats_dim = feats_dim
        self.get_frontend(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.is_optimizable():
            self.num_heads = self.model.num_heads
            self.hidden_size = self.model.hidden_size

    def get_frontend(self, kwargs):
        from espnet_onnx.export.asr.models import get_frontend_models

        self.frontend_model = get_frontend_models(self.frontend, kwargs)
        if self.frontend_model is not None:
            self.submodel = []
            self.submodel.append(self.frontend_model)
            self.feats_dim = self.frontend_model.output_dim

    def forward(self, feats):
        feats_length = torch.ones(feats[:, :, 0].shape).sum(dim=-1).type(torch.long)
        return self.model(feats, feats_length)

    def get_output_size(self):
        if "RNNEncoder" in type(self.model).__module__:
            # check RNN first
            return self.model.model_output_size
        elif "espnet2" in type(self.model).__module__:
            # default espnet model
            return self.model.encoders[0].size
        else:
            # optimized espnet_onnx model
            return self.model.model.encoders[0].size

    def is_optimizable(self):
        return (
            "espnet_onnx" in type(self.model).__module__
            and "rnn" not in type(self.model).__module__
        )

    def get_dummy_inputs(self):
        feats = torch.randn(1, 100, self.feats_dim)
        return feats

    def get_input_names(self):
        return ["feats"]

    def get_output_names(self):
        return ["encoder_out", "encoder_out_lens"]

    def get_dynamic_axes(self):
        return {"feats": {1: "feats_length"}, "encoder_out": {1: "enc_out_length"}}

    def get_model_config(self, asr_model=None, path=None):
        ret = {}
        is_vggrnn = "rnn" in type(self.model).__module__ and any(
            ["OnnxVGG2l" in type(m).__name__ for m in asr_model.encoder.modules()]
        )

        ret.update(
            enc_type="DefaultEncoder",
            model_path=os.path.join(path, f"{self.model_name}.onnx"),
            is_vggrnn=is_vggrnn,
            frontend=get_frontend_config(
                asr_model.frontend, self.frontend_model, path=path
            ),
            do_normalize=asr_model.normalize is not None,
            do_postencoder=asr_model.postencoder is not None,
        )
        if ret["do_normalize"]:
            ret.update(normalize=get_norm_config(asr_model.normalize, path))
        return ret
