
import numpy as np
import torch
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask as espnet_subsequent_mask
from espnet_onnx.utils.function import (
    mask_fill, make_pad_mask, subsequent_mask
)


def rnn_onnx_enc(model, dummy_input):
    feat_length = np.array([dummy_input.shape[1]])
    encoder_out, encoder_out_lens = \
        model.run(["encoder_out", "encoder_out_lens"], {
            "feats": dummy_input,
            "feats_length": feat_length
        })
    encoder_out = mask_fill(encoder_out, make_pad_mask(
        feat_length, encoder_out, 1), 0.0)
    return encoder_out


def xformer_onnx_enc(model, dummy_input):
    feat_length = np.array([dummy_input.shape[1]])
    mask = (make_pad_mask(feat_length)[
        :, None, :] == False).astype(np.float64)
    encoder_out = \
        model.run(["encoder_out", "encoder_out_lens"], {
            "feats": dummy_input,
            "mask": mask
        })[0]
    return encoder_out


def xformer_onnx_dec(onnx_model, dummy_input):
    ys = np.array([[0, 1]]).astype(np.int64)
    logp, state = onnx_model.batch_score(
        ys, states=[None], xs=dummy_input
    )
    return logp


def xformer_torch_dec(model, dummy_input):
    tgt = torch.LongTensor([0, 1]).unsqueeze(0)
    ys_mask = espnet_subsequent_mask(len(tgt)).unsqueeze(0)
    logp, state = model.forward_one_step(
        tgt=tgt,
        tgt_mask=ys_mask,
        memory=dummy_input,
        cache=None
    )
    return logp


def rnn_torch_dec(model, dummy_input):
    dummy_input = dummy_input[0]
    state = model.init_state(dummy_input)
    yseq = torch.LongTensor([0, 1])
    logp, state = model.score(yseq, state, dummy_input)
    return logp


def rnn_onnx_dec(onnx_model, dummy_input):
    dummy_input = dummy_input[0]
    state = onnx_model.init_state(dummy_input)
    yseq = np.array([0, 1]).astype(np.int64)
    logp, state = onnx_model.score(yseq, state, dummy_input)
    return logp

