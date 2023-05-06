import numpy as np
import torch

from espnet_onnx.utils.function import make_pad_mask, mask_fill
from espnet_onnx.utils.torch_function import subsequent_mask


def run_onnx_enc(model, dummy_input, model_type):
    feat_length = np.array([dummy_input.shape[1]])
    encoder_out, _ = model.run(
        ["encoder_out", "encoder_out_lens"], {"feats": dummy_input}
    )
    if model_type[:3] == "rnn":
        encoder_out = mask_fill(
            encoder_out, make_pad_mask(feat_length, encoder_out, 1), 0.0
        )
    return encoder_out


def run_xformer_dec(model, dummy_input, dummy_yseq, model_type):
    if model_type == "torch":
        ys_mask = subsequent_mask(dummy_yseq.size(-1)).unsqueeze(0)
        logp, _ = model.forward_one_step(
            tgt=dummy_yseq, tgt_mask=ys_mask, memory=dummy_input, cache=None
        )
    else:
        logp, _ = model.batch_score(dummy_yseq, states=[None], xs=dummy_input)
    return logp


def run_rnn_dec(model, dummy_input, dummy_yseq):
    dummy_input = dummy_input[0]
    state = model.init_state(dummy_input)
    logp, state = model.score(dummy_yseq, state, dummy_input)
    return logp


def run_trans_dec(model, dummy_input, h, model_type):
    if model_type == "torch":
        emb = model.embed(dummy_input)
        y, _ = model.rnn_forward(emb, (h, h))
    else:
        input_dict = {"labels": dummy_input, "h_cache": h, "c_cache": h}
        y = model.decoder.run(["sequence"], input_dict)[0]
    return y


def run_onnx_front(model, dummy_input):
    input_dic = {
        "wav": dummy_input,
    }
    y = model.run(["feats", "feats_lens"], input_dic)[0]
    return y


def run_streaming_enc(model, dummy_input, dic, model_type):
    if model_type == "torch":
        y, *_ = model(
            dummy_input,
            torch.Tensor([dummy_input.shape[1]]),
            prev_states=dic,
            is_final=False,
            infer_mode=True,
        )
        return y
    else:
        encoder_out = model.run(["ys_pad"], dic)[0]
        return encoder_out
