import glob
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from espnet_onnx.asr.model.decoder import get_decoder
from espnet_onnx.asr.model.lm import get_lm
from espnet_onnx.utils.config import get_config

from .forward_utils import (run_onnx_enc, run_rnn_dec, run_trans_dec,
                            run_xformer_dec)

encoder_cases = [
    ("conformer_abs_pos", [50, 100]),
    ("conformer_rel_pos", [50, 100]),
    ("conformer_rpe_latest", [50, 100]),
    ("conformer_scaled", [50, 100]),
    ("transformer", [50, 100]),
    ("rnn_rnn", [50, 100]),
    ("rnn_rnnp", [50, 100]),
    ("rnn_vggrnn", [50, 100]),
]

decoder_cases = [
    ("transformer", [50, 100]),
    ("transducer", [1]),
    # ('lightweight_conv', [50, 100]),
    # ('lightweight_conv2d', [50, 100]),
    # ('dynamic_conv', [50, 100]),
    # ('dynamic_conv2d', [50, 100]),
    ("rnn_noatt", [50, 100]),
    ("rnn_dot", [50, 100]),
    ("rnn_add", [50, 100]),
    ("rnn_loc", [50, 100]),
    # ('rnn_loc2d', [50, 100]),
    ("rnn_coverage", [50, 100]),
    ("rnn_covloc", [50, 100]),
]

lm_cases = [
    ("transformer", [50, 100]),
    ("seqrnn", [50, 100]),
    ("transformer_pe", [50, 100]),
]

CACHE_DIR = Path.home() / ".cache" / "espnet_onnx" / "test"
PROVIDERS = ["CPUExecutionProvider"]


def check_output(out_t, out_o):
    out_t = out_t.detach().cpu().numpy()
    assert (
        out_t.shape == out_o.shape
    ), f"The shape of output of onnx {out_o.shape} should be the same with the output of torch model {out_t.shape}"

    mean_dif = np.mean((out_t - out_o) ** 2)
    assert mean_dif < 1e-10, f"Result of torch model and onnx model differs."


def get_predec_models(dec_type):
    if dec_type[:3] != "rnn":
        return []

    model_dir = CACHE_DIR / "pre_decoder" / f"./cache_{dec_type}"
    predecoders = []
    for p in glob.glob(os.path.join(model_dir, "*.onnx")):
        predecoders.append(ort.InferenceSession(p, providers=PROVIDERS))
    return predecoders


@pytest.mark.parametrize("enc_type, feat_lens", encoder_cases)
def test_infer_encoder(enc_type, feat_lens, load_config, get_class):
    model_dir = CACHE_DIR / "encoder" / f"./cache_{enc_type}"
    model_config = load_config(enc_type, model_type="encoder")

    # prepare input_dim from frontend
    frontend = get_class(
        "frontend", model_config.frontend, model_config.frontend_conf.dic
    )
    input_size = frontend.output_size()

    # prepare encoder model
    encoder_espnet = get_class(
        "encoder",
        model_config.encoder,
        model_config.encoder_conf.dic,
        input_size=input_size,
    )
    encoder_espnet.load_state_dict(torch.load(glob.glob(str(model_dir / "*.pth"))[0]))
    encoder_espnet.eval()
    model_file = glob.glob(os.path.join(model_dir, "*encoder.onnx"))[0]
    encoder_onnx = ort.InferenceSession(model_file, providers=PROVIDERS)

    # test output
    for fl in feat_lens:
        dummy_input = torch.randn(1, fl, input_size)  # (B, L, D)
        # compute torch model
        torch_out = encoder_espnet(dummy_input, torch.Tensor([fl]))
        if type(torch_out) == tuple:
            torch_out = torch_out[0]
        # compute onnx model
        onnx_out = run_onnx_enc(encoder_onnx, dummy_input.numpy(), enc_type)
        check_output(torch_out, onnx_out)


@pytest.mark.parametrize("dec_type, feat_lens", decoder_cases)
def test_infer_decoder(dec_type, feat_lens, load_config, get_class):
    model_dir = CACHE_DIR / "decoder" / f"./cache_{dec_type}"
    model_config = load_config(dec_type, model_type="decoder")

    # prepare decoder model
    if model_config.decoder == "transducer":
        kwargs = {"vocab_size": 32000, "embed_pad": 0}
    else:
        kwargs = {"vocab_size": 32000, "encoder_output_size": 256}
    decoder_espnet = get_class(
        "decoder", model_config.decoder, model_config.decoder_conf.dic, **kwargs
    )
    decoder_espnet.load_state_dict(torch.load(glob.glob(str(model_dir / "*.pth"))[0]))
    decoder_espnet.eval()
    decoder_onnx = get_decoder(
        get_config(model_dir / "config.yaml"), providers=PROVIDERS
    )

    # test output
    for fl in feat_lens:
        dummy_input = torch.randn(1, fl, 256)
        dummy_yseq = torch.LongTensor([0])
        if dec_type[:3] == "rnn":
            torch_out = run_rnn_dec(decoder_espnet, dummy_input, dummy_yseq)
            onnx_out = run_rnn_dec(
                decoder_onnx, dummy_input.numpy(), dummy_yseq.numpy()
            )

        elif dec_type == "transducer":
            dummy_yseq = dummy_yseq.unsqueeze(0)
            h = torch.randn(1, 1, 256)
            torch_out = run_trans_dec(decoder_espnet, dummy_yseq, h, "torch")
            onnx_out = run_trans_dec(
                decoder_onnx, dummy_yseq.numpy(), h.numpy(), "onnx"
            )

        else:
            dummy_yseq = dummy_yseq.unsqueeze(0)
            torch_out = run_xformer_dec(
                decoder_espnet, dummy_input, dummy_yseq, "torch"
            )
            onnx_out = run_xformer_dec(
                decoder_onnx, dummy_input.numpy(), dummy_yseq.numpy(), "onnx"
            )

        check_output(torch_out, onnx_out)


@pytest.mark.parametrize("lm_type, feat_lens", lm_cases)
def test_infer_lm(lm_type, feat_lens, load_config, get_class):
    model_dir = CACHE_DIR / "lm" / f"./cache_{lm_type}"
    model_config = load_config(lm_type, model_type="lm")

    # prepare language model
    torch_model = get_class(
        "lm",
        model_config.lm,
        model_config.lm_conf.dic,
        vocab_size=32000,
    )
    torch_model.load_state_dict(torch.load(glob.glob(str(model_dir / "*.pth"))[0]))
    torch_model.eval()
    onnx_model = get_lm(get_config(model_dir / "config.yaml"), providers=PROVIDERS)

    # test output
    for fl in feat_lens:
        dummy_input = torch.randn(1, fl, 256)
        tgt = torch.LongTensor([0, 1]).unsqueeze(0)

        torch_out, _ = torch_model.batch_score(tgt, [None], dummy_input)
        onnx_out, _ = onnx_model.batch_score(tgt.numpy(), [None], dummy_input.numpy())
        check_output(torch_out, onnx_out)
