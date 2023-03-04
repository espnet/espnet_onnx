import os
from pathlib import Path

import numpy as np
import pytest
import torch

from espnet_onnx.export.asr.models import get_decoder, get_encoder, get_lm
from espnet_onnx.utils.config import save_config

encoder_cases = [
    "conformer_abs_pos",
    "conformer_rel_pos",
    "conformer_rpe_latest",
    "conformer_scaled",
    "transformer",
    "rnn_rnn",
    "rnn_rnnp",
    "rnn_vggrnn",
    "contextual_block_conformer",
    "contextual_block_transformer",
]

decoder_cases = [
    "transformer",
    # 'lightweight_conv',
    # 'lightweight_conv2d',
    # 'dynamic_conv',
    # 'dynamic_conv2d',
    "rnn_noatt",
    "rnn_dot",
    "rnn_add",
    "rnn_loc",
    # 'rnn_loc2d'
    "rnn_coverage",
    "rnn_covloc",
    "transducer",
]

lm_cases = ["transformer", "seqrnn", "transformer_pe"]


def save_model(onnx_model, export_dir, model_export, model_type):
    if model_type == "encoder":
        model_export._export_encoder(onnx_model, export_dir, verbose=False)
    elif model_type == "decoder":
        model_export._export_decoder(onnx_model, 256, export_dir, verbose=False)
    elif model_type == "lm":
        model_export._export_lm(onnx_model, export_dir, verbose=False)


@pytest.mark.parametrize("enc_type", encoder_cases)
def test_export_encoder(enc_type, load_config, model_export, get_class):
    model_config = load_config(enc_type, model_type="encoder")
    # prepare input_dim from frontend
    frontend = get_class(
        "frontend", model_config.frontend, model_config.frontend_conf.dic
    )
    input_size = frontend.output_size()

    # prepare encoder model
    encoder = get_class(
        "encoder",
        model_config.encoder,
        model_config.encoder_conf.dic,
        input_size=input_size,
    )
    export_dir = (
        Path(model_export.cache_dir) / "test" / "encoder" / f"./cache_{enc_type}"
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), str(export_dir / f"{enc_type}.pth"))

    # create encoder onnx wrapper and export
    enc_wrapper = get_encoder(encoder, frontend, None, {})
    save_model(enc_wrapper, export_dir, model_export, "encoder")

    if enc_type in ("contextual_block_conformer", "contextual_block_transformer"):
        # save position encoder parameters.
        np.save(export_dir / "pe", encoder.pos_enc.pe.numpy())
    assert len(os.path.join(export_dir, "*encoder.onnx")) > 0


@pytest.mark.parametrize("dec_type", decoder_cases)
def test_export_decoder(dec_type, load_config, model_export, get_class):
    model_config = load_config(dec_type, model_type="decoder")

    # prepare encoder model
    if model_config.decoder == "transducer":
        kwargs = {"vocab_size": 32000, "embed_pad": 0}
    else:
        kwargs = {"vocab_size": 32000, "encoder_output_size": 256}

    decoder = get_class(
        "decoder", model_config.decoder, model_config.decoder_conf.dic, **kwargs
    )
    export_dir = (
        Path(model_export.cache_dir) / "test" / "decoder" / f"./cache_{dec_type}"
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), str(export_dir / f"{dec_type}.pth"))

    dec_wrapper = get_decoder(decoder, {})
    save_model(dec_wrapper, export_dir, model_export, "decoder")

    decoder_config = dec_wrapper.get_model_config(export_dir)
    save_config(decoder_config, export_dir / "config.yaml")
    assert len(os.path.join(export_dir, "*decoder.onnx")) > 0


@pytest.mark.parametrize("lm_type", lm_cases)
def test_export_lm(lm_type, load_config, model_export, get_class):
    model_config = load_config(lm_type, model_type="lm")

    lm = get_class(
        "lm",
        model_config.lm,
        model_config.lm_conf.dic,
        vocab_size=32000,
    )
    lm.eval()
    export_dir = Path(model_export.cache_dir) / "test" / "lm" / f"./cache_{lm_type}"
    export_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lm.state_dict(), str(export_dir / f"{lm_type}.pth"))

    lm_wrapper = get_lm(lm, {})
    save_model(lm_wrapper, export_dir, model_export, "lm")

    lm_config = {"lm": lm_wrapper.get_model_config(export_dir)}
    lm_config["lm"].update({"use_lm": True})
    save_config(lm_config, export_dir / "config.yaml")
    assert len(os.path.join(export_dir, "*lm.onnx")) > 0
