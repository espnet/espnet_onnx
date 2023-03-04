import glob
import os
from pathlib import Path

import pytest
import torch

from espnet_onnx.export.asr.models import get_encoder
from espnet_onnx.export.optimize.optimizer import optimize_model

from ..op_test_utils import check_op_type_count

encoder_cases = [
    "conformer_hubert",
    "transformer_hubert",
    "rnn_hubert",
]

optimize_cases = [
    ["conformer_hubert", "transformer", 4, 768, 12, 0, False],
    ["conformer_hubert", "transformer", 4, 768, 12, 0, True],
]


def save_model(onnx_model, export_dir, model_export):
    model_export._export_encoder(onnx_model, export_dir, verbose=False)


@pytest.mark.parametrize("enc_type", encoder_cases)
def test_export_frontend(enc_type, load_config, model_export, get_class):
    model_config = load_config(enc_type, model_type="frontend")
    # prepare input_dim from frontend
    frontend = get_class(
        "frontend", model_config.frontend, model_config.frontend_conf.dic
    )

    export_dir = (
        Path(model_export.cache_dir) / "test" / "frontend" / f"./cache_{enc_type}"
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    torch.save(frontend.state_dict(), str(export_dir / f"{enc_type}_frontend.pth"))

    # create encoder onnx wrapper and export
    # prepare encoder model
    input_size = frontend.output_size()
    encoder = get_class(
        "encoder",
        model_config.encoder,
        model_config.encoder_conf.dic,
        input_size=input_size,
    )
    enc_wrapper = get_encoder(encoder, frontend, None, {})
    save_model(enc_wrapper, export_dir, model_export)

    assert len(os.path.join(export_dir, "*frontend.onnx")) > 0


@pytest.mark.parametrize(
    "model_type, model_name, n_head, h_size, n_att, n_cross_att, use_custom_ort",
    optimize_cases,
)
def test_optimize_frontend(
    model_type,
    model_name,
    n_head,
    h_size,
    n_att,
    n_cross_att,
    use_custom_ort,
    model_export,
):
    export_dir = model_export.cache_dir / "test" / "frontend" / f"./cache_{model_type}"
    output_dir = (
        model_export.cache_dir
        / "test"
        / "optimize"
        / model_type
        / f"cache_{model_name}"
    )

    input_model = glob.glob(os.path.join(export_dir, f"*frontend.onnx"))[0]
    model_name = os.path.basename(input_model)
    if use_custom_ort:
        opt_model_type = "espnet"
    else:
        opt_model_type = "bert"

    optimize_model(
        input_model=str(input_model),
        output_model=str(output_dir / model_name),
        num_heads=n_head,
        hidden_size=h_size,
        model_type=opt_model_type,
    )

    # load the optimized model and check if the number of fused nodes is correct.
    nodes = {}
    if n_att > 0:
        nodes["Attention"] = n_att
    if n_cross_att > 0:
        nodes["CrossAttention"] = n_cross_att

    check_op_type_count(str(output_dir / model_name), nodes)
