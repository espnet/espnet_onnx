import glob
import os

import torch
import pytest

from espnet_onnx.export.optimize.optimizer import optimize_model
from espnet_onnx.export.asr.models import get_encoder

from ..op_test_utils import check_op_type_count

test_cases = [
    ["encoder", "transformer", 4, 256, "Attention", 3, False, False],
    # ['encoder', 'contextual_block_transformer', 4, 256, 3, 0, False],
    ["encoder", "transformer", 4, 256, "Attention", 3, True, False],
    ["encoder", "conformer_rpe_latest", 4, 256, "RelativeShift", 3, True, True],
    ["encoder", "conformer_rel_pos", 4, 256, "RelativeShift", 3, True, True],
    ["encoder", "conformer_rpe_latest", 4, 256, "RelPosAttention", 3, True, False],
    ["encoder", "conformer_rel_pos", 4, 256, "RelPosAttention", 3, True, False],
    # ['encoder', 'contextual_block_transformer', 4, 256, 3, 0, True, False],
    ["decoder", "transformer", 4, 256, "CrossAttention", 3, True, False],

    # CrossAttention with Slice node is not supported.
    # ["lm", "transformer_pe", 4, 256, "CrossAttention", 3, True, False],
    # ["lm", "transformer", 4, 256, "CrossAttention", 3, True, False],
]


@pytest.mark.parametrize(
    "model_type, model_name, n_head, h_size, node_name, node_num, use_custom_ort, use_gpu",
    test_cases,
)
def test_optimize(
    model_type,
    model_name,
    n_head,
    h_size,
    node_name,
    node_num,
    use_custom_ort,
    use_gpu,
    model_export,
    load_config,
    get_class,
    get_convert_map,
):
    export_dir = model_export.cache_dir / "test" / model_type / f"cache_{model_name}"
    export_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "encoder":
        export_dir = model_export.cache_dir / "test" / model_type / f"cache_{model_name}_opt"
        export_dir.mkdir(parents=True, exist_ok=True)
        model_config = load_config(model_name, model_type="encoder")
        frontend = get_class(
            "frontend", model_config.frontend, model_config.frontend_conf.dic
        )
        input_size = frontend.output_size()
        encoder = get_class(
            "encoder",
            model_config.encoder,
            model_config.encoder_conf.dic,
            input_size=input_size,
        )
        enc_wrapper = get_encoder(encoder, frontend, None, { "optimize": True }, get_convert_map)
        print(enc_wrapper)
        model_export._export_encoder(enc_wrapper, export_dir, verbose=False)

    output_dir = (
        model_export.cache_dir
        / "test"
        / "optimize"
        / model_type
        / f"cache_{model_name}"
    )

    input_model = glob.glob(os.path.join(export_dir, f"*{model_type}*"))[0]
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
        use_gpu=use_gpu,
    )

    # load the optimized model and check if the number of fused nodes is correct.
    nodes = {node_name: node_num}
    check_op_type_count(str(output_dir / model_name), nodes)
