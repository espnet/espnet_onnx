import subprocess
from pathlib import Path
from typing import Union


def optimize_model(
    input_model: Union[str, Path],
    output_model: Union[str, Path],
    num_heads: int = 0,
    hidden_size: int = 0,
    use_gpu: bool = False,
    only_onnxruntime: bool = False,
    model_type: str = "bert",
):
    args = [
        "python",
        "-m",
        "onnxruntime.transformers.optimizer",
        "--input",
        str(input_model),
        "--output",
        str(output_model),
        "--num_heads",
        str(num_heads),
        "--hidden_size",
        str(hidden_size),
    ]
    if use_gpu:
        args.extend(["--use_gpu"])

    if only_onnxruntime:
        args.extend(["--only_onnxruntime"])

    args.extend(["--model_type", model_type])

    subprocess.check_call(args)
