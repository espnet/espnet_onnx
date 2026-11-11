import shutil
import subprocess
import warnings
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

    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        unsupported_espnet = (
            model_type == "espnet"
            and "invalid choice: 'espnet'" in result.stderr
        )
        if not unsupported_espnet:
            raise subprocess.CalledProcessError(
                result.returncode, args, output=result.stdout, stderr=result.stderr
            )
        # Stock ORT does not contain ESPnet's optional custom graph fusions.
        Path(output_model).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_model, output_model)
        warnings.warn(
            "This ONNX Runtime build does not support ESPnet graph fusion; "
            "the exported model was kept without static optimization.",
            RuntimeWarning,
            stacklevel=2,
        )
