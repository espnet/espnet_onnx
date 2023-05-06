import argparse

from .asr.export_asr import ASRModelExport
from .tts.export_tts import TTSModelExport


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        required=True,
        type=str,
        choices=["asr", "tts"],
        help="task type",
    )

    parser.add_argument(
        "--input", required=True, type=str, help="path to the zip file."
    )
    parser.add_argument(
        "--tag", required=False, type=str, default=None, help="model name."
    )
    parser.add_argument(
        "--output",
        required=False,
        type=str,
        default=None,
        help="Path to the output model directory."
        + "If not provided, then output=${HOME}/.cache/espnet_onnx",
    )

    parser.add_argument(
        "--apply_quantize",
        required=False,
        action="store_true",
        help="apply quantization",
    )

    parser.add_argument(
        "--apply_optimize",
        required=False,
        action="store_true",
        help="apply optimization",
    )
    parser.add_argument(
        "--only_onnxruntime",
        required=False,
        action="store_true",
        help="apply optimization with onnxruntime.",
    )
    parser.add_argument(
        "--use_gpu",
        required=False,
        action="store_true",
        help="apply optimization for GPU execution",
    )
    parser.add_argument(
        "--float16",
        required=False,
        action="store_true",
        help="Convert all weights and nodes in float32 to float16",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_arguments()

    if args.model_type == "asr":
        m = ASRModelExport(args.output)

    elif args.model_type == "tts":
        m = TTSModelExport(args.output)

    if args.apply_optimize:
        m.set_export_config(
            optimize_lm=True,
            use_gpu=args.use_gpu,
            only_onnxruntime=args.only_onnxruntime,
            float16=args.float16,
        )

    m.export_from_zip(
        args.input, args.tag, quantize=args.apply_quantize, optimize=args.apply_optimize
    )
