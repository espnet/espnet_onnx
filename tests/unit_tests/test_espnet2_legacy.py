from pathlib import Path

import torch
from espnet2.legacy.nets.pytorch_backend.transformer.embedding import PositionalEncoding

from espnet_onnx.export.asr.models.layers.embed import OnnxPositionalEncoding
from espnet_onnx.utils.export_function import get_replace_modules, replace_modules


def test_espnet2_positional_encoding_conversion():
    convert_map = (
        Path(__file__).resolve().parents[2] / "espnet_onnx/export/convert_map.yml"
    )
    original = PositionalEncoding(d_model=8, dropout_rate=0.0).eval()
    inputs = torch.randn(2, 12, 8)
    expected = original(inputs)

    converted = replace_modules(get_replace_modules(convert_map, "asr"), original)

    assert isinstance(converted, OnnxPositionalEncoding)
    torch.testing.assert_close(converted(inputs), expected)
