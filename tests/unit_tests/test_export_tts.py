import argparse
import os
from pathlib import Path

import pytest
import torch
from espnet2.tasks.tts import TTSTask

from espnet_onnx.export.tts.get_config import get_preprocess_config
from espnet_onnx.export.tts.models import get_vocoder
from espnet_onnx.export.tts.models.tts_models.fastspeech2 import \
    OnnxFastSpeech2
from espnet_onnx.export.tts.models.tts_models.jets import OnnxJETSModel
from espnet_onnx.export.tts.models.tts_models.tacotron2 import (
    OnnxTacotron2Decoder, OnnxTacotron2Encoder)
from espnet_onnx.export.tts.models.tts_models.vits import OnnxVITSModel

tts_cases = [
    ["vits", OnnxVITSModel],
    ["fastspeech2", OnnxFastSpeech2],
    ["tacotron2_loc", [OnnxTacotron2Encoder, OnnxTacotron2Decoder]],
    # ['tacotron2_for', [OnnxTacotron2Encoder, OnnxTacotron2Decoder]],
    ["jets", OnnxJETSModel],
]

voc_cases = [
    "hifigan",
    "melgan",
    "parallel_wavegan",
    # 'style_melgan'
]

prepro_type = [
    "no_preprocess",
    "default",
    "tokenType_none",
    # 'bpe_whisper_en'
]


def save_model(torch_model, onnx_model, model_export, model_type, model_name):
    export_dir = (
        Path(model_export.cache_dir) / "test" / model_type / f"./cache_{model_name}"
    )
    export_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "tts":
        model_export._export_tts(onnx_model, export_dir, verbose=False)

    if model_type == "vocoder":
        model_export._export_vocoder(onnx_model, export_dir, verbose=False)

    torch.save(torch_model.state_dict(), str(export_dir / f"{model_type}.pth"))
    return export_dir


@pytest.mark.parametrize("tts_type, cls", tts_cases)
def test_export_tts(tts_type, cls, load_config, model_export_tts, get_class):
    model_config = load_config(tts_type, model_type="tts")
    tts = get_class(
        "tts", model_config.tts, model_config.tts_conf.dic, idim=78, odim=513
    )
    if "tacotron" in tts_type:
        tts_wrapper_encoder = cls[0](tts)
        save_model(
            tts.enc, tts_wrapper_encoder, model_export_tts, "tts", tts_type + "_encoder"
        )
        tts_wrapper_decoder = cls[1](tts)
        export_dir = save_model(
            tts.dec, tts_wrapper_decoder, model_export_tts, "tts", tts_type + "_decoder"
        )
    else:
        tts_wrapper = cls(tts)
        export_dir = save_model(tts, tts_wrapper, model_export_tts, "tts", tts_type)
    assert len(os.path.join(export_dir, "*.onnx")) > 0


@pytest.mark.parametrize("voc_type", voc_cases)
def test_export_vocoder(voc_type, load_config, model_export_tts, get_class):
    model_config = load_config(voc_type, model_type="vocoder")
    vocoder = get_class(
        "vocoder", model_config.vocoder_type, model_config.vocoder_params
    )
    voc_wrapper, _ = get_vocoder(vocoder, {})
    export_dir = save_model(vocoder, voc_wrapper, model_export_tts, "vocoder", voc_type)
    assert len(os.path.join(export_dir, "*.onnx")) > 0


# Test if configuration of the preprocess is correct
@pytest.mark.parametrize("prepro_type", prepro_type)
def test_export_preprocess(prepro_type, load_config):
    model_config = load_config(prepro_type, model_type="tts_preprocess")
    preprocess_fn = TTSTask.build_preprocess_fn(
        argparse.Namespace(**model_config.dic), False
    )
    preprocess_config = get_preprocess_config(preprocess_fn, "")
