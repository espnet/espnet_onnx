"""Run the notebook's public streaming API with real exported ASR models."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
import yaml
from espnet2.bin.asr_inference import Speech2Text

from espnet_onnx import StreamingSpeech2Text
from espnet_onnx.export import ASRModelExport


@pytest.fixture(scope="module")
def streaming_demo_model(tmp_path_factory):
    # A small, locally built model keeps this regression independent of Hub
    # downloads while exercising the same frontend/export/decoder pipeline.
    directory = tmp_path_factory.mktemp("streaming-demo")
    config_path = (
        Path(__file__).parents[1]
        / "test_config/integration/model_config/trf_ctc_att.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    config.update(
        encoder="contextual_block_transformer",
        normalize=None,
        normalize_conf={},
        token_type="char",
        bpemodel=None,
        token_list=["<blank>", "a", "b", "c", "<space>", "<unk>", "<sos/eos>"],
    )
    config["encoder_conf"].update(
        input_layer="conv2d", block_size=40, hop_size=16, look_ahead=16,
        init_average=True, ctx_pos_enc=True,
    )
    config["frontend_conf"].update(n_fft=512, hop_length=128)
    training_config = directory / "train.yaml"
    training_config.write_text(yaml.safe_dump(config))
    torch.manual_seed(0)
    model = Speech2Text(
        asr_train_config=training_config,
        beam_size=2,
        ctc_weight=0.3,
        lm_weight=0.0,
        maxlenratio=-8,
    )
    ASRModelExport(directory).export(model, "streaming-demo")
    return directory


@pytest.mark.parametrize("chunk_size", [777, None])
@pytest.mark.parametrize("short_recording", [False, True])
def test_demo_start_chunks_end(
    streaming_demo_model, chunk_size, short_recording, monkeypatch
):
    # Restrict threads, without replacing inference or any ASR component.
    session_class = ort.InferenceSession

    def session(*args, **kwargs):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        kwargs["sess_options"] = options
        return session_class(*args, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", session)
    streaming_model = StreamingSpeech2Text(
        "streaming-demo", cache_dir=streaming_demo_model
    )
    length = (
        1024 if short_recording
        else streaming_model.initial_wav_length + 32 * 16 * 4 * 128 + 37
    )
    y = (0.1 * np.sin(2 * np.pi * 220 * np.arange(length) / 16000)).astype(np.float32)
    step = chunk_size or streaming_model.hop_size

    # Keep this sequence aligned with demo/onnx_conversion_demo.ipynb.
    streaming_model.start()
    partial_results = []
    for start in range(0, len(y), step):
        nbest = streaming_model(y[start : start + step])
        if nbest:
            partial_results.append(nbest[0][0])
    final_nbest = streaming_model.end()

    assert final_nbest
    assert final_nbest[0][3].yseq[-1] == streaming_model.beam_search.eos
    assert np.isfinite(final_nbest[0][3].score)
    assert streaming_model.end() is final_nbest
    if not short_recording:
        assert partial_results
        assert streaming_model.encoder.n_processed_blocks > 32
    assert (
        streaming_model.streaming_states["buffer_after_downsampling"].shape[1] == 24
    )
    manual_features = np.array(streaming_model.enc_feats)
    simulated = streaming_model.simulate(y)
    assert simulated[0][2] == final_nbest[0][2]
    np.testing.assert_allclose(streaming_model.enc_feats, manual_features)
