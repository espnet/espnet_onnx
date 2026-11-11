import glob
import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import onnxruntime as ort
import pytest
import torch

from espnet_onnx.asr.model.decoder import get_decoder
from espnet_onnx.asr.model.lm import get_lm
from espnet_onnx.asr.model.encoders.streaming import StreamingEncoder
from espnet_onnx.asr.asr_streaming import StreamingSpeech2Text
from espnet_onnx.asr.frontend.frontend import Frontend
from espnet_onnx.export.asr.get_config import get_frontend_config
from espnet_onnx.utils.config import Config, get_config

from .forward_utils import (run_onnx_enc, run_rnn_dec, run_streaming_enc,
                            run_trans_dec, run_xformer_dec)

encoder_cases = [
    ("conformer_abs_pos", [50, 100]),
    ("conformer_rel_pos", [50, 100]),
    ("conformer_rpe_latest", [50, 100]),
    ("conformer_scaled", [50, 100]),
    ("transformer", [50, 100]),
    ("rnn_rnn", [50, 100]),
    ("rnn_rnnp", [50, 100]),
    ("rnn_vggrnn", [50, 100]),
    ("e_branchformer", [50, 100]),
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

streaming_cases = [
    ("contextual_block_transformer", 4, 0),
    ("contextual_block_transformer", 4, 1),
    ("contextual_block_conformer", 4, 0),
    ("contextual_block_conformer", 4, 1),
    ("contextual_block_transformer6", 6, 0),
    ("contextual_block_transformer6", 6, 1),
    ("contextual_block_conformer6", 6, 0),
    ("contextual_block_conformer6", 6, 1),
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
        torch_out = encoder_espnet(dummy_input, torch.LongTensor([fl]))
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
        batch_size = 4 if dec_type == "transformer" else 1
        dummy_input = torch.randn(batch_size, fl, 256)
        if dec_type[:3] == "rnn":
            dummy_yseq = torch.LongTensor([0])
        else:
            dummy_yseq = torch.zeros((batch_size, 1), dtype=torch.long)
        if dec_type[:3] == "rnn":
            torch_out = run_rnn_dec(decoder_espnet, dummy_input, dummy_yseq)
            onnx_out = run_rnn_dec(
                decoder_onnx, dummy_input.numpy(), dummy_yseq.numpy()
            )

        elif dec_type == "transducer":
            dummy_yseq = dummy_yseq[:1]
            h = torch.randn(1, 1, 256)
            torch_out = run_trans_dec(decoder_espnet, dummy_yseq, h, "torch")
            onnx_out = run_trans_dec(
                decoder_onnx, dummy_yseq.numpy(), h.numpy(), "onnx"
            )

        else:
            torch_out = run_xformer_dec(
                decoder_espnet, dummy_input, dummy_yseq, "torch"
            )
            onnx_out = run_xformer_dec(
                decoder_onnx, dummy_input.numpy(), dummy_yseq.numpy(), "onnx"
            )

        check_output(torch_out, onnx_out)

    if dec_type == "transformer":
        for batch_size in (1, 4, 20):
            check_cached_batch_scores(decoder_espnet, decoder_onnx, batch_size)


def check_cached_batch_scores(torch_model, onnx_model, batch_size):
    """Exercise changing prefix lengths with real per-hypothesis caches."""
    memory = torch.randn(batch_size, 30, 256)
    tokens = torch.randint(1, 100, (batch_size, 3))
    torch_states = [None] * batch_size
    onnx_states = [None] * batch_size
    for length in (1, 2, 3):
        prefix = tokens[:, :length]
        with torch.no_grad():
            expected, torch_states = torch_model.batch_score(
                prefix, torch_states, memory
            )
        actual, onnx_states = onnx_model.batch_score(
            prefix.numpy(), onnx_states, memory.numpy()
        )
        check_output(expected, actual)


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

    if lm_type.startswith("transformer"):
        for batch_size in (1, 4, 20):
            check_cached_batch_scores(torch_model, onnx_model, batch_size)


@pytest.mark.parametrize(
    "streaming_type, subsample, n_processed_blocks",
    streaming_cases,
)
def test_infer_streaming_encoder(
    streaming_type,
    subsample,
    n_processed_blocks,
    load_config,
    get_class,
):
    model_dir = CACHE_DIR / "encoder" / f"./cache_{streaming_type}"
    model_config = load_config(streaming_type, model_type="encoder")
    block_size = model_config.encoder_conf.block_size
    hop_size = model_config.encoder_conf.hop_size
    look_ahead = model_config.encoder_conf.look_ahead

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

    # prepare pos enc
    pe = np.load(os.path.join(model_dir, "pe.npy"))

    # test output for the first iteration
    dummy_input, torch_cache, onnx_input = create_input_stream(
        input_size,
        block_size,
        hop_size,
        subsample,
        model_config.encoder_conf.linear_units,
        model_config.encoder_conf.num_blocks,
        look_ahead,
        n_processed_blocks,
        pe,
        random=n_processed_blocks > 0,
    )

    # compute torch model
    torch_out = run_streaming_enc(encoder_espnet, dummy_input, torch_cache, "torch")

    # compute onnx model
    onnx_out = run_streaming_enc(encoder_onnx, None, onnx_input, "onnx")
    check_output(torch_out, onnx_out)

    if n_processed_blocks == 0:
        # Exercise real state handoff, including the final look-ahead flush.
        streaming = StreamingEncoder.__new__(StreamingEncoder)
        streaming.config = Config({
            "block_size": block_size,
            "hop_size": hop_size,
            "look_ahead": look_ahead,
            "subsample": subsample,
            "frontend": {"logmel": {"n_mels": input_size}},
        })
        streaming.encoder = encoder_onnx
        streaming.pos_enc = pe
        streaming.n_layers = model_config.encoder_conf.num_blocks
        streaming.outputs = [output.name for output in encoder_onnx.get_outputs()]
        streaming.reset()
        states = streaming.init_state()
        for length in ((block_size + 2) * subsample + 1, hop_size * subsample + 1):
            features = np.zeros((1, length, input_size), dtype=np.float32)
            _, states = streaming.forward_encoder(features, states)
            streaming.increment()
        tail, states = streaming.forward_final(states)
        assert tail.shape == (1, hop_size, pe.shape[-1])
        assert np.isfinite(tail).all()
        assert states["buffer_after_downsampling"].shape[1] == block_size - hop_size

        # The public API must accept fixed-size live chunks, including the
        # first and final partial windows, without accumulating STFT frames.
        frontend_config = get_frontend_config(frontend)
        frontend_config["stft"].update(
            n_fft=512, win_length=512, hop_length=128, center=False
        )
        frontend_config["logmel"]["n_fft"] = 512
        streaming.config.frontend = Config(frontend_config)
        streaming.config.do_normalize = False
        streaming.frontend = Frontend(streaming.config.frontend, PROVIDERS)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        streaming.encoder = ort.InferenceSession(
            model_file, sess_options=options, providers=PROVIDERS
        )
        live = StreamingSpeech2Text.__new__(StreamingSpeech2Text)
        live.config = Config({"encoder": streaming.config})
        live.encoder = streaming
        live.beam_search = Mock(return_value=[])
        live._window_overlap = 512
        live.initial_wav_length = (block_size + 2) * subsample * 128 + 512
        live.hop_size = hop_size * subsample * 128 + 512
        speech = np.zeros(
            live.initial_wav_length + 32 * (live.hop_size - 512) + 37,
            dtype=np.float32,
        )
        live.start()
        for start in range(0, len(speech), live.hop_size):
            live(speech[start : start + live.hop_size])
        assert live.end() == []
        expected = np.array(live.enc_feats)
        assert np.isfinite(expected).all()
        assert (
            live.streaming_states["buffer_after_downsampling"].shape[1]
            == block_size - hop_size
        )
        assert live.simulate(speech) == []
        np.testing.assert_allclose(live.enc_feats, expected)


def create_input_stream(
    input_size,
    block_size,
    hop_size,
    subsample,
    linear_units,
    num_blocks,
    look_ahead,
    n_processed_blocks,
    pe,
    random=True,
):
    mask = np.zeros(
        (1, 1, block_size + 2, block_size + 2),
        dtype=np.float32,
    )
    mask[..., 1:, :-1] = 1
    if n_processed_blocks == 0:
        # +subsample is for test. No need to add for inference.
        dummy_input = torch.randn(
            1, (block_size + 2) * subsample + subsample * 2, input_size
        )
        start = 0
        is_first = np.array([1], dtype=np.int64)
    else:
        # +subsample is for test. No need to add for inference.
        dummy_input = torch.randn(1, hop_size * subsample + subsample * 2, input_size)
        start = hop_size * n_processed_blocks
        is_first = np.array(
            [0],
            dtype=np.int64,
        )

    if random:
        buffer_before = torch.randn(1, subsample * 2, dummy_input.shape[2])
        buffer_after = torch.randn(1, block_size - hop_size, linear_units)
        prev_addin = torch.randn(1, 1, linear_units)
        past_encoder_ctx = torch.randn(1, num_blocks, linear_units)
        torch_cache = {
            "prev_addin": prev_addin,
            "buffer_before_downsampling": buffer_before,
            "buffer_after_downsampling": buffer_after,
            "ilens_buffer": hop_size * n_processed_blocks,
            "n_processed_blocks": n_processed_blocks,
            "past_encoder_ctx": past_encoder_ctx,
        }
        onnx_cache = {
            "xs_pad": dummy_input.detach().numpy(),
            "mask": mask,
            "buffer_before_downsampling": buffer_before.detach().numpy(),
            "buffer_after_downsampling": buffer_after.detach().numpy(),
            "prev_addin": prev_addin.detach().numpy(),
            "pos_enc_xs": pe[:, start : start + block_size],
            "pos_enc_addin": pe[:, n_processed_blocks : n_processed_blocks + 1],
            "past_encoder_ctx": past_encoder_ctx.detach().numpy(),
            "is_first": is_first,
        }
    else:
        buffer_before = torch.zeros(1, subsample * 2, dummy_input.shape[2])
        buffer_after = torch.zeros(1, block_size - hop_size, linear_units)
        prev_addin = torch.zeros(1, 1, linear_units)
        past_encoder_ctx = torch.zeros(1, num_blocks, linear_units)
        torch_cache = None
        onnx_cache = {
            "xs_pad": dummy_input.detach().numpy(),
            "mask": mask,
            "buffer_before_downsampling": buffer_before.detach().numpy(),
            "buffer_after_downsampling": buffer_after.detach().numpy(),
            "prev_addin": prev_addin.detach().numpy(),
            "pos_enc_xs": pe[:, start : start + block_size],
            "pos_enc_addin": pe[:, n_processed_blocks : n_processed_blocks + 1],
            "past_encoder_ctx": past_encoder_ctx.detach().numpy(),
            "is_first": is_first,
        }
    return dummy_input, torch_cache, onnx_cache
