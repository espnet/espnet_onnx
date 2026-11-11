from types import SimpleNamespace

import numpy as np
import pytest

from espnet_onnx.asr.asr_streaming import StreamingSpeech2Text
from espnet_onnx.asr.beam_search.batch_beam_search_online import BatchBeamSearchOnline
from espnet_onnx.asr.scorer.interface import BatchScorerInterface
from espnet_onnx.utils.config import Config


class TailScorer(BatchScorerInterface):
    """Recognize a token that is only available after encoder flushing."""

    def batch_score(self, ys, states, xs):
        scores = np.full((len(ys), 4), -100.0, dtype=np.float32)
        for i in range(len(ys)):
            token = int(xs[i, -1, 0]) if ys.shape[1] == 1 else 3
            scores[i, token] = 0
        return scores, [None] * len(ys)


class BufferedEncoder:
    def reset(self):
        self.blocks = 0
        self.n_processed_blocks = 0
        self.flush_calls = 0
        self.chunk_lengths = []
        self.chunks = []

    def init_state(self):
        return {"step": 0}

    def __call__(self, speech, speech_length, states):
        self.chunk_lengths.append(speech.shape[1])
        self.chunks.append(speech[0].copy())
        return np.ones((1, 4, 1), dtype=np.float32), states

    def forward_final(self, states):
        assert states["step"] == self.flush_calls
        self.flush_calls += 1
        return np.full((1, 2, 1), 2, dtype=np.float32), {"step": self.flush_calls}

    def increment(self):
        self.blocks += 1
        self.n_processed_blocks += 1


def make_streaming_model(look_ahead=2):
    model = StreamingSpeech2Text.__new__(StreamingSpeech2Text)
    model.config = Config({"encoder": {"hop_size": 2, "look_ahead": look_ahead}})
    model.encoder = BufferedEncoder()
    model.initial_wav_length = 10
    model.hop_size = 4
    model._window_overlap = 0
    model.beam_search = BatchBeamSearchOnline(
        Config({
            "beam_size": 1,
            "pre_beam_ratio": 1.0,
            "pre_beam_score_key": None,
            "minlenratio": 0.0,
            "maxlenratio": 0.0,
        }),
        Config({"sos": 3, "eos": 3, "list": ["blank", "partial", "final", "eos"]}),
        {"decoder": TailScorer()},
        {"decoder": 1.0},
    )
    model.converter = SimpleNamespace(ids2tokens=lambda ids: [str(i) for i in ids])
    model.tokenizer = SimpleNamespace(tokens2text=lambda tokens: " ".join(tokens))
    model.start()
    return model


@pytest.mark.parametrize("look_ahead", [1, 2, 3])
def test_end_decodes_buffered_tail_and_is_idempotent(look_ahead):
    model = make_streaming_model(look_ahead)
    partial = model(np.zeros(10, dtype=np.float32))
    assert partial[0][0] == "1"
    final = model.end()
    assert final[0][0] == "2"
    assert final[0][3].yseq[-1] == model.beam_search.eos
    assert len(model.enc_feats) == 4 + look_ahead
    calls = model.encoder.flush_calls
    assert model.end() is final
    assert model.encoder.flush_calls == calls
    with pytest.raises(RuntimeError, match="start"):
        model(np.zeros(4, dtype=np.float32))
    model.start()
    assert model(np.zeros(10, dtype=np.float32))[0][0] == "1"


@pytest.mark.parametrize(
    "length, chunks",
    [(1, [10]), (10, [10]), (11, [10, 4]), (14, [10, 4]), (18, [10, 4, 4])],
)
def test_simulate_returns_final_result(length, chunks):
    model = make_streaming_model()
    result = model.simulate(np.zeros(length, dtype=np.float32))
    assert result[0][0] == "2"
    assert model.encoder.chunk_lengths == chunks


def test_empty_utterance_has_no_hypothesis():
    model = make_streaming_model()
    assert model.simulate(np.zeros(0, dtype=np.float32)) == []
    assert model.encoder.flush_calls == 0


@pytest.mark.parametrize("chunk_size", [1, 3, 4, 10, 29])
def test_start_chunks_end_preserves_audio_and_finalizes(chunk_size):
    speech = np.arange(29, dtype=np.float32)
    model = make_streaming_model()
    model.start()
    for offset in range(0, len(speech), chunk_size):
        model(speech[offset : offset + chunk_size])
    result = model.end()
    assert result[0][0] == "2"
    assert model.encoder.chunk_lengths == [10, 4, 4, 4, 4, 4]
    received = np.concatenate(model.encoder.chunks)
    np.testing.assert_array_equal(received[:len(speech)], speech)
    np.testing.assert_array_equal(received[len(speech):], 0)


def test_initial_small_chunk_waits_for_context():
    model = make_streaming_model()
    assert model(np.zeros(4, dtype=np.float32)) == []
    assert model.encoder.chunk_lengths == []
    assert model.end()[0][0] == "2"
    assert model.encoder.chunk_lengths == [10]


def test_left_padding_once_and_no_padding_between_chunks():
    model = make_streaming_model()
    model._window_overlap = 2
    speech = np.arange(1, 18, dtype=np.float32)
    model.start()
    for offset in range(0, len(speech), 3):
        model(speech[offset : offset + 3])
    chunks_before_end = len(model.encoder.chunks)
    model.end()
    assert len(model.encoder.chunks) == chunks_before_end
    assert model.encoder.chunk_lengths == [10, 4, 4, 4, 4]
    chunks = model.encoder.chunks
    np.testing.assert_array_equal(chunks[0], np.r_[0, speech[:9]])
    for previous, current in zip(chunks, chunks[1:]):
        np.testing.assert_array_equal(previous[-2:], current[:2])
        assert np.all(current > 0)
    reconstructed = np.concatenate([chunks[0], *(chunk[2:] for chunk in chunks[1:])])
    np.testing.assert_array_equal(reconstructed, np.r_[0, speech])
