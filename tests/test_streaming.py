"""
Tests for the streaming module.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from voice_translator.streaming import AudioStreamer, StreamState

# ---------------------------------------------------------------------------
# _rms
# ---------------------------------------------------------------------------


def test_rms_of_silence_is_zero():
    silence = np.zeros(1600, dtype=np.float32)
    assert AudioStreamer._rms(silence) == 0.0


def test_rms_of_constant_signal():
    signal = np.full(1600, 0.5, dtype=np.float32)
    assert abs(AudioStreamer._rms(signal) - 0.5) < 1e-6


def test_rms_increases_with_amplitude():
    quiet = np.full(1600, 0.01, dtype=np.float32)
    loud = np.full(1600, 0.1, dtype=np.float32)
    assert AudioStreamer._rms(loud) > AudioStreamer._rms(quiet)


# ---------------------------------------------------------------------------
# _transcribe_chunk
# ---------------------------------------------------------------------------


def _make_segment(text: str) -> MagicMock:
    segment = MagicMock()
    segment.text = text
    return segment


def test_transcribe_chunk_joins_segments():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        [_make_segment(" Hello "), _make_segment("world ")],
        MagicMock(),
    )
    streamer = AudioStreamer(mock_model, source_language="en")

    result = streamer._transcribe_chunk(np.zeros(1600, dtype=np.float32))

    assert result == "Hello world"


def test_transcribe_chunk_passes_vad_filter_and_language():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock())
    streamer = AudioStreamer(mock_model, source_language="pt")

    streamer._transcribe_chunk(np.zeros(1600, dtype=np.float32))

    call_kwargs = mock_model.transcribe.call_args.kwargs
    assert call_kwargs["vad_filter"] is True
    assert call_kwargs["language"] == "pt"


# ---------------------------------------------------------------------------
# push_chunk
# ---------------------------------------------------------------------------


@patch("voice_translator.streaming.STREAM_SAMPLE_RATE", 100)
@patch("voice_translator.streaming.MIN_CHUNK_DURATION_S", 0.2)
def test_push_chunk_keeps_accumulating_below_min_duration():
    mock_model = MagicMock()
    streamer = AudioStreamer(mock_model, source_language=None)

    loud_chunk = np.full(5, 0.5, dtype=np.float32)  # above silence threshold
    state, text = streamer.push_chunk(StreamState(), loud_chunk, chunk_duration_s=0.05)

    assert text is None
    assert len(state.buffer) == 5
    mock_model.transcribe.assert_not_called()


@patch("voice_translator.streaming.STREAM_SAMPLE_RATE", 100)
@patch("voice_translator.streaming.MIN_CHUNK_DURATION_S", 0.1)
@patch("voice_translator.streaming.SILENCE_DURATION_S", 0.15)
def test_push_chunk_closes_after_enough_silence():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([_make_segment("hi")], MagicMock())
    streamer = AudioStreamer(mock_model, source_language=None)

    loud_chunk = np.full(15, 0.5, dtype=np.float32)  # 0.15s at rate=100, past min
    silent_chunk = np.zeros(5, dtype=np.float32)

    state, text = streamer.push_chunk(StreamState(), loud_chunk, chunk_duration_s=0.15)
    assert text is None  # enough audio, but no silence yet

    state, text = streamer.push_chunk(state, silent_chunk, chunk_duration_s=0.05)
    assert text is None  # 0.05s of silence, below the 0.15s threshold

    state, text = streamer.push_chunk(state, silent_chunk, chunk_duration_s=0.05)
    assert text is None  # 0.10s of silence, still below 0.15s

    state, text = streamer.push_chunk(state, silent_chunk, chunk_duration_s=0.05)
    assert text == "hi"  # 0.15s of silence reached: chunk closes
    assert len(state.buffer) == 0  # state reset after closing


@patch("voice_translator.streaming.STREAM_SAMPLE_RATE", 100)
@patch("voice_translator.streaming.MIN_CHUNK_DURATION_S", 0.05)
@patch("voice_translator.streaming.SILENCE_DURATION_S", 999)  # never close via silence
@patch("voice_translator.streaming.MAX_CHUNK_DURATION_S", 0.15)
def test_push_chunk_force_closes_at_max_duration():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([_make_segment("forced")], MagicMock())
    streamer = AudioStreamer(mock_model, source_language=None)

    loud_chunk = np.full(10, 0.5, dtype=np.float32)  # 0.10s per chunk at rate=100

    state, text = streamer.push_chunk(StreamState(), loud_chunk, chunk_duration_s=0.10)
    assert text is None  # 0.10s accumulated, below 0.15s max

    state, text = streamer.push_chunk(state, loud_chunk, chunk_duration_s=0.10)
    assert text == "forced"  # 0.20s accumulated, past 0.15s max: force closed


@patch("voice_translator.streaming.STREAM_SAMPLE_RATE", 100)
@patch("voice_translator.streaming.MIN_CHUNK_DURATION_S", 0.1)
@patch("voice_translator.streaming.SILENCE_DURATION_S", 0.05)
def test_push_chunk_returns_none_text_when_transcription_is_empty():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock())  # VAD removed everything
    streamer = AudioStreamer(mock_model, source_language=None)

    loud_chunk = np.full(15, 0.5, dtype=np.float32)
    silent_chunk = np.zeros(5, dtype=np.float32)

    state, _ = streamer.push_chunk(StreamState(), loud_chunk, chunk_duration_s=0.15)
    state, text = streamer.push_chunk(state, silent_chunk, chunk_duration_s=0.05)

    assert text is None  # empty transcription surfaces as None, not ""
    assert len(state.buffer) == 0  # buffer still resets even with empty text


def test_push_chunk_resets_silence_counter_on_new_speech():
    mock_model = MagicMock()
    streamer = AudioStreamer(mock_model, source_language=None)

    loud_chunk = np.full(1600, 0.5, dtype=np.float32)
    silent_chunk = np.zeros(1600, dtype=np.float32)

    state, _ = streamer.push_chunk(StreamState(), loud_chunk, chunk_duration_s=0.1)
    state, _ = streamer.push_chunk(state, silent_chunk, chunk_duration_s=0.1)
    assert state.silence_duration_s == 0.1

    # speech resumes: silence counter must reset to 0
    state, _ = streamer.push_chunk(state, loud_chunk, chunk_duration_s=0.1)
    assert state.silence_duration_s == 0.0
