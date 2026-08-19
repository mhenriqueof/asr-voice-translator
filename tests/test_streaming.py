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
    streamer = AudioStreamer(mock_model)

    result = streamer._transcribe_chunk(np.zeros(1600, dtype=np.float32), "en")

    assert result == "Hello world"


def test_transcribe_chunk_passes_vad_filter_and_language():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock())
    streamer = AudioStreamer(mock_model)

    streamer._transcribe_chunk(np.zeros(1600, dtype=np.float32), "pt")

    call_kwargs = mock_model.transcribe.call_args.kwargs
    assert call_kwargs["vad_filter"] is True
    assert call_kwargs["language"] == "pt"


def test_transcribe_chunk_defaults_language_to_none():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock())
    streamer = AudioStreamer(mock_model)

    streamer._transcribe_chunk(np.zeros(1600, dtype=np.float32))

    call_kwargs = mock_model.transcribe.call_args.kwargs
    assert call_kwargs["language"] is None


# ---------------------------------------------------------------------------
# push_chunk
# ---------------------------------------------------------------------------


@patch("voice_translator.streaming.STREAM_SAMPLE_RATE", 100)
@patch("voice_translator.streaming.MIN_CHUNK_DURATION_S", 0.2)
def test_push_chunk_keeps_accumulating_below_min_duration():
    mock_model = MagicMock()
    streamer = AudioStreamer(mock_model)

    loud_chunk = np.full(5, 0.5, dtype=np.float32)  # above silence threshold
    state, accumulated, new_text = streamer.push_chunk(
        StreamState(), loud_chunk, chunk_duration_s=0.05, source_language="pt"
    )

    assert accumulated == ""  # nothing transcribed yet
    assert new_text == ""
    assert len(state.buffer) == 5
    mock_model.transcribe.assert_not_called()


@patch("voice_translator.streaming.STREAM_SAMPLE_RATE", 100)
@patch("voice_translator.streaming.MIN_CHUNK_DURATION_S", 0.1)
@patch("voice_translator.streaming.SILENCE_DURATION_S", 0.15)
def test_push_chunk_closes_after_enough_silence():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([_make_segment("hi")], MagicMock())
    streamer = AudioStreamer(mock_model)

    loud_chunk = np.full(15, 0.5, dtype=np.float32)  # 0.15s at rate=100, past min
    silent_chunk = np.zeros(5, dtype=np.float32)

    state, accumulated, new_text = streamer.push_chunk(
        StreamState(), loud_chunk, chunk_duration_s=0.15, source_language="pt"
    )
    assert accumulated == "" and new_text == ""  # enough audio, but no silence yet

    state, accumulated, new_text = streamer.push_chunk(
        state, silent_chunk, chunk_duration_s=0.05, source_language="pt"
    )
    assert accumulated == "" and new_text == ""  # 0.05s silence, below 0.15s

    state, accumulated, new_text = streamer.push_chunk(
        state, silent_chunk, chunk_duration_s=0.05, source_language="pt"
    )
    assert accumulated == "" and new_text == ""  # 0.10s silence, still below 0.15s

    state, accumulated, new_text = streamer.push_chunk(
        state, silent_chunk, chunk_duration_s=0.05, source_language="pt"
    )
    assert accumulated == "hi"  # 0.15s of silence reached: chunk closes
    assert new_text == "hi"  # this call is the one that produced new text
    assert state.accumulated_text == "hi"
    assert len(state.buffer) == 0  # buffer resets after closing


@patch("voice_translator.streaming.STREAM_SAMPLE_RATE", 100)
@patch("voice_translator.streaming.MIN_CHUNK_DURATION_S", 0.05)
@patch("voice_translator.streaming.SILENCE_DURATION_S", 999)  # never close via silence
@patch("voice_translator.streaming.MAX_CHUNK_DURATION_S", 0.15)
def test_push_chunk_force_closes_at_max_duration():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([_make_segment("forced")], MagicMock())
    streamer = AudioStreamer(mock_model)

    loud_chunk = np.full(10, 0.5, dtype=np.float32)  # 0.10s per chunk at rate=100

    state, accumulated, new_text = streamer.push_chunk(
        StreamState(), loud_chunk, chunk_duration_s=0.10, source_language="pt"
    )
    assert accumulated == "" and new_text == ""  # 0.10s accumulated, below 0.15s max

    state, accumulated, new_text = streamer.push_chunk(
        state, loud_chunk, chunk_duration_s=0.10, source_language="pt"
    )
    assert accumulated == "forced"  # 0.20s accumulated, past 0.15s max: force closed
    assert new_text == "forced"


@patch("voice_translator.streaming.STREAM_SAMPLE_RATE", 100)
@patch("voice_translator.streaming.MIN_CHUNK_DURATION_S", 0.1)
@patch("voice_translator.streaming.SILENCE_DURATION_S", 0.05)
def test_push_chunk_keeps_previous_text_when_transcription_is_empty():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock())  # VAD removed everything
    streamer = AudioStreamer(mock_model)

    loud_chunk = np.full(15, 0.5, dtype=np.float32)
    silent_chunk = np.zeros(5, dtype=np.float32)

    state, _, _ = streamer.push_chunk(
        StreamState(), loud_chunk, chunk_duration_s=0.15, source_language="pt"
    )
    state, accumulated, new_text = streamer.push_chunk(
        state, silent_chunk, chunk_duration_s=0.05, source_language="pt"
    )

    assert accumulated == ""  # empty transcription: accumulated text stays empty
    assert new_text == ""
    assert state.accumulated_text == ""
    assert len(state.buffer) == 0  # buffer still resets even with empty text


def test_push_chunk_resets_silence_counter_on_new_speech():
    mock_model = MagicMock()
    streamer = AudioStreamer(mock_model)

    loud_chunk = np.full(1600, 0.5, dtype=np.float32)
    silent_chunk = np.zeros(1600, dtype=np.float32)

    state, _, _ = streamer.push_chunk(
        StreamState(), loud_chunk, chunk_duration_s=0.1, source_language="pt"
    )
    state, _, _ = streamer.push_chunk(
        state, silent_chunk, chunk_duration_s=0.1, source_language="pt"
    )
    assert state.silence_duration_s == 0.1

    # speech resumes: silence counter must reset to 0
    state, _, _ = streamer.push_chunk(
        state, loud_chunk, chunk_duration_s=0.1, source_language="pt"
    )
    assert state.silence_duration_s == 0.0


@patch("voice_translator.streaming.STREAM_SAMPLE_RATE", 100)
@patch("voice_translator.streaming.MIN_CHUNK_DURATION_S", 0.1)
@patch("voice_translator.streaming.SILENCE_DURATION_S", 0.05)
def test_push_chunk_accumulates_text_across_multiple_closed_chunks():
    mock_model = MagicMock()
    streamer = AudioStreamer(mock_model)

    loud_chunk = np.full(15, 0.5, dtype=np.float32)
    silent_chunk = np.zeros(5, dtype=np.float32)

    # first sentence
    mock_model.transcribe.return_value = ([_make_segment("hello")], MagicMock())
    state, _, _ = streamer.push_chunk(
        StreamState(), loud_chunk, chunk_duration_s=0.15, source_language="en"
    )
    state, accumulated, new_text = streamer.push_chunk(
        state, silent_chunk, chunk_duration_s=0.05, source_language="en"
    )
    assert accumulated == "hello"
    assert new_text == "hello"

    # second sentence, should append to the first
    mock_model.transcribe.return_value = ([_make_segment("world")], MagicMock())
    state, _, _ = streamer.push_chunk(
        state, loud_chunk, chunk_duration_s=0.15, source_language="en"
    )
    state, accumulated, new_text = streamer.push_chunk(
        state, silent_chunk, chunk_duration_s=0.05, source_language="en"
    )
    assert accumulated == "hello world"
    assert new_text == "world"  # only the newly transcribed piece, not the full text


@patch("voice_translator.streaming.STREAM_SAMPLE_RATE", 100)
@patch("voice_translator.streaming.MIN_CHUNK_DURATION_S", 0.1)
@patch("voice_translator.streaming.SILENCE_DURATION_S", 0.05)
def test_push_chunk_uses_source_language_per_call():
    """Different calls can use different languages (e.g. dropdown changed)."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([_make_segment("bonjour")], MagicMock())
    streamer = AudioStreamer(mock_model)

    loud_chunk = np.full(15, 0.5, dtype=np.float32)
    silent_chunk = np.zeros(5, dtype=np.float32)

    state, _, _ = streamer.push_chunk(
        StreamState(), loud_chunk, chunk_duration_s=0.15, source_language="fr"
    )
    streamer.push_chunk(
        state, silent_chunk, chunk_duration_s=0.05, source_language="fr"
    )

    call_kwargs = mock_model.transcribe.call_args.kwargs
    assert call_kwargs["language"] == "fr"


# ---------------------------------------------------------------------------
# flush
# ---------------------------------------------------------------------------


def test_flush_transcribes_remaining_buffer():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([_make_segment("trailing")], MagicMock())
    streamer = AudioStreamer(mock_model)

    state = StreamState(
        buffer=np.full(100, 0.5, dtype=np.float32),
        accumulated_text="hello",
    )

    new_state, accumulated, new_text = streamer.flush(state, source_language="pt")

    assert accumulated == "hello trailing"
    assert new_text == "trailing"
    assert new_state.accumulated_text == "hello trailing"
    assert len(new_state.buffer) == 0


def test_flush_with_empty_buffer_returns_unchanged_state():
    mock_model = MagicMock()
    streamer = AudioStreamer(mock_model)

    state = StreamState(accumulated_text="hello")
    new_state, accumulated, new_text = streamer.flush(state, source_language="pt")

    assert accumulated == "hello"
    assert new_text == ""
    mock_model.transcribe.assert_not_called()
