"""
Tests for the streaming module.
"""

import threading
from unittest.mock import MagicMock, patch

import numpy as np

from voice_translator.streaming import AudioStreamer

# ---------------------------------------------------------------------------
# _rms
# ---------------------------------------------------------------------------


def test_rms_of_silence_is_zero():
    silence = np.zeros(1600, dtype=np.float32)
    assert AudioStreamer._rms(silence) == 0.0


def test_rms_of_constant_signal():
    # RMS of a constant signal equals its absolute amplitude
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


def test_transcribe_chunk_returns_empty_string_when_no_speech():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock())
    streamer = AudioStreamer(mock_model, source_language=None)

    result = streamer._transcribe_chunk(np.zeros(1600, dtype=np.float32))

    assert result == ""


# ---------------------------------------------------------------------------
# _accumulate_and_transcribe (worker loop, without real audio hardware)
# ---------------------------------------------------------------------------


@patch(
    "voice_translator.streaming.MIN_CHUNK_DURATION_S",
    0.2,
)
@patch(
    "voice_translator.streaming.SILENCE_DURATION_S",
    0.2,
)
@patch(
    "voice_translator.streaming.STREAM_SAMPLE_RATE",
    100,  # tiny sample rate to keep test blocks small and fast
)
def test_worker_closes_chunk_after_silence_and_emits_text():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([_make_segment("hi")], MagicMock())
    streamer = AudioStreamer(mock_model, source_language=None)

    block_duration_s = 0.05  # 5ms blocks, matches STREAM_SAMPLE_RATE=100
    loud_block = np.full(5, 0.5, dtype=np.float32)  # above silence threshold
    silent_block = np.zeros(5, dtype=np.float32)

    # feed one loud block (speech) followed by several silent blocks
    streamer._audio_queue.put(loud_block)
    for _ in range(6):
        streamer._audio_queue.put(silent_block)

    worker = threading.Thread(
        target=streamer._accumulate_and_transcribe,
        args=(block_duration_s,),
        daemon=True,
    )
    worker.start()

    try:
        text = streamer._text_queue.get(timeout=2.0)
    finally:
        streamer._stop_event.set()
        worker.join(timeout=1.0)

    assert text == "hi"
    mock_model.transcribe.assert_called_once()


@patch("voice_translator.streaming.MAX_CHUNK_DURATION_S", 0.2)
@patch("voice_translator.streaming.MIN_CHUNK_DURATION_S", 0.1)
@patch("voice_translator.streaming.SILENCE_DURATION_S", 999)  # never close via silence
@patch("voice_translator.streaming.STREAM_SAMPLE_RATE", 100)
def test_worker_force_closes_chunk_at_max_duration():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([_make_segment("forced")], MagicMock())
    streamer = AudioStreamer(mock_model, source_language=None)

    block_duration_s = 0.05
    loud_block = np.full(5, 0.5, dtype=np.float32)

    # keep feeding loud (non-silent) blocks; only max-duration should close it
    for _ in range(10):
        streamer._audio_queue.put(loud_block)

    worker = threading.Thread(
        target=streamer._accumulate_and_transcribe,
        args=(block_duration_s,),
        daemon=True,
    )
    worker.start()

    try:
        text = streamer._text_queue.get(timeout=2.0)
    finally:
        streamer._stop_event.set()
        worker.join(timeout=1.0)

    assert text == "forced"


@patch(
    "voice_translator.streaming.MIN_CHUNK_DURATION_S",
    0.2,
)
@patch(
    "voice_translator.streaming.SILENCE_DURATION_S",
    0.2,
)
@patch(
    "voice_translator.streaming.STREAM_SAMPLE_RATE",
    100,
)
def test_worker_does_not_emit_text_for_empty_transcription():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock())  # VAD removed everything
    streamer = AudioStreamer(mock_model, source_language=None)

    block_duration_s = 0.05
    loud_block = np.full(5, 0.5, dtype=np.float32)
    silent_block = np.zeros(5, dtype=np.float32)

    streamer._audio_queue.put(loud_block)
    for _ in range(6):
        streamer._audio_queue.put(silent_block)

    worker = threading.Thread(
        target=streamer._accumulate_and_transcribe,
        args=(block_duration_s,),
        daemon=True,
    )
    worker.start()
    streamer._stop_event.set()
    worker.join(timeout=1.0)

    assert streamer._text_queue.empty()
