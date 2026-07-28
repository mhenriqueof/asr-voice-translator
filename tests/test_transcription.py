"""
Tests for the transcription module.
"""

from unittest.mock import MagicMock, patch

from voice_translator.transcription import (
    get_compute_type,
    load_transcriber,
    transcribe,
)

# ---------------------------------------------------------------------------
# get_compute_type
# ---------------------------------------------------------------------------


@patch("voice_translator.transcription.DEVICE", "cpu")
def test_get_compute_type_returns_int8_on_cpu():
    assert get_compute_type() == "int8"


@patch("voice_translator.transcription.DEVICE", "cuda")
def test_get_compute_type_returns_float16_on_cuda():
    assert get_compute_type() == "float16"


# ---------------------------------------------------------------------------
# load_transcriber
# ---------------------------------------------------------------------------


@patch("voice_translator.transcription.WhisperModel")
def test_load_transcriber_returns_model(mock_whisper_model):
    mock_whisper_model.return_value = MagicMock()
    result = load_transcriber("base")
    assert result is mock_whisper_model.return_value


@patch("voice_translator.transcription.WhisperModel")
def test_load_transcriber_calls_whisper_model_with_size(mock_whisper_model):
    load_transcriber("base")
    call_args = mock_whisper_model.call_args
    assert call_args.args[0] == "base"


@patch("voice_translator.transcription.WhisperModel")
def test_load_transcriber_uses_configured_device(mock_whisper_model):
    load_transcriber("base")
    call_kwargs = mock_whisper_model.call_args.kwargs
    assert call_kwargs["device"] in ("cuda", "cpu")


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------


def _make_segment(text: str) -> MagicMock:
    segment = MagicMock()
    segment.text = text
    return segment


def test_transcribe_returns_joined_stripped_text():
    mock_model = MagicMock()
    mock_info = MagicMock(language="en", language_probability=0.99)
    mock_model.transcribe.return_value = (
        [_make_segment("  Hello  "), _make_segment("world  ")],
        mock_info,
    )

    result = transcribe(mock_model, "fake/path.wav")

    assert result == "Hello world"


def test_transcribe_passes_language_when_provided():
    mock_model = MagicMock()
    mock_info = MagicMock(language="pt", language_probability=0.95)
    mock_model.transcribe.return_value = ([_make_segment("Olá mundo")], mock_info)

    transcribe(mock_model, "fake/path.wav", source_language="pt")

    call_kwargs = mock_model.transcribe.call_args.kwargs
    assert call_kwargs["language"] == "pt"


def test_transcribe_omits_language_when_none():
    mock_model = MagicMock()
    mock_info = MagicMock(language="en", language_probability=0.9)
    mock_model.transcribe.return_value = ([_make_segment("Hello")], mock_info)

    transcribe(mock_model, "fake/path.wav", source_language=None)

    call_kwargs = mock_model.transcribe.call_args.kwargs
    assert call_kwargs["language"] is None
