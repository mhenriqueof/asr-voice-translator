"""
Tests for the transcription module.
"""

from unittest.mock import MagicMock, patch

import pytest

from voice_translator.transcription import get_device, load_transcriber, transcribe


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

def test_get_device_returns_string():
    device = get_device()
    assert isinstance(device, str)


def test_get_device_valid_value():
    device = get_device()
    assert device in ("cuda", "cpu")


# ---------------------------------------------------------------------------
# load_transcriber
# ---------------------------------------------------------------------------

@patch("voice_translator.transcription.AutoModelForSpeechSeq2Seq.from_pretrained")
@patch("voice_translator.transcription.AutoProcessor.from_pretrained")
@patch("voice_translator.transcription.pipeline")
def test_load_transcriber_returns_pipeline(
    mock_pipeline, mock_processor, mock_model
):
    mock_pipeline.return_value = MagicMock()
    result = load_transcriber("openai/whisper-base")
    assert result is mock_pipeline.return_value


@patch("voice_translator.transcription.AutoModelForSpeechSeq2Seq.from_pretrained")
@patch("voice_translator.transcription.AutoProcessor.from_pretrained")
@patch("voice_translator.transcription.pipeline")
def test_load_transcriber_calls_pipeline_with_asr_task(
    mock_pipeline, mock_processor, mock_model
):
    load_transcriber("openai/whisper-base")
    call_kwargs = mock_pipeline.call_args
    assert call_kwargs.kwargs["task"] == "automatic-speech-recognition"


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------

def test_transcribe_returns_stripped_text():
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = {"text": "  Hello world  "}

    result = transcribe(mock_pipeline, "fake/path.wav")

    assert result == "Hello world"


def test_transcribe_passes_language_when_provided():
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = {"text": "Olá mundo"}

    transcribe(mock_pipeline, "fake/path.wav", source_language="pt")

    _, call_kwargs = mock_pipeline.call_args
    assert call_kwargs["generate_kwargs"]["language"] == "pt"


def test_transcribe_omits_language_when_none():
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = {"text": "Hello"}

    transcribe(mock_pipeline, "fake/path.wav", source_language=None)

    _, call_kwargs = mock_pipeline.call_args
    assert call_kwargs["generate_kwargs"] == {}
    