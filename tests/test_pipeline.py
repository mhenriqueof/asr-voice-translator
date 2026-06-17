"""
Tests for the pipeline module.
"""

from unittest.mock import MagicMock, patch

from voice_translator.pipeline import build_pipeline, run_pipeline

# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------


@patch("voice_translator.pipeline.load_transcriber")
@patch("voice_translator.pipeline.load_translator")
def test_build_pipeline_returns_dict(mock_load_translator, mock_load_transcriber):
    mock_load_transcriber.return_value = MagicMock()
    mock_load_translator.return_value = (MagicMock(), MagicMock())

    result = build_pipeline(source_language="pt", target_language="en")

    assert isinstance(result, dict)


@patch("voice_translator.pipeline.load_transcriber")
@patch("voice_translator.pipeline.load_translator")
def test_build_pipeline_contains_expected_keys(
    mock_load_translator, mock_load_transcriber
):
    mock_load_transcriber.return_value = MagicMock()
    mock_load_translator.return_value = (MagicMock(), MagicMock())

    result = build_pipeline(source_language="pt", target_language="en")

    expected_keys = {
        "asr",
        "translation_model",
        "translation_tokenizer",
        "source_language",
        "target_language",
    }
    assert expected_keys.issubset(result.keys())


@patch("voice_translator.pipeline.load_transcriber")
@patch("voice_translator.pipeline.load_translator")
def test_build_pipeline_stores_languages(mock_load_translator, mock_load_transcriber):
    mock_load_transcriber.return_value = MagicMock()
    mock_load_translator.return_value = (MagicMock(), MagicMock())

    result = build_pipeline(source_language="pt", target_language="en")

    assert result["source_language"] == "pt"
    assert result["target_language"] == "en"


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


@patch("voice_translator.pipeline.transcribe")
@patch("voice_translator.pipeline.translate")
def test_run_pipeline_returns_transcription(mock_translate, mock_transcribe):
    mock_transcribe.return_value = "Olá mundo"
    mock_translate.return_value = "Hello world"

    pipeline_ctx = {
        "asr": MagicMock(),
        "translation_model": MagicMock(),
        "translation_tokenizer": MagicMock(),
        "source_language": "pt",
        "target_language": "en",
    }

    result = run_pipeline(pipeline_ctx, "fake/path.wav")

    assert result["transcription"] == "Olá mundo"


@patch("voice_translator.pipeline.transcribe")
@patch("voice_translator.pipeline.translate")
def test_run_pipeline_with_translation(mock_translate, mock_transcribe):
    mock_transcribe.return_value = "Olá mundo"
    mock_translate.return_value = "Hello world"

    pipeline_ctx = {
        "asr": MagicMock(),
        "translation_model": MagicMock(),
        "translation_tokenizer": MagicMock(),
        "source_language": "pt",
        "target_language": "en",
    }

    result = run_pipeline(pipeline_ctx, "fake/path.wav", translate_audio=True)

    assert result["translation"] == "Hello world"


@patch("voice_translator.pipeline.transcribe")
@patch("voice_translator.pipeline.translate")
def test_run_pipeline_without_translation(mock_translate, mock_transcribe):
    mock_transcribe.return_value = "Olá mundo"

    pipeline_ctx = {
        "asr": MagicMock(),
        "translation_model": MagicMock(),
        "translation_tokenizer": MagicMock(),
        "source_language": "pt",
        "target_language": "en",
    }

    result = run_pipeline(pipeline_ctx, "fake/path.wav", translate_audio=False)

    assert "translation" not in result
    mock_translate.assert_not_called()
