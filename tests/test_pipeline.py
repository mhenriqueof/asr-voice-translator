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

    result = build_pipeline(source_language="por_Latn", target_language="eng_Latn")

    assert isinstance(result, dict)


@patch("voice_translator.pipeline.load_transcriber")
@patch("voice_translator.pipeline.load_translator")
def test_build_pipeline_contains_expected_keys(
    mock_load_translator, mock_load_transcriber
):
    mock_load_transcriber.return_value = MagicMock()
    mock_load_translator.return_value = (MagicMock(), MagicMock())

    result = build_pipeline(source_language="por_Latn", target_language="eng_Latn")

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

    result = build_pipeline(source_language="por_Latn", target_language="eng_Latn")

    assert result["source_language"] == "por_Latn"
    assert result["target_language"] == "eng_Latn"


@patch("voice_translator.pipeline.load_transcriber")
@patch("voice_translator.pipeline.load_translator")
def test_build_pipeline_calls_load_translator_once(
    mock_load_translator, mock_load_transcriber
):
    mock_load_transcriber.return_value = MagicMock()
    mock_load_translator.return_value = (MagicMock(), MagicMock())

    build_pipeline(source_language="por_Latn", target_language="eng_Latn")

    mock_load_translator.assert_called_once()


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
        "source_language": "por_Latn",
        "target_language": "eng_Latn",
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
        "source_language": "por_Latn",
        "target_language": "eng_Latn",
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
        "source_language": "por_Latn",
        "target_language": "eng_Latn",
    }

    result = run_pipeline(pipeline_ctx, "fake/path.wav", translate_audio=False)

    assert "translation" not in result
    mock_translate.assert_not_called()


@patch("voice_translator.pipeline.transcribe")
@patch("voice_translator.pipeline.translate")
def test_run_pipeline_passes_whisper_language(mock_translate, mock_transcribe):
    mock_transcribe.return_value = "Olá mundo"
    mock_translate.return_value = "Hello world"

    pipeline_ctx = {
        "asr": MagicMock(),
        "translation_model": MagicMock(),
        "translation_tokenizer": MagicMock(),
        "source_language": "por_Latn",
        "target_language": "eng_Latn",
    }

    run_pipeline(pipeline_ctx, "fake/path.wav", whisper_language="pt")

    _, call_kwargs = mock_transcribe.call_args
    assert call_kwargs["source_language"] == "pt"
