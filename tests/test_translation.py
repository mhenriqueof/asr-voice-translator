"""
Tests for the translation module.
"""

from unittest.mock import MagicMock, patch

import pytest

from voice_translator.translation import build_model_id, load_translator, translate


# ---------------------------------------------------------------------------
# build_model_id
# ---------------------------------------------------------------------------

def test_build_model_id_format():
    result = build_model_id("pt", "en")
    assert result == "Helsinki-NLP/opus-mt-pt-en"


def test_build_model_id_different_pair():
    result = build_model_id("en", "fr")
    assert result == "Helsinki-NLP/opus-mt-en-fr"


# ---------------------------------------------------------------------------
# load_translator
# ---------------------------------------------------------------------------

@patch("voice_translator.translation.MarianMTModel.from_pretrained")
@patch("voice_translator.translation.MarianTokenizer.from_pretrained")
def test_load_translator_returns_tuple(mock_tokenizer, mock_model):
    mock_model.return_value = MagicMock()
    mock_tokenizer.return_value = MagicMock()

    result = load_translator("pt", "en")

    assert isinstance(result, tuple)
    assert len(result) == 2


@patch("voice_translator.translation.MarianMTModel.from_pretrained")
@patch("voice_translator.translation.MarianTokenizer.from_pretrained")
def test_load_translator_calls_correct_model_id(mock_tokenizer, mock_model):
    load_translator("pt", "en")

    expected_model_id = "Helsinki-NLP/opus-mt-pt-en"
    mock_model.assert_called_once_with(expected_model_id)
    mock_tokenizer.assert_called_once_with(expected_model_id)


# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------

def test_translate_returns_string():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_tokenizer.return_value = {"input_ids": MagicMock()}
    mock_model.generate.return_value = [MagicMock()]
    mock_tokenizer.decode.return_value = "Hello world"

    result = translate(mock_model, mock_tokenizer, "Olá mundo")

    assert isinstance(result, str)
    assert result == "Hello world"


def test_translate_calls_decode_with_skip_special_tokens():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_tokenizer.return_value = {"input_ids": MagicMock()}
    generated = MagicMock()
    mock_model.generate.return_value = [generated]
    mock_tokenizer.decode.return_value = "Hello"

    translate(mock_model, mock_tokenizer, "Olá")

    mock_tokenizer.decode.assert_called_once_with(generated, skip_special_tokens=True)


def test_translate_empty_string():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_tokenizer.return_value = {"input_ids": MagicMock()}
    mock_model.generate.return_value = [MagicMock()]
    mock_tokenizer.decode.return_value = ""

    result = translate(mock_model, mock_tokenizer, "")

    assert result == ""
    