"""
Tests for the translation module.
"""

from unittest.mock import MagicMock, patch

from voice_translator.translation import load_translator, translate

# ---------------------------------------------------------------------------
# load_translator
# ---------------------------------------------------------------------------


@patch("voice_translator.translation.AutoModelForSeq2SeqLM.from_pretrained")
@patch("voice_translator.translation.AutoTokenizer.from_pretrained")
def test_load_translator_returns_tuple(mock_tokenizer, mock_model):
    mock_model.return_value = MagicMock()
    mock_tokenizer.return_value = MagicMock()

    result = load_translator()

    assert isinstance(result, tuple)
    assert len(result) == 2


@patch("voice_translator.translation.AutoModelForSeq2SeqLM.from_pretrained")
@patch("voice_translator.translation.AutoTokenizer.from_pretrained")
def test_load_translator_calls_correct_model_id(mock_tokenizer, mock_model):
    from voice_translator.config import NLLB_MODEL_ID

    load_translator()

    mock_model.assert_called_once_with(NLLB_MODEL_ID)
    mock_tokenizer.assert_called_once_with(NLLB_MODEL_ID)


# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------


def test_translate_returns_string():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_tokenizer.return_value = {"input_ids": MagicMock()}
    mock_tokenizer.convert_tokens_to_ids.return_value = 1234
    mock_model.generate.return_value = [MagicMock()]
    mock_tokenizer.decode.return_value = "Hello world"

    result = translate(mock_model, mock_tokenizer, "Olá mundo", "por_Latn", "eng_Latn")

    assert isinstance(result, str)
    assert result == "Hello world"


def test_translate_calls_decode_with_skip_special_tokens():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_tokenizer.return_value = {"input_ids": MagicMock()}
    mock_tokenizer.convert_tokens_to_ids.return_value = 1234
    generated = MagicMock()
    mock_model.generate.return_value = [generated]
    mock_tokenizer.decode.return_value = "Hello"

    translate(mock_model, mock_tokenizer, "Olá", "por_Latn", "eng_Latn")

    mock_tokenizer.decode.assert_called_once_with(generated, skip_special_tokens=True)


def test_translate_calls_convert_tokens_to_ids_with_target_language():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_tokenizer.return_value = {"input_ids": MagicMock()}
    mock_tokenizer.convert_tokens_to_ids.return_value = 1234
    mock_model.generate.return_value = [MagicMock()]
    mock_tokenizer.decode.return_value = "Hello"

    translate(mock_model, mock_tokenizer, "Olá", "por_Latn", "eng_Latn")

    mock_tokenizer.convert_tokens_to_ids.assert_called_once_with("eng_Latn")


def test_translate_empty_string():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_tokenizer.return_value = {"input_ids": MagicMock()}
    mock_tokenizer.convert_tokens_to_ids.return_value = 1234
    mock_model.generate.return_value = [MagicMock()]
    mock_tokenizer.decode.return_value = ""

    result = translate(mock_model, mock_tokenizer, "", "por_Latn", "eng_Latn")

    assert result == ""
