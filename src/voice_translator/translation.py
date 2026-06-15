"""
Translation module for Voice Translator.
Handles text translation using Helsinki-NLP MarianMT models via Hugging Face Transformers.
"""

import logging

from transformers import MarianMTModel, MarianTokenizer

from voice_translator.config import TRANSLATION_MODEL_ID

logger = logging.getLogger(__name__)

def build_model_id(source_language: str, target_language: str) -> str:
    """
    Build the Hugging Face model identifier for a language pair.

    Args:
        source_language: ISO 639-1 source language code (e.g. 'pt').
        target_language: ISO 639-1 target language code (e.g. 'en').

    Returns:
        A Hugging Face model identifier string.
    """
    return TRANSLATION_MODEL_ID.format(src=source_language, tgt=target_language)

def load_translator(source_language: str, target_language: str) -> tuple:
    """
    Load and return a MarianMT model and tokenizer for the given language pair.

    Args:
        source_language: ISO 639-1 source language code.
        target_language: ISO 639-1 target language code.

    Returns:
        A tuple of (MarianMTModel, MarianTokenizer).
    """
    model_id = build_model_id(source_language, target_language)
    logger.info("Loading translation model '%s'.", model_id)

    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id)

    logger.info("Translation model loaded successfully.")
    return model, tokenizer

def translate(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    text: str,
) -> str:
    """
    Translate text using the provided MarianMT model and tokenizer.

    Args:
        model: A loaded MarianMT model.
        tokenizer: A loaded MarianMT tokenizer.
        text: Text to translate.

    Returns:
        Translated text as a string.
    """
    logger.info("Translating text (%d characters).", len(text))

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)
    translated_text: str = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    logger.info("Translation complete: %d characters.", len(translated_text))
    return translated_text
