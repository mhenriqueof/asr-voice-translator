"""
Translation module for Voice Translator.
Handles text translation using Meta's NLLB-200 model
via Hugging Face Transformers.
"""

import logging

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from voice_translator.config import NLLB_MODEL_ID

logger = logging.getLogger(__name__)


def load_translator() -> tuple:
    """
    Load and return the NLLB-200 model and tokenizer.

    Returns:
        A tuple of (AutoModelForSeq2SeqLM, AutoTokenizer).
    """
    logger.info("Loading NLLB-200 translation model '%s'.", NLLB_MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID)

    logger.info("Translation model loaded successfully.")
    return model, tokenizer


def translate(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    text: str,
    source_language: str,
    target_language: str,
) -> str:
    """
    Translate text using the NLLB-200 model.

    Args:
        model: A loaded NLLB-200 model.
        tokenizer: A loaded NLLB-200 tokenizer.
        text: Text to translate.
        source_language: BCP-47 source language code (e.g. 'por_Latn').
        target_language: BCP-47 target language code (e.g. 'eng_Latn').

    Returns:
        Translated text as a string.
    """
    logger.info(
        "Translating text (%d characters) from '%s' to '%s'.",
        len(text),
        source_language,
        target_language,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        src_lang=source_language,
    )

    target_lang_id = tokenizer.convert_tokens_to_ids(target_language)

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=target_lang_id,
    )

    translated_text: str = tokenizer.decode(
        translated_tokens[0],
        skip_special_tokens=True,
    )

    logger.info("Translation complete: %d characters.", len(translated_text))
    return translated_text
