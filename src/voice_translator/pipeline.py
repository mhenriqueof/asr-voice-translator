"""
Pipeline module for Voice Translator.
Orchestrates transcription and translation into a single cohesive workflow.
"""

import logging

from voice_translator.config import DEFAULT_SOURCE_LANGUAGE, DEFAULT_TARGET_LANGUAGE
from voice_translator.transcription import load_transcriber, transcribe
from voice_translator.translation import load_translator, translate

logger = logging.getLogger(__name__)


def build_pipeline(
    whisper_model_size: str | None = None,
    source_language: str = DEFAULT_SOURCE_LANGUAGE,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
) -> dict:
    """
    Load all models and return a pipeline context.

    Args:
        whisper_model_size: faster-whisper model size identifier (e.g. 'base', 'small').
                            If None, uses the default from config.
        source_language: BCP-47 source language code (e.g. 'por_Latn').
        target_language: BCP-47 target language code (e.g. 'eng_Latn').

    Returns:
        A dict containing loaded models and configuration.
    """
    logger.info(
        "Building pipeline (source=%s, target=%s).",
        source_language,
        target_language,
    )

    asr = (
        load_transcriber(whisper_model_size)
        if whisper_model_size
        else load_transcriber()
    )
    translation_model, translation_tokenizer = load_translator()

    return {
        "asr": asr,
        "translation_model": translation_model,
        "translation_tokenizer": translation_tokenizer,
        "source_language": source_language,
        "target_language": target_language,
    }


def run_pipeline(
    pipeline_ctx: dict,
    audio_path: str,
    translate_audio: bool = True,
    whisper_language: str | None = None,
) -> dict:
    """
    Run the full transcription and (optionally) translation pipeline.

    Args:
        pipeline_ctx: A pipeline context returned by build_pipeline.
        audio_path: Path to the audio file to process.
        translate_audio: Whether to translate the transcribed text.
        whisper_language: ISO 639-1 language code for Whisper (e.g. 'pt').
                          If None, Whisper auto-detects the language.

    Returns:
        A dict with 'transcription' and (optionally) 'translation' keys.
    """
    logger.info("Running pipeline on '%s'.", audio_path)

    transcription = transcribe(
        pipeline_ctx["asr"],
        audio_path,
        source_language=whisper_language,
    )

    result = {"transcription": transcription}

    if translate_audio:
        translation = translate(
            pipeline_ctx["translation_model"],
            pipeline_ctx["translation_tokenizer"],
            transcription,
            source_language=pipeline_ctx["source_language"],
            target_language=pipeline_ctx["target_language"],
        )
        result["translation"] = translation

    logger.info("Pipeline complete.")
    return result
