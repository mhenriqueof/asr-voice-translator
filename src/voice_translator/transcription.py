"""
Transcription module for Voice Translator.
Handles audio transcription using faster-whisper (CTranslate2-based Whisper).
"""

import logging

from faster_whisper import WhisperModel

from voice_translator.config import DEVICE, WHISPER_MODEL_SIZE

logger = logging.getLogger(__name__)


def get_compute_type() -> str:
    """Return the best compute type for the configured device."""
    return "float16" if DEVICE == "cuda" else "int8"


def load_transcriber(model_size: str = WHISPER_MODEL_SIZE) -> WhisperModel:
    """
    Load and return a faster-whisper model.

    Args:
        model_size: Whisper model size identifier (e.g. 'base', 'small').

    Returns:
        A loaded WhisperModel instance.
    """
    compute_type = get_compute_type()
    logger.info(
        "Loading faster-whisper model '%s' on device '%s' (compute_type=%s).",
        model_size,
        DEVICE,
        compute_type,
    )

    model = WhisperModel(model_size, device=DEVICE, compute_type=compute_type)

    logger.info("faster-whisper model loaded successfully.")
    return model


def transcribe(
    model: WhisperModel,
    audio_path: str,
    source_language: str | None = None,
) -> str:
    """
    Transcribe an audio file using the provided faster-whisper model.

    Args:
        model: A loaded WhisperModel instance.
        audio_path: Path to the audio file (wav, mp3, etc.).
        source_language: ISO 639-1 language code (e.g. 'pt', 'en').
                         If None, Whisper auto-detects the language.

    Returns:
        Transcribed text as a string.
    """
    logger.info(
        "Transcribing file '%s' (language=%s).", audio_path, source_language or "auto"
    )

    segments, info = model.transcribe(audio_path, language=source_language)

    # segments is a generator — must be consumed to actually run inference
    transcription = " ".join(segment.text.strip() for segment in segments).strip()

    logger.info(
        "Transcription complete: %d characters "
        "(detected language=%s, probability=%.2f).",
        len(transcription),
        info.language,
        info.language_probability,
    )
    return transcription
