"""
Configuration module for Voice Translator.
Centralizes constants and settings used across the application.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------

WHISPER_MODEL_ID = "openai/whisper-base"

SUPPORTED_LANGUAGES: dict[str, str] = {
    "Português": "pt",
    "English": "en",
    "Español": "es",
    "Français": "fr",
    "Deutsch": "de",
    "Italiano": "it",
    "日本語": "ja",
    "中文": "zh",
}

DEFAULT_SOURCE_LANGUAGE = "pt"
DEFAULT_TARGET_LANGUAGE = "en"

TRANSLATION_MODEL_ID = "Helsinki-NLP/opus-mt-{src}-{tgt}"


# ---------------------------------------------------------------------------
# Audio settings
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000  # Hz — required by Whisper


# ---------------------------------------------------------------------------
# App settings
# ---------------------------------------------------------------------------

APP_TITLE = "🎙️ Voice Translator"
APP_DESCRIPTION = "Transcribe and translate audio in real time using Whisper."
