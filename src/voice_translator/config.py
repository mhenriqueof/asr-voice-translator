"""
Configuration module for Voice Translator.
Centralizes constants and settings used across the application.
"""

import os

# ---------------------------------------------------------------------
# Device settings
# ---------------------------------------------------------------------

DEVICE = os.getenv("VOICE_TRANSLATOR_DEVICE", "cpu")  # "cpu" or "cuda"


# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------

WHISPER_MODEL_SIZE = "base"

NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"

# NLLB uses BCP-47 language codes with script suffix
SUPPORTED_LANGUAGES: dict[str, str] = {
    "Português": "por_Latn",
    "English": "eng_Latn",
    "Español": "spa_Latn",
    "Français": "fra_Latn",
    "Deutsch": "deu_Latn",
    "Italiano": "ita_Latn",
    "日本語": "jpn_Jpan",
    "中文": "zho_Hans",
}

# Whisper uses standard ISO 639-1 codes (separate from NLLB codes)
WHISPER_LANGUAGE_CODES: dict[str, str] = {
    "Português": "pt",
    "English": "en",
    "Español": "es",
    "Français": "fr",
    "Deutsch": "de",
    "Italiano": "it",
    "日本語": "ja",
    "中文": "zh",
}

DEFAULT_SOURCE_LANGUAGE = "por_Latn"
DEFAULT_TARGET_LANGUAGE = "eng_Latn"
DEFAULT_WHISPER_LANGUAGE = "pt"


# ---------------------------------------------------------------------
# Streaming settings
# ---------------------------------------------------------------------

STREAM_SAMPLE_RATE = 16_000  # Hz, must match Whisper's expected input
SILENCE_DURATION_S = 0.5  # seconds of silence to close a chunk
MIN_CHUNK_DURATION_S = 1.0  # minimum audio length before transcribing
MAX_CHUNK_DURATION_S = 8.0  # force-close a chunk after this long


# ---------------------------------------------------------------------------
# Audio settings
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000  # Hz — required by Whisper


# ---------------------------------------------------------------------------
# App settings
# ---------------------------------------------------------------------------

APP_TITLE = "🎙️ Voice Translator"
APP_DESCRIPTION = (
    "Transcribe and translate audio in real time using Whisper and NLLB-200."
)
