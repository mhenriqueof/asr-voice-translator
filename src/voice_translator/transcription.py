"""
Transcription module for Voice Translator.
Handles audio transcription using OpenAI's Whisper model via Hugging Face Transformers.
"""

import logging

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from voice_translator.config import DEVICE, WHISPER_MODEL_ID

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Return the best available device for inference."""
    return DEVICE


def load_transcriber(model_id: str = WHISPER_MODEL_ID) -> pipeline:
    """
    Load and return a Whisper ASR pipeline.

    Args:
        model_id: Hugging Face model identifier.

    Returns:
        A Hugging Face pipeline configured for automatic speech recognition.
    """
    device = get_device()
    logger.info("Loading Whisper model '%s' on device '%s'.", model_id, device)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    asr_pipeline = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
    )

    logger.info("Whisper model loaded successfully.")
    return asr_pipeline


def transcribe(
    asr_pipeline: pipeline,
    audio_path: str,
    source_language: str | None = None,
) -> str:
    """
    Transcribe an audio file using the provided ASR pipeline.

    Args:
        asr_pipeline: A loaded Hugging Face ASR pipeline.
        audio_path: Path to the audio file (wav, mp3, etc.).
        source_language: ISO 639-1 language code (e.g. 'pt', 'en').
                         If None, Whisper auto-detects the language.

    Returns:
        Transcribed text as a string.
    """
    logger.info(
        "Transcribing file '%s' (language=%s).", audio_path, source_language or "auto"
    )

    generate_kwargs = {}
    if source_language:
        generate_kwargs["language"] = source_language

    result = asr_pipeline(audio_path, generate_kwargs=generate_kwargs)

    transcription: str = result["text"].strip()
    logger.info("Transcription complete: %d characters.", len(transcription))
    return transcription
