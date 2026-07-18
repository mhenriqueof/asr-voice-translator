"""
Voice Translator — Gradio application entry point.
Transcribes and optionally translates audio using Whisper and NLLB-200.
"""

import logging
import sys
from pathlib import Path

# Ensure the src/ package is importable regardless of how the app is launched
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr

from voice_translator.config import (
    APP_DESCRIPTION,
    APP_TITLE,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    SUPPORTED_LANGUAGES,
    WHISPER_LANGUAGE_CODES,
)
from voice_translator.pipeline import build_pipeline, run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------

logger.info("Loading models at startup...")
pipeline_ctx = build_pipeline(
    source_language=DEFAULT_SOURCE_LANGUAGE,
    target_language=DEFAULT_TARGET_LANGUAGE,
)
logger.info("Models loaded successfully.")


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------


def on_language_change(source: str, target: str) -> None:
    """
    Reload pipeline context when the user changes the language pair.

    Args:
        source: Display name of the source language (e.g. 'Português').
        target: Display name of the target language (e.g. 'English').
    """
    global pipeline_ctx

    src_code = SUPPORTED_LANGUAGES[source]
    tgt_code = SUPPORTED_LANGUAGES[target]

    logger.info("Language pair changed to %s -> %s.", src_code, tgt_code)
    pipeline_ctx = build_pipeline(
        source_language=src_code,
        target_language=tgt_code,
    )


def process_audio(
    audio_path: str,
    translate_audio: bool,
    source_language_name: str,
) -> tuple[str, str]:
    """
    Transcribe and optionally translate the provided audio file.

    Args:
        audio_path: Path to the uploaded or recorded audio file.
        translate_audio: Whether to translate the transcribed text.
        source_language_name: Display name of the source language (e.g. 'Português').

    Returns:
        A tuple of (transcription, translation). Translation is empty string
        if translate_audio is False.
    """
    if audio_path is None:
        return "No audio provided.", ""

    whisper_language = WHISPER_LANGUAGE_CODES[source_language_name]

    result = run_pipeline(
        pipeline_ctx,
        audio_path,
        translate_audio=translate_audio,
        whisper_language=whisper_language,
    )

    transcription = result.get("transcription", "")
    translation = result.get("translation", "")

    return transcription, translation


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

language_names = list(SUPPORTED_LANGUAGES.keys())

with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(APP_DESCRIPTION)

    with gr.Row():
        source_dropdown = gr.Dropdown(
            choices=language_names,
            value="Português",
            label="Source Language",
        )
        target_dropdown = gr.Dropdown(
            choices=language_names,
            value="English",
            label="Target Language",
        )

    audio_input = gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        label="Audio Input",
    )

    translate_checkbox = gr.Checkbox(
        value=True,
        label="Translate",
    )

    submit_btn = gr.Button("Transcribe / Translate", variant="primary")

    with gr.Row():
        transcription_output = gr.Textbox(
            label="Transcription",
            placeholder="Transcription will appear here...",
            lines=4,
        )
        translation_output = gr.Textbox(
            label="Translation",
            placeholder="Translation will appear here...",
            lines=4,
        )

    # --- events ---

    source_dropdown.change(
        fn=on_language_change,
        inputs=[source_dropdown, target_dropdown],
        outputs=[],
    )

    target_dropdown.change(
        fn=on_language_change,
        inputs=[source_dropdown, target_dropdown],
        outputs=[],
    )

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, translate_checkbox, source_dropdown],
        outputs=[transcription_output, translation_output],
    )

demo.launch(share=True)
