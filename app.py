"""
Voice Translator — Gradio application entry point.
Transcribes and optionally translates audio using Whisper and NLLB-200.
"""

import logging
import sys
from pathlib import Path

# Ensure the src/ package is importable regardless of how the app is launched
sys.path.insert(0, str(Path(__file__).parent / "src"))

import dataclasses

import gradio as gr
import numpy as np
from scipy.signal import resample_poly

from voice_translator.config import (
    APP_DESCRIPTION,
    APP_TITLE,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    STREAM_SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    WHISPER_LANGUAGE_CODES,
)
from voice_translator.pipeline import build_pipeline, run_pipeline
from voice_translator.streaming import AudioStreamer, StreamState
from voice_translator.translation import translate

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
# Streaming setup (once at startup, reuses the same ASR model)
# ---------------------------------------------------------------------------

streamer = AudioStreamer(model=pipeline_ctx["asr"])


# ---------------------------------------------------------------------------
# Gradio event handlers — batch mode
# ---------------------------------------------------------------------------


def on_language_change(source: str, target: str) -> dict:
    """
    Reload pipeline context when the user changes the language pair.

    Args:
        source: Display name of the source language (e.g. 'Português').
        target: Display name of the target language (e.g. 'English').

    Returns:
        A Gradio update re-enabling the submit button once the reload
        is complete.
    """
    global pipeline_ctx

    src_code = SUPPORTED_LANGUAGES[source]
    tgt_code = SUPPORTED_LANGUAGES[target]

    logger.info("Language pair changed to %s -> %s.", src_code, tgt_code)
    pipeline_ctx = build_pipeline(
        source_language=src_code,
        target_language=tgt_code,
    )
    logger.info("Pipeline reloaded successfully.")

    return gr.update(interactive=True)


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
# Gradio event handlers — streaming mode
# ---------------------------------------------------------------------------


def _translate_new_text(
    state: StreamState,
    new_text: str,
    source_language_name: str,
    target_language_name: str,
) -> str:
    logger.info(
        "_translate_new_text called: new_text=%r source=%s target=%s",
        new_text,
        source_language_name,
        target_language_name,
    )
    if not new_text:
        return state.accumulated_translation

    nllb_source = SUPPORTED_LANGUAGES[source_language_name]
    nllb_target = SUPPORTED_LANGUAGES[target_language_name]
    logger.info(
        "Translating with nllb_source=%s nllb_target=%s", nllb_source, nllb_target
    )

    new_translation = translate(
        pipeline_ctx["translation_model"],
        pipeline_ctx["translation_tokenizer"],
        new_text,
        source_language=nllb_source,
        target_language=nllb_target,
    )
    logger.info("Translation result: %r", new_translation)

    return f"{state.accumulated_translation} {new_translation}".strip()


def process_streaming_chunk(
    stream_state: StreamState | None,
    new_chunk: tuple[int, np.ndarray] | None,
    source_language_name: str,
    target_language_name: str,
) -> tuple[StreamState, str, str]:
    """
    ...
    """
    state = stream_state or StreamState()

    if new_chunk is None:
        return state, state.accumulated_text, state.accumulated_translation

    whisper_language = WHISPER_LANGUAGE_CODES[source_language_name]

    sample_rate, audio_array = new_chunk
    audio_float = audio_array.astype(np.float32) / 32768.0

    if sample_rate != STREAM_SAMPLE_RATE:
        audio_float = resample_poly(audio_float, STREAM_SAMPLE_RATE, sample_rate)

    chunk_duration_s = len(audio_float) / STREAM_SAMPLE_RATE
    updated_state, accumulated_text, new_text = streamer.push_chunk(
        state, audio_float, chunk_duration_s, source_language=whisper_language
    )

    accumulated_translation = _translate_new_text(
        state, new_text, source_language_name, target_language_name
    )
    updated_state = dataclasses.replace(
        updated_state, accumulated_translation=accumulated_translation
    )

    return updated_state, accumulated_text, accumulated_translation


def flush_streaming_buffer(
    stream_state: StreamState | None,
    source_language_name: str,
    target_language_name: str,
) -> tuple[StreamState, str, str]:
    """
    ...
    """
    state = stream_state or StreamState()
    whisper_language = WHISPER_LANGUAGE_CODES[source_language_name]
    updated_state, accumulated_text, new_text = streamer.flush(
        state, source_language=whisper_language
    )

    accumulated_translation = _translate_new_text(
        state, new_text, source_language_name, target_language_name
    )
    updated_state = dataclasses.replace(
        updated_state, accumulated_translation=accumulated_translation
    )

    return updated_state, accumulated_text, accumulated_translation


def clear_streaming_state() -> tuple[StreamState, str, str]:
    """Reset streaming state when the user starts a new recording session."""
    return StreamState(), "", ""


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

    # --- Real-time streaming section (top) ---

    gr.Markdown("## Real-time Streaming")

    stream_audio_input = gr.Audio(
        sources=["microphone"],
        streaming=True,
        type="numpy",
        label="Speak now",
    )

    with gr.Row():
        streaming_output = gr.Textbox(
            label="Live Transcription",
            placeholder="Transcription will appear here as you speak...",
            lines=4,
        )
        streaming_translation_output = gr.Textbox(
            label="Live Translation",
            placeholder="Translation will appear here as you speak...",
            lines=4,
        )

    stream_state = gr.State(value=StreamState())

    # --- Batch mode section (bottom) ---

    gr.Markdown("---")
    gr.Markdown("## Upload or Record Audio")

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

    # --- events: language dropdowns (disable submit while reloading) ---

    source_dropdown.change(
        fn=lambda: gr.update(interactive=False),
        inputs=[],
        outputs=[submit_btn],
    ).then(
        fn=on_language_change,
        inputs=[source_dropdown, target_dropdown],
        outputs=[submit_btn],
    )

    target_dropdown.change(
        fn=lambda: gr.update(interactive=False),
        inputs=[],
        outputs=[submit_btn],
    ).then(
        fn=on_language_change,
        inputs=[source_dropdown, target_dropdown],
        outputs=[submit_btn],
    )

    # --- events: streaming mode ---

    stream_audio_input.stream(
        fn=process_streaming_chunk,
        inputs=[stream_state, stream_audio_input, source_dropdown, target_dropdown],
        outputs=[stream_state, streaming_output, streaming_translation_output],
    )

    stream_audio_input.start_recording(
        fn=clear_streaming_state,
        inputs=[],
        outputs=[stream_state, streaming_output, streaming_translation_output],
    )

    stream_audio_input.stop_recording(
        fn=flush_streaming_buffer,
        inputs=[stream_state, source_dropdown, target_dropdown],
        outputs=[stream_state, streaming_output, streaming_translation_output],
    )

    # --- events: batch mode ---

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, translate_checkbox, source_dropdown],
        outputs=[transcription_output, translation_output],
    )

demo.launch(share=True)
