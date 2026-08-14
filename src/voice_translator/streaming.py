"""
Streaming module for Voice Translator.
Accumulates incoming audio chunks (e.g. from Gradio's streaming microphone
input) and decides when to close and transcribe them, based on silence
detection.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from faster_whisper import WhisperModel

from voice_translator.config import (
    MAX_CHUNK_DURATION_S,
    MIN_CHUNK_DURATION_S,
    SILENCE_DURATION_S,
    STREAM_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)

# RMS amplitude threshold below which a chunk is considered silence.
# Calibrated against real microphone data; may need tuning per device.
SILENCE_RMS_THRESHOLD = 0.02


@dataclass
class StreamState:
    """
    Holds the accumulated audio buffer, silence tracking, and transcribed
    text between consecutive calls to AudioStreamer.push_chunk(). Meant to
    be carried across calls (e.g. as Gradio's gr.State).
    """

    buffer: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    silence_duration_s: float = 0.0
    accumulated_text: str = ""


class AudioStreamer:
    """
    Accumulates audio chunks pushed from an external source (e.g. Gradio's
    streaming microphone component) and decides when a chunk contains
    enough speech, followed by enough silence, to close and transcribe it.

    This class does not capture audio itself — it only processes chunks
    handed to it via push_chunk(), so it works the same way whether the
    audio originates from a local microphone or a user's browser.
    """

    def __init__(self, model: WhisperModel, source_language: str | None = None):
        self.model = model
        self.source_language = source_language

    @staticmethod
    def _rms(chunk: np.ndarray) -> float:
        """Root-mean-square amplitude of an audio chunk."""
        return float(np.sqrt(np.mean(np.square(chunk))))

    def push_chunk(
        self, state: StreamState, chunk: np.ndarray, chunk_duration_s: float
    ) -> tuple[StreamState, str]:
        """
        Add a new audio chunk to the buffer and decide whether to close
        and transcribe it.

        Args:
            state: The accumulated state from the previous call.
            chunk: New audio samples (mono, float32, at STREAM_SAMPLE_RATE).
            chunk_duration_s: Duration of this chunk in seconds, used to
                track accumulated silence across calls.

        Returns:
            A tuple of (updated_state, accumulated_text). accumulated_text
            reflects everything transcribed so far in this session.
        """
        buffer = np.concatenate([state.buffer, chunk])

        is_silent = self._rms(chunk) < SILENCE_RMS_THRESHOLD
        silence_duration_s = (
            state.silence_duration_s + chunk_duration_s if is_silent else 0.0
        )

        min_samples = int(MIN_CHUNK_DURATION_S * STREAM_SAMPLE_RATE)
        max_samples = int(MAX_CHUNK_DURATION_S * STREAM_SAMPLE_RATE)

        enough_audio = len(buffer) >= min_samples
        silence_closed = silence_duration_s >= SILENCE_DURATION_S
        force_closed = len(buffer) >= max_samples

        if enough_audio and (silence_closed or force_closed):
            text = self._transcribe_chunk(buffer)
            accumulated_text = (
                f"{state.accumulated_text} {text}".strip()
                if text
                else state.accumulated_text
            )
            new_state = StreamState(accumulated_text=accumulated_text)
            return new_state, accumulated_text

        new_state = StreamState(
            buffer=buffer,
            silence_duration_s=silence_duration_s,
            accumulated_text=state.accumulated_text,
        )
        return new_state, state.accumulated_text

    def _transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """Run faster-whisper on a single accumulated audio chunk."""
        segments, _ = self.model.transcribe(
            audio_chunk,
            language=self.source_language,
            vad_filter=True,
        )
        segments_list = list(segments)  # consume the generator once, log it

        text = " ".join(segment.text.strip() for segment in segments_list).strip()
        return text

    def flush(self, state: StreamState) -> tuple[StreamState, str]:
        """
        Force-transcribe whatever audio remains in the buffer, regardless
        of silence or duration thresholds. Meant to be called when the
        user stops recording, so trailing speech isn't lost.

        Args:
            state: The current accumulated state.

        Returns:
            A tuple of (reset_state, accumulated_text).
        """
        if len(state.buffer) == 0:
            return state, state.accumulated_text

        text = self._transcribe_chunk(state.buffer)
        accumulated_text = (
            f"{state.accumulated_text} {text}".strip()
            if text
            else state.accumulated_text
        )
        new_state = StreamState(accumulated_text=accumulated_text)
        return new_state, accumulated_text
