"""
Streaming module for Voice Translator.
Captures microphone audio in real time and emits transcriptions incrementally.
"""

import logging
import queue

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from voice_translator.config import (
    MAX_CHUNK_DURATION_S,
    MIN_CHUNK_DURATION_S,
    SILENCE_DURATION_S,
    STREAM_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


class AudioStreamer:
    """
    Captures microphone audio and yields transcribed text incrementally.

    Uses faster-whisper's built-in VAD to decide when a chunk of audio contains enough
    speech to close and transcribe, instead of relying on a fixed-size window.
    """

    def __init__(self, model: WhisperModel, source_language: str | None = None):
        self.model = model
        self.source_language = source_language
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback: pushes incoming audio blocks onto the queue."""
        if status:
            logger.warning("Audio input status: %s", status)
        # indata is float32, shape (frames, channels); we want mono
        self._audio_queue.put(indata[:, 0].copy())

    def listen(self):
        """
        Start capturing microphone audio and yield transcribed text chunks
        as they become available.

        Yields:
            Transcribed text for each closed audio chunk.
        """
        buffer = np.empty((0,), dtype=np.float32)
        silence_blocks = 0
        block_duration_s = 0.1  # matches the blocksize below
        silence_blocks_threshold = int(SILENCE_DURATION_S / block_duration_s)
        min_samples = int(MIN_CHUNK_DURATION_S * STREAM_SAMPLE_RATE)
        max_samples = int(MAX_CHUNK_DURATION_S * STREAM_SAMPLE_RATE)

        logger.info("Starting audio stream (sample_rate=%d).", STREAM_SAMPLE_RATE)

        with sd.InputStream(
            samplerate=STREAM_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(STREAM_SAMPLE_RATE * block_duration_s),
            callback=self._audio_callback,
        ):
            while True:
                block = self._audio_queue.get()
                buffer = np.concatenate([buffer, block])

                is_silent = np.abs(block).mean() < 0.01
                silence_blocks = silence_blocks + 1 if is_silent else 0

                enough_audio = len(buffer) >= min_samples
                silence_closed = silence_blocks >= silence_blocks_threshold
                force_closed = len(buffer) >= max_samples

                if enough_audio and (silence_closed or force_closed):
                    text = self._transcribe_chunk(buffer)
                    if text:
                        yield text
                    buffer = np.empty((0,), dtype=np.float32)
                    silence_blocks = 0

    def _transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """Run faster-whisper on a single accumulated audio chunk."""
        segments, _ = self.model.transcribe(
            audio_chunk,
            language=self.source_language,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text
