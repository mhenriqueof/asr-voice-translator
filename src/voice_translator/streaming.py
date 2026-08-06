"""
Streaming module for Voice Translator.
Captures microphone audio in real time and emits transcriptions incrementally.
"""

import logging
import queue
import threading

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

# RMS amplitude threshold below which a block is considered silence.
# Calibrated for typical mic input levels; may need tuning per device.
SILENCE_RMS_THRESHOLD = 0.02


class AudioStreamer:
    """
    Captures microphone audio and yields transcribed text incrementally.

    Audio capture and transcription run on separate threads: the audio
    callback only ever pushes raw blocks onto a queue, while a dedicated
    worker thread accumulates blocks into chunks and runs faster-whisper
    on them. This prevents slow inference from blocking capture and
    causing dropped audio (input overflow).
    """

    def __init__(self, model: WhisperModel, source_language: str | None = None):
        self.model = model
        self.source_language = source_language
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._text_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback: pushes incoming audio blocks onto the queue.

        Must be fast and non-blocking — no transcription happens here.
        """
        if status:
            logger.warning("Audio input status: %s", status)
        self._audio_queue.put(indata[:, 0].copy())

    @staticmethod
    def _rms(block: np.ndarray) -> float:
        """Root-mean-square amplitude of an audio block."""
        return float(np.sqrt(np.mean(np.square(block))))

    def _accumulate_and_transcribe(self, block_duration_s: float):
        """
        Worker loop: consumes audio blocks from the queue, accumulates
        them into chunks based on silence detection, and transcribes
        each closed chunk. Runs on its own thread.
        """
        buffer = np.empty((0,), dtype=np.float32)
        silence_blocks = 0
        silence_blocks_threshold = int(SILENCE_DURATION_S / block_duration_s)
        min_samples = int(MIN_CHUNK_DURATION_S * STREAM_SAMPLE_RATE)
        max_samples = int(MAX_CHUNK_DURATION_S * STREAM_SAMPLE_RATE)

        while not self._stop_event.is_set():
            try:
                block = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            buffer = np.concatenate([buffer, block])

            is_silent = self._rms(block) < SILENCE_RMS_THRESHOLD
            silence_blocks = silence_blocks + 1 if is_silent else 0

            enough_audio = len(buffer) >= min_samples
            silence_closed = silence_blocks >= silence_blocks_threshold
            force_closed = len(buffer) >= max_samples

            if enough_audio and (silence_closed or force_closed):
                chunk, buffer = buffer, np.empty((0,), dtype=np.float32)
                silence_blocks = 0
                text = self._transcribe_chunk(chunk)
                if text:
                    self._text_queue.put(text)

    def _transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """Run faster-whisper on a single accumulated audio chunk."""
        segments, _ = self.model.transcribe(
            audio_chunk,
            language=self.source_language,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text

    def listen(self):
        """
        Start capturing microphone audio and yield transcribed text chunks
        as they become available.

        Yields:
            Transcribed text for each closed audio chunk that contained speech.
        """
        block_duration_s = 0.1
        logger.info("Starting audio stream (sample_rate=%d).", STREAM_SAMPLE_RATE)

        worker = threading.Thread(
            target=self._accumulate_and_transcribe,
            args=(block_duration_s,),
            daemon=True,
        )

        self._stop_event.clear()
        worker.start()

        try:
            with sd.InputStream(
                samplerate=STREAM_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=int(STREAM_SAMPLE_RATE * block_duration_s),
                callback=self._audio_callback,
            ):
                while True:
                    try:
                        yield self._text_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
        finally:
            self._stop_event.set()
            worker.join(timeout=2.0)
