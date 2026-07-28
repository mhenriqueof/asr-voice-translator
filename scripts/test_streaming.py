"""
Standalone script to manually test real-time streaming transcription.
Run this from the terminal and speak into your microphone.
"""

import logging

from voice_translator.config import DEFAULT_WHISPER_LANGUAGE
from voice_translator.streaming import AudioStreamer
from voice_translator.transcription import load_transcriber

logging.basicConfig(level=logging.INFO)


def main():
    print("Loading model...")
    model = load_transcriber()

    streamer = AudioStreamer(model, source_language=DEFAULT_WHISPER_LANGUAGE)

    print("Listening... speak into your microphone (Ctrl+C to stop).")
    try:
        for text in streamer.listen():
            print(f">> {text}")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
