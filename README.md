---
title: Voice Translator
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# 🎙️ Voice Translator

Transcribe and translate audio using [Whisper](https://huggingface.co/openai/whisper-base) and [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M).

## Supported Languages

| Language | Transcription | Translation |
|----------|--------------|-------------|
| Português | ✅ | ✅ |
| English | ✅ | ✅ |
| Español | ✅ | ✅ |
| Français | ✅ | ✅ |
| Deutsch | ✅ | ✅ |
| Italiano | ✅ | ✅ |
| 日本語 | ✅ | ✅ |
| 中文 | ✅ | ✅ |

## How to Use

1. Select the source and target languages
2. Record your voice or upload an audio file
3. Click **Transcribe / Translate**

> ⚠️ Running on CPU — processing may take a few seconds depending on audio length.

## Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/mhenriqueof/voice-translator.git
cd voice-translator

# 2. Install system dependency
sudo apt install ffmpeg

# 3. Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run
python app.py
```

## Models

- **ASR**: [openai/whisper-base](https://huggingface.co/openai/whisper-base)
- **Translation**: [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)