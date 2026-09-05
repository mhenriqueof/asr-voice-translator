# **Voice Translator**
A real-time speech transcription and translation system that transcribes and translates spoken
language live, as you talk. Built with faster-whisper, NLLB-200 and Gradio.

[Try it on Hugging Face Spaces](https://huggingface.co/spaces/mhenriqueof/voice-translator)


## **Objective**
The goal of this project was to build a system that connects audio processing and
natural language processing, fields I have recently studied, to apply and consolidate the
knowledge I acquired.

Beyond the core pipeline, the project was also an opportunity to practice software engineering principles,
including code versioning, structured documentation, modular design, clean code practices and
real-time systems design (streaming, buffering, concurrency and state management).


## **Overview**
Instead of processing audio as a single static file, the system supports two modes:

* Batch transcription and translation of a full audio file
* Real-time streaming transcription and translation from the microphone

Real-time mode relies on:

* A CTranslate2-based Whisper reimplementation for fast CPU inference
* Voice activity detection (VAD) to decide when a spoken phrase is complete
* Incremental state carried across streaming calls, so transcription and translation grow
  together as the user speaks
* New phrases are transcribed and translated as they are spoken, with no need to wait for the
recording to end.


## **How It Works**
The application operates in two modes:

### 1. Streaming Mode (Real-time)
Live transcription and translation as the user speaks.

Pipeline:

```
Mic Chunk → Resample to 16kHz → Buffer → Silence Detected? → faster-whisper → NLLB-200 → Live Display
                                  ↑__________________________________|
                              (keeps accumulating until a pause is detected)
```

Features:

* Live transcription and translation displayed side by side
* Dynamic source/target language selection
* Trailing speech is flushed and transcribed when recording stops

### 2. Batch Mode
Upload or record a full audio file for one-shot transcription and translation.

Pipeline:

```
Audio File → faster-whisper → Full Transcription → NLLB-200 → Full Translation
```


## **Core Concepts**

### Speech Recognition
* **Model**: faster-whisper (`whisper-base`), a CTranslate2-based reimplementation of OpenAI's Whisper
* `int8` quantization on CPU for faster inference
* Built-in Silero VAD filters out silence within each transcribed chunk

### Translation
* **Model**: NLLB-200 (`nllb-200-distilled-600M`), Meta's 200-language translation model
* Source and target languages identified via BCP-47 codes (e.g. `por_Latn`, `eng_Latn`)
* Each closed phrase is translated independently, right after being transcribed

### Streaming State Management
Real-time mode requires carrying state across repeated calls, since each incoming audio chunk is
handled independently by the web interface:

* **Audio buffer**: accumulates incoming chunks until enough speech, followed by enough silence,
  has been detected
* **Silence tracking**: measured in accumulated seconds (not block counts), since incoming chunks
  can vary in duration
* **Accumulated transcription and translation**: preserved and extended across calls, so the
  displayed text grows phrase by phrase instead of resetting

### Silence Detection
Voice activity is estimated using **RMS (root-mean-square) amplitude** of each incoming audio chunk:

```
rms = sqrt(mean(chunk²))
```

A chunk is considered silent when its RMS falls below a calibrated threshold. A buffer closes and
gets transcribed once enough silence has accumulated after enough speech or once a maximum
duration safeguard is reached.


## **Usage**

### Controls
| Component | Mode | Action |
|-----------|------|--------|
| Source/Target Language dropdowns | Both | Select spoken and target language |
| Microphone (streaming) | Streaming | Speak to transcribe/translate live |
| Stop recording | Streaming | Flush remaining buffered audio |
| Audio upload/record | Batch | Provide a full audio file |
| Transcribe - Translate button | Batch | Run the pipeline on the provided audio |

### 1. Real-time Streaming
1. Select source and target language at the top of the page
2. Click the microphone and start speaking
3. Transcription and translation appear live, side by side
4.  Click stop to flush any remaining speech in the buffer

### 2. Batch Mode
1. Select source and target language at the top of the page
2. Upload or record an audio file
3. Click "Transcribe - Translate"


## **Installation**
### Requirements
- Python 3.11
- A microphone (for streaming mode)

### Clone the Repository
```bash
git clone https://github.com/mhenriqueof/voice-translator.git
cd voice-translator
```

### Create a Virtual Environment
```bash
python -m venv venv
source venv\Scripts\activate # on Linux/macOS: venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

On first run, the system automatically downloads the `faster-whisper` and NLLB-200 model weights.

### Run the App
```bash
python app.py
```

### Configuration
Inference device and streaming behavior can be tuned via environment variables and
`src/voice_translator/config.py`:

```bash
# Force a specific inference device (defaults to "cpu")
VOICE_TRANSLATOR_DEVICE=cuda python app.py
```


## **Known Limitations**
* Fragmented speech (very short phrases, or phrases cut mid-sentence by silence detection) can
  occasionally produce lower-quality translations, since each phrase is translated independently
  without broader conversational context.
* Silence detection relies on a fixed RMS threshold, which may need tuning for noisy environments
  or unusual microphone setups.
* Running on CPU means there's a small delay between speaking and seeing the transcription or
  translation, not instant, but close to real-time.


## **Project Structure**
```
voice-translator/
├── src/voice_translator/
│   ├── config.py        # Language mappings, model IDs, streaming tuning
│   ├── transcription.py # faster-whisper wrapper (batch mode)
│   ├── translation.py   # NLLB-200 wrapper
│   ├── streaming.py     # AudioStreamer: buffering, VAD and state for real-time mode
│   └── pipeline.py      # Ties transcription and translation together (batch mode)
├── tests/           # pytest suite
├── app.py           # Gradio application entry point
├── requirements.txt # Dependencies and libraries required for the project
├── README.md        # The main landing page and project overview
└── pyproject.toml   # Dependency manifest
```

---

This project connects speech recognition and machine translation into a single real-time pipeline.
It was challenging in ways I didn't expect, making me learn a lot throughout the development process.

Thanks!
