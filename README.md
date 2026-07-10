---
title: AI Voice Recovery Agent
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 🎤 AI Voice Recovery Agent

Convert unclear, heavy, low-pitch, noisy or impaired speech into a clear,
natural **English** voice.

**Pipeline:** Voice → Speech-to-Text (Whisper) → AI text cleanup (FLAN-T5) →
editable review → Text-to-Speech (gTTS).

## How it works

1. **Record or upload** a voice message (microphone or file).
2. The app **transcribes** it to English text with OpenAI Whisper. Whisper runs
   in `translate` mode so the output is **always English**, even for non-English
   or hard-to-understand speech.
3. An optional **FLAN-T5** step cleans up grammar/clarity without changing the
   meaning.
4. **Review and edit** the corrected text.
5. Click **Generate Human Voice** to synthesize a natural English voice you can
   play, download or forward.

## Run locally

Requires **Python 3.10+** and **ffmpeg** (Whisper needs it).

```bash
# system dependency
sudo apt-get update && sudo apt-get install -y ffmpeg   # Debian/Ubuntu
# or: brew install ffmpeg                                # macOS

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python app.py
```

Then open the printed local URL (default `http://localhost:7860`).

### Configuration (optional)

| Env var            | Default                 | Description                          |
| ------------------ | ----------------------- | ------------------------------------ |
| `WHISPER_MODEL`    | `base`                  | Whisper size (`tiny`/`base`/`small`/`medium`/`large`). Larger = more accurate, slower. |
| `CORRECTION_MODEL` | `google/flan-t5-small`  | HF seq2seq model for text cleanup.   |

## Tests

```bash
pip install pytest
pytest -q
```

The tests mock the heavy models, so they run fast and need no GPU or network.

## Deployment

This repo is ready for **Hugging Face Spaces** (Gradio SDK). The YAML header at
the top of this file and `packages.txt` (ffmpeg) configure the Space
automatically — push the repo to a Space and it builds on its own.
