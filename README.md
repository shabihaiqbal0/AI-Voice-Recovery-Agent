# 🎤 AI Voice Recovery Agent

Convert unclear, heavy, low-pitch, noisy or impaired speech into a clear,
natural **English** voice.

**Pipeline:** Voice → Speech-to-Text (Whisper) → AI text cleanup (FLAN-T5) →
editable review → Text-to-Speech (gTTS).

There are two interchangeable front-ends over the same core (`pipeline.py`):

| File               | UI        | Use for                                   |
| ------------------ | --------- | ----------------------------------------- |
| `streamlit_app.py` | Streamlit | Streamlit Community Cloud (default deploy)|
| `app.py`           | Gradio    | Local use / Hugging Face Spaces           |

## How it works

1. **Record or upload** a voice message.
2. The app **transcribes** it to English text with OpenAI Whisper. Whisper runs
   in `translate` mode so the output is **always English**, even for non-English
   or hard-to-understand speech.
3. An optional **FLAN-T5** step cleans up grammar/clarity without changing the
   meaning.
4. **Review and edit** the corrected text.
5. Generate a natural English voice you can play, download or forward.

## Run locally

Requires **Python 3.10+** and **ffmpeg** (Whisper needs it).

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg   # Debian/Ubuntu
# or: brew install ffmpeg                                # macOS

python -m venv .venv
source .venv/bin/activate

# Streamlit front-end (default):
pip install -r requirements.txt
streamlit run streamlit_app.py

# ...or the Gradio front-end:
pip install -r requirements-gradio.txt
python app.py
```

### Configuration (optional)

| Env var            | Default                | Description                                                              |
| ------------------ | ---------------------- | ----------------------------------------------------------------------- |
| `WHISPER_MODEL`    | `tiny`                 | Whisper size (`tiny`/`base`/`small`/`medium`/`large`). Larger = more accurate, slower, more memory. |
| `CORRECTION_MODEL` | `google/flan-t5-small` | HF seq2seq model for text cleanup.                                       |

`tiny` is the default to stay within free-hosting memory limits. Set
`WHISPER_MODEL=base` (or larger) locally for higher accuracy.

## Tests

```bash
pip install pytest
pytest -q
```

Tests mock the heavy models, so they run fast and need no GPU or network.

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub (done).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. **New app** → pick this repo, branch, and set **Main file path** to
   `streamlit_app.py`.
4. Deploy. `packages.txt` installs `ffmpeg` and `requirements.txt` installs the
   Python deps automatically.

One-click deploy link (replace branch if needed):
`https://share.streamlit.io/deploy?repository=shabihaiqbal0/AI-Voice-Recovery-Agent&branch=main&mainModule=streamlit_app.py`
