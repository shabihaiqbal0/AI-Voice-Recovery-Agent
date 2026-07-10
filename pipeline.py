"""Core Voice Recovery pipeline (UI-agnostic).

Voice -> Speech-to-Text (Whisper, translate->English) -> AI text cleanup
(FLAN-T5) -> Text-to-Speech (gTTS).

Both the Gradio (`app.py`) and Streamlit (`streamlit_app.py`) front-ends import
from this module so the logic lives in one place.
"""

import os
import tempfile
import warnings
from functools import lru_cache

import torch

# Suppress warnings early
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set environment variables for optimization
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import whisper
from gtts import gTTS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Whisper "tiny" keeps memory low enough for free hosting (e.g. Streamlit Cloud).
# Override with WHISPER_MODEL=base/small/medium/large for more accuracy.
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
CORRECTION_MODEL_NAME = os.getenv("CORRECTION_MODEL", "google/flan-t5-small")

# Ensure CPU-only torch on Streamlit Cloud
device = "cpu"
print(f"Using device: {device}")


@lru_cache(maxsize=1)
def get_whisper_model():
    try:
        print(f"Loading Whisper model '{WHISPER_MODEL_NAME}'...")
        model = whisper.load_model(WHISPER_MODEL_NAME, device=device)
        print("✓ Whisper model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Failed to load Whisper model: {e}")
        raise


@lru_cache(maxsize=1)
def get_correction_model():
    """Return (tokenizer, model) or (None, None) if loading fails."""
    try:
        print(f"Loading correction model '{CORRECTION_MODEL_NAME}'...")
        tokenizer = AutoTokenizer.from_pretrained(CORRECTION_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            CORRECTION_MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        print("✓ Correction model loaded successfully")
        return tokenizer, model
    except Exception as e:  # noqa: BLE001 - model download is best-effort
        print(f"⚠ Correction model failed to load, continuing without it: {e}")
        return None, None


def speech_to_text(audio):
    """Transcribe an audio file to English text using Whisper.

    ``task="translate"`` makes Whisper always emit English, so unclear,
    heavy, low-pitch or non-English speech is still returned in English.
    """
    if not audio:
        return ""

    try:
        model = get_whisper_model()
        result = model.transcribe(
            audio,
            task="translate",
            fp16=False,  # Disable fp16 on CPU
        )
        return result["text"].strip()
    except Exception as e:  # noqa: BLE001 - surface errors to the UI
        error_msg = f"Speech recognition error: {str(e)}"
        print(f"✗ {error_msg}")
        return error_msg


def fix_text(text):
    """Clean up grammar/clarity without changing meaning."""
    if not text:
        return ""

    tokenizer, model = get_correction_model()
    if model is None or tokenizer is None:
        return text

    try:
        prompt = (
            "Correct grammar and make this sentence clear "
            "without changing meaning: " + text
        )

        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip() or text
    except Exception as e:  # noqa: BLE001 - fall back to raw text
        print(f"⚠ Text correction error: {e}")
        return text


def process_voice(audio):
    """Full transcription + correction step. Returns (raw, corrected)."""
    raw = speech_to_text(audio)
    corrected = fix_text(raw)
    return raw, corrected


def text_to_voice(text):
    """Convert approved English text into a natural human-like voice (mp3 path)."""
    if not text or not text.strip():
        return None

    audio_file = None
    try:
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        audio_file.close()

        voice = gTTS(text=text, lang="en")
        voice.save(audio_file.name)

        return audio_file.name
    except Exception as e:  # noqa: BLE001 - surface TTS failures gracefully
        print(f"✗ Text-to-speech error: {e}")
        if audio_file and os.path.exists(audio_file.name):
            try:
                os.unlink(audio_file.name)
            except Exception as cleanup_error:
                print(f"⚠ Failed to clean up temp file: {cleanup_error}")
        return None
