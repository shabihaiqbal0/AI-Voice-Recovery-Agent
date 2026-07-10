"""AI Voice Recovery Agent.

Pipeline: Voice -> Speech-to-Text (Whisper) -> AI text cleanup (FLAN-T5)
-> editable review -> Text-to-Speech (gTTS).

The transcription is forced to English so the displayed and spoken output
is always English, regardless of the input language or speech clarity.
"""

import os
import tempfile
from functools import lru_cache

import gradio as gr
import torch
import whisper
from gtts import gTTS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ==========================
# Configuration
# ==========================

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
CORRECTION_MODEL_NAME = os.getenv("CORRECTION_MODEL", "google/flan-t5-small")


# ==========================
# Lazy model loading (kept out of import time so tests stay fast/offline)
# ==========================

@lru_cache(maxsize=1)
def get_whisper_model():
    print(f"Loading Whisper model '{WHISPER_MODEL_NAME}'...")
    model = whisper.load_model(WHISPER_MODEL_NAME)
    print("Whisper model loaded")
    return model


@lru_cache(maxsize=1)
def get_correction_model():
    """Return (tokenizer, model) or (None, None) if loading fails."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(CORRECTION_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            CORRECTION_MODEL_NAME, torch_dtype="auto"
        )
        print("Correction model loaded")
        return tokenizer, model
    except Exception as e:  # noqa: BLE001 - model download is best-effort
        print(f"Correction model failed to load, continuing without it: {e}")
        return None, None


# ==========================
# Speech To Text
# ==========================

def speech_to_text(audio):
    """Transcribe an audio file to English text using Whisper.

    ``task="translate"`` makes Whisper always emit English, so unclear,
    heavy, low-pitch or non-English speech is still returned in English.
    """
    if not audio:
        return ""

    try:
        result = get_whisper_model().transcribe(
            audio,
            task="translate",
            fp16=torch.cuda.is_available(),
        )
        return result["text"].strip()
    except Exception as e:  # noqa: BLE001 - surface errors to the UI
        return f"Speech recognition error: {e}"


# ==========================
# AI Text Correction
# ==========================

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
        print(f"Text correction error: {e}")
        return text


# ==========================
# Main AI Pipeline
# ==========================

def process_voice(audio):
    raw = speech_to_text(audio)
    corrected = fix_text(raw)
    return raw, corrected


# ==========================
# Text To Human Voice
# ==========================

def text_to_voice(text):
    """Convert approved English text into a natural human-like voice."""
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
        print(f"Text-to-speech error: {e}")
        if audio_file and os.path.exists(audio_file.name):
            os.unlink(audio_file.name)
        return None


# ==========================
# Gradio UI
# ==========================

def build_demo():
    with gr.Blocks(
        title="AI Voice Recovery Agent",
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 900px !important; }",
    ) as demo:
        gr.Markdown(
            """
            # 🎤 AI Voice Recovery Agent

            **Voice → Text → AI Correction → Human Voice**

            Helps users convert unclear, heavy, low-pitch or impaired speech
            into a clear, natural English voice.

            ---
            ### How to use:
            1. **Record or upload** your voice message
            2. **Click "Process Voice"** to transcribe and correct the text
            3. **Review and edit** the corrected message if needed
            4. **Click "Generate Human Voice"** to convert text to speech
            """
        )

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="🎙️ Record or Upload Voice",
                )
                convert_btn = gr.Button(
                    "🚀 Process Voice", variant="primary", size="lg"
                )

        with gr.Row():
            with gr.Column():
                raw_text = gr.Textbox(
                    label="📝 Detected Speech",
                    placeholder="Transcribed text will appear here...",
                    lines=3,
                )
            with gr.Column():
                corrected_text = gr.Textbox(
                    label="✨ AI Corrected Message (editable)",
                    placeholder="Corrected text will appear here...",
                    lines=3,
                    interactive=True,
                )

        with gr.Row():
            confirm_btn = gr.Button(
                "🔊 Generate Human Voice", variant="secondary", size="lg"
            )

        output_voice = gr.Audio(label="🎧 Generated Human Voice", autoplay=False)

        gr.Markdown(
            """
            ---
            💡 **Tip**: For best results, speak clearly in a quiet environment.
            """
        )

        convert_btn.click(
            process_voice, inputs=audio_input, outputs=[raw_text, corrected_text]
        )
        confirm_btn.click(
            text_to_voice, inputs=corrected_text, outputs=output_voice
        )

    return demo


if __name__ == "__main__":
    print("Starting AI Voice Recovery Agent...")
    build_demo().launch(server_name="0.0.0.0")
