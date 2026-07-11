import gradio as gr
import os
import tempfile

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import whisper
from gtts import gTTS


# ==========================
# Load AI Models
# ==========================

print("Starting AI Voice Agent...")

whisper_model = whisper.load_model("base")

correction_model = None
correction_tokenizer = None

try:
    model_name = "google/flan-t5-small"
    correction_tokenizer = AutoTokenizer.from_pretrained(model_name)
    correction_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype="auto")
    print("Correction model loaded")

except Exception as e:
    correction_model = None
    correction_tokenizer = None
    print(f"Correction model failed: {e}")



# ==========================
# Speech To Text
# ==========================

def speech_to_text(audio):

    if audio is None:
        return "No audio detected"

    try:

        result = whisper_model.transcribe(
            audio,
            fp16=False
        )

        return result["text"].strip()

    except Exception as e:

        return f"Speech recognition error: {e}"



# ==========================
# AI Text Correction
# ==========================

def fix_text(text):

    if not text:
        return ""

    if correction_model is None or correction_tokenizer is None:
        return text

    try:
        prompt = (
            "Correct grammar and make this sentence clear "
            "without changing meaning: "
            + text
        )

        inputs = correction_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = correction_model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False
            )

        result = correction_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

    except Exception as e:
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

    if not text:
        return None

    audio_file = None
    try:
        audio_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )

        voice = gTTS(
            text=text,
            lang="en"
        )

        voice.save(audio_file.name)

        return audio_file.name


    except Exception as e:
        print(f"Text-to-speech error: {e}")
        if audio_file and os.path.exists(audio_file.name):
            os.unlink(audio_file.name)
        return None




# ==========================
# Gradio UI
# ==========================

with gr.Blocks(
    title="AI Voice Accessibility Agent"
) as demo:

    gr.Markdown(
        """
        # 🎤 AI Voice Accessibility Agent
        
        **Voice → Text → AI Correction → Human Voice**
        
        Helps users convert unclear speech into understandable communication.
        
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
                label="🎙️ Record or Upload Voice"
            )
            convert_btn = gr.Button(
                "🚀 Process Voice",
                variant="primary",
                size="lg"
            )

    with gr.Row():
        with gr.Column():
            raw_text = gr.Textbox(
                label="📝 Detected Speech",
                placeholder="Transcribed text will appear here...",
                lines=3
            )
        with gr.Column():
            corrected_text = gr.Textbox(
                label="✨ AI Corrected Message",
                placeholder="Corrected text will appear here... (editable)",
                lines=3,
                interactive=True
            )

    with gr.Row():
        confirm_btn = gr.Button(
            "🔊 Generate Human Voice",
            variant="secondary",
            size="lg"
        )

    output_voice = gr.Audio(
        label="🎧 Generated Human Voice",
        autoplay=False
    )

    gr.Markdown(
        """
        ---
        💡 **Tip**: For best results, speak clearly in a quiet environment.
        """
    )

    convert_btn.click(
        process_voice,
        inputs=audio_input,
        outputs=[raw_text, corrected_text]
    )

    confirm_btn.click(
        text_to_voice,
        inputs=corrected_text,
        outputs=output_voice
    )



if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 800px !important;
        }
        """
    )