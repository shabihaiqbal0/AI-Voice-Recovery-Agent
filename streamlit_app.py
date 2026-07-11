import os
import tempfile

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import whisper
from gtts import gTTS


# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="AI Voice Accessibility Agent",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ==========================
# Load AI Models (cached)
# ==========================

@st.cache_resource
def load_models():
    """Load AI models with caching to avoid reloading on every interaction."""
    print("Loading AI models...")
    
    whisper_model = whisper.load_model("base")
    
    correction_model = None
    correction_tokenizer = None
    
    try:
        model_name = "google/flan-t5-small"
        correction_tokenizer = AutoTokenizer.from_pretrained(model_name)
        correction_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype="auto")
        print("Correction model loaded")
    except Exception as e:
        print(f"Correction model failed: {e}")
    
    return whisper_model, correction_model, correction_tokenizer


# Load models at startup
whisper_model, correction_model, correction_tokenizer = load_models()


# ==========================
# Speech To Text
# ==========================

def speech_to_text(audio_file):
    """Convert audio to text using Whisper."""
    if audio_file is None:
        return "No audio detected"
    
    try:
        result = whisper_model.transcribe(
            audio_file,
            fp16=False
        )
        return result["text"].strip()
    except Exception as e:
        return f"Speech recognition error: {e}"


# ==========================
# AI Text Correction
# ==========================

def fix_text(text):
    """Correct text using FLAN-T5 model."""
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
        st.error(f"Text correction error: {e}")
        return text


# ==========================
# Text To Human Voice
# ==========================

def text_to_voice(text):
    """Convert text to speech using gTTS."""
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
        st.error(f"Text-to-speech error: {e}")
        if audio_file and os.path.exists(audio_file.name):
            os.unlink(audio_file.name)
        return None


# ==========================
# Main UI
# ==========================

def main():
    st.title("🎤 AI Voice Accessibility Agent")
    st.markdown("**Voice → Text → AI Correction → Human Voice**")
    st.markdown("Helps users convert unclear speech into understandable communication.")
    
    st.markdown("---")
    
    st.markdown("### How to use:")
    st.markdown("""
    1. **Record or upload** your voice message
    2. **Click "Process Voice"** to transcribe and correct the text
    3. **Review and edit** the corrected message if needed
    4. **Click "Generate Human Voice"** to convert text to speech
    """)
    
    st.markdown("---")
    
    # Audio input
    audio_file = st.file_uploader("🎙️ Upload Voice Recording", type=['wav', 'mp3', 'ogg', 'flac', 'm4a'])
    
    # Process button
    if st.button("🚀 Process Voice", type="primary", use_container_width=True):
        if audio_file is not None:
            with st.spinner("Processing audio..."):
                # Save uploaded file to temp location
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_audio.write(audio_file.read())
                temp_audio.close()
                audio_path = temp_audio.name
                
                # Transcribe
                raw_text = speech_to_text(audio_path)
                
                # Correct
                corrected_text = fix_text(raw_text)
                
                # Clean up temp file
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("📝 Detected Speech", raw_text, height=100, disabled=True)
                with col2:
                    corrected = st.text_area("✨ AI Corrected Message", corrected_text, height=100, key="corrected")
        else:
            st.warning("Please record or upload an audio file first.")
    
    st.markdown("---")
    
    # Generate voice button
    if st.button("🔊 Generate Human Voice", type="secondary", use_container_width=True):
        corrected = st.session_state.get("corrected", "")
        if corrected:
            with st.spinner("Generating voice..."):
                audio_output = text_to_voice(corrected)
                if audio_output:
                    st.audio(audio_output, format="audio/mp3")
                    # Clean up temp file
                    if os.path.exists(audio_output):
                        os.unlink(audio_output)
        else:
            st.warning("Please process voice first to generate corrected text.")
    
    st.markdown("---")
    st.info("💡 **Tip**: For best results, speak clearly in a quiet environment.")


if __name__ == "__main__":
    main()
