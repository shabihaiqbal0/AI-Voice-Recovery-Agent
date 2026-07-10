"""Streamlit front-end for the AI Voice Recovery Agent.

Voice -> Text -> AI Correction -> Human Voice. Core logic lives in
``pipeline.py``. This is the entrypoint used for Streamlit Community Cloud.
"""

import os
import tempfile
import warnings

import streamlit as st

# Suppress warnings early
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set environment variables for optimization
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pipeline import process_voice, text_to_voice

st.set_page_config(
    page_title="AI Voice Recovery Agent",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("🎤 AI Voice Recovery Agent")
st.caption("Voice → Text → AI Correction → Human Voice")
st.markdown(
    "Convert unclear, heavy, low-pitch, noisy or impaired speech into a clear, "
    "natural **English** voice."
)

for key in ("raw", "corrected"):
    st.session_state.setdefault(key, "")


def _save_upload(uploaded) -> str:
    suffix = "." + uploaded.name.rsplit(".", 1)[-1] if getattr(
        uploaded, "name", None
    ) and "." in uploaded.name else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
        fh.write(uploaded.getvalue())
        return fh.name


st.subheader("1. Record or upload your voice")
rec_tab, upload_tab = st.tabs(["🎙️ Record", "📁 Upload"])
audio = None
with rec_tab:
    recorded = st.audio_input("Record a voice message")
    if recorded is not None:
        audio = recorded
with upload_tab:
    uploaded = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
    )
    if uploaded is not None:
        audio = uploaded

if st.button("🚀 Process Voice", type="primary", disabled=audio is None):
    path = _save_upload(audio)
    with st.spinner("Transcribing and correcting..."):
        try:
            raw, corrected = process_voice(path)
            st.session_state.raw = raw
            st.session_state.corrected = corrected
            st.success("✅ Voice processed successfully!")
        except Exception as e:
            st.error(f"❌ Error processing voice: {str(e)}")

if st.session_state.raw:
    st.subheader("2. Review & edit the text")
    st.text_area("📝 Detected Speech", value=st.session_state.raw, disabled=True)
    st.session_state.corrected = st.text_area(
        "✨ AI Corrected Message (editable)",
        value=st.session_state.corrected,
    )

    st.subheader("3. Generate the human voice")
    if st.button("🔊 Generate Human Voice", type="secondary"):
        with st.spinner("Generating natural voice..."):
            try:
                mp3_path = text_to_voice(st.session_state.corrected)
                if mp3_path:
                    st.audio(mp3_path, format="audio/mp3")
                    with open(mp3_path, "rb") as fh:
                        st.download_button(
                            "⬇️ Download voice (MP3)",
                            data=fh,
                            file_name="recovered_voice.mp3",
                            mime="audio/mpeg",
                        )
                    st.success("✅ Voice generated successfully!")
                else:
                    st.error("Could not generate audio. Please try again.")
            except Exception as e:
                st.error(f"❌ Error generating voice: {str(e)}")

st.divider()
st.caption("💡 Tip: for best results, speak clearly in a quiet environment.")
