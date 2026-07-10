"""Gradio front-end for the AI Voice Recovery Agent.

Voice -> Text -> AI Correction -> Human Voice. Core logic lives in
``pipeline.py`` (shared with the Streamlit front-end).
"""

import gradio as gr

from pipeline import process_voice, text_to_voice


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
    print("Starting AI Voice Recovery Agent (Gradio)...")
    build_demo().launch(server_name="0.0.0.0")
